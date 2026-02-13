import os
import random
import uuid
import json
import requests
import time
import asyncio
from threading import Thread
from typing import Iterable

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
)
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.2em !important;
}

/* RadioAnimated Styles */
.ra-wrap{ width: fit-content; }
.ra-inner{
  position: relative; display: inline-flex; align-items: center; gap: 0; padding: 6px;
  background: var(--neutral-200); border-radius: 9999px; overflow: hidden;
}
.ra-input{ display: none; }
.ra-label{
  position: relative; z-index: 2; padding: 8px 16px;
  font-family: inherit; font-size: 14px; font-weight: 600;
  color: var(--neutral-500); cursor: pointer; transition: color 0.2s; white-space: nowrap;
}
.ra-highlight{
  position: absolute; z-index: 1; top: 6px; left: 6px;
  height: calc(100% - 12px); border-radius: 9999px;
  background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.2s, width 0.2s;
}
.ra-input:checked + .ra-label{ color: black; }

/* Dark mode adjustments for Radio */
.dark .ra-inner { background: var(--neutral-800); }
.dark .ra-label { color: var(--neutral-400); }
.dark .ra-highlight { background: var(--neutral-600); }
.dark .ra-input:checked + .ra-label { color: white; }

#gpu-duration-container {
    padding: 10px;
    border-radius: 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    margin-top: 10px;
}
"""

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RadioAnimated(gr.HTML):
    def __init__(self, choices, value=None, **kwargs):
        if not choices or len(choices) < 2:
            raise ValueError("RadioAnimated requires at least 2 choices.")
        if value is None:
            value = choices[0]

        uid = uuid.uuid4().hex[:8]
        group_name = f"ra-{uid}"

        inputs_html = "\n".join(
            f"""
            <input class="ra-input" type="radio" name="{group_name}" id="{group_name}-{i}" value="{c}">
            <label class="ra-label" for="{group_name}-{i}">{c}</label>
            """
            for i, c in enumerate(choices)
        )

        html_template = f"""
        <div class="ra-wrap" data-ra="{uid}">
          <div class="ra-inner">
            <div class="ra-highlight"></div>
            {inputs_html}
          </div>
        </div>
        """

        js_on_load = r"""
        (() => {
          const wrap = element.querySelector('.ra-wrap');
          const inner = element.querySelector('.ra-inner');
          const highlight = element.querySelector('.ra-highlight');
          const inputs = Array.from(element.querySelectorAll('.ra-input'));

          if (!inputs.length) return;

          const choices = inputs.map(i => i.value);

          function setHighlightByIndex(idx) {
            const n = choices.length;
            const pct = 100 / n;
            highlight.style.width = `calc(${pct}% - 6px)`;
            highlight.style.transform = `translateX(${idx * 100}%)`;
          }

          function setCheckedByValue(val, shouldTrigger=false) {
            const idx = Math.max(0, choices.indexOf(val));
            inputs.forEach((inp, i) => { inp.checked = (i === idx); });
            setHighlightByIndex(idx);

            props.value = choices[idx];
            if (shouldTrigger) trigger('change', props.value);
          }

          setCheckedByValue(props.value ?? choices[0], false);

          inputs.forEach((inp) => {
            inp.addEventListener('change', () => {
              setCheckedByValue(inp.value, true);
            });
          });
        })();
        """

        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            **kwargs
        )

def apply_gpu_duration(val: str):
    return int(val)

MODEL_ID_N = "prithivMLmods/DeepCaption-VLA-7B"
processor_n = AutoProcessor.from_pretrained(MODEL_ID_N, trust_remote_code=True)
model_n = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_N,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_M = "Skywork/SkyCaptioner-V1"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_Z = "remyxai/SpaceThinker-Qwen2.5VL-3B"
processor_z = AutoProcessor.from_pretrained(MODEL_ID_Z, trust_remote_code=True)
model_z = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Z,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_K = "prithivMLmods/coreOCR-7B-050325-preview"
processor_k = AutoProcessor.from_pretrained(MODEL_ID_K, trust_remote_code=True)
model_k = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_K,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_Y = "remyxai/SpaceOm"
processor_y = AutoProcessor.from_pretrained(MODEL_ID_Y, trust_remote_code=True)
model_y = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Y,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

def calc_timeout_image(model_name: str, text: str, image: Image.Image,
                       max_new_tokens: int, temperature: float, top_p: float,
                       top_k: int, repetition_penalty: float, gpu_timeout: int):
    """Calculate GPU timeout duration for image inference."""
    try:
        return int(gpu_timeout)
    except:
        return 60

def calc_timeout_video(model_name: str, text: str, video_path: str,
                       max_new_tokens: int, temperature: float, top_p: float,
                       top_k: int, repetition_penalty: float, gpu_timeout: int):
    """Calculate GPU timeout duration for video inference."""
    try:
        return int(gpu_timeout)
    except:
        return 60
        
@spaces.GPU(duration=calc_timeout_image)
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2,
                   gpu_timeout: int = 60):
    """
    Generates responses using the selected model for image input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "SkyCaptioner-V1":
        processor, model = processor_m, model_m
    elif model_name == "DeepCaption-VLA-7B":
        processor, model = processor_n, model_n
    elif model_name == "SpaceThinker-3B":
        processor, model = processor_z, model_z
    elif model_name == "coreOCR-7B-050325-preview":
        processor, model = processor_k, model_k
    elif model_name == "SpaceOm-3B":
        processor, model = processor_y, model_y
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU(duration=calc_timeout_video)
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2,
                   gpu_timeout: int = 90):
    """
    Generates responses using the selected model for video input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "SkyCaptioner-V1":
        processor, model = processor_m, model_m
    elif model_name == "DeepCaption-VLA-7B":
        processor, model = processor_n, model_n
    elif model_name == "SpaceThinker-3B":
        processor, model = processor_z, model_z
    elif model_name == "coreOCR-7B-050325-preview":
        processor, model = processor_k, model_k
    elif model_name == "SpaceOm-3B":
        processor, model = processor_y, model_y
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

image_examples = [
    ["type out the messy hand-writing as accurately as you can.", "images/1.jpg"],
    ["count the number of birds and explain the scene in detail.", "images/2.jpeg"],
    ["how far is the Goal from the penalty taker in this image?.", "images/3.png"],
    ["approximately how many meters apart are the chair and bookshelf?.", "images/4.png"],
    ["how far is the man in the red hat from the pallet of boxes in feet?.", "images/5.jpg"],
]

video_examples = [
    ["give the highlights of the movie scene video.", "videos/1.mp4"],
    ["explain the advertisement in detail.", "videos/2.mp4"]
]

with gr.Blocks() as demo:
    gr.Markdown("# **VisionScope R2**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Upload Image", height=290)
                    image_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=image_examples, inputs=[image_query, image_upload])
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Upload Video(<= 30s)", height=290)
                    video_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=video_examples, inputs=[video_query, video_upload])
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            output = gr.Textbox(label="Raw Output Stream", interactive=True, lines=11)
            with gr.Accordion("(Result.md)", open=False):
                markdown_output = gr.Markdown(label="Formatted Result")
            model_choice = gr.Radio(
                choices=["DeepCaption-VLA-7B", "SkyCaptioner-V1", "SpaceThinker-3B", "coreOCR-7B-050325-preview", "SpaceOm-3B"],
                label="Select Model",
                value="DeepCaption-VLA-7B"
            )
            
            with gr.Row(elem_id="gpu-duration-container"):
                with gr.Column():
                    gr.Markdown("**GPU Duration (seconds)**")
                    radioanimated_gpu_duration = RadioAnimated(
                        choices=["60", "90", "120", "180", "240", "300"],
                        value="60",
                        elem_id="radioanimated_gpu_duration"
                    )
                    gpu_duration_state = gr.Number(value=60, visible=False)

            gr.Markdown("*Note: Higher GPU duration allows for longer processing but consumes more GPU quota.*")

    radioanimated_gpu_duration.change(
        fn=apply_gpu_duration, 
        inputs=radioanimated_gpu_duration, 
        outputs=[gpu_duration_state], 
        api_visibility="private"
    )

    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty, gpu_duration_state],
        outputs=[output, markdown_output]
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty, gpu_duration_state],
        outputs=[output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(css=css, theme=steel_blue_theme, mcp_server=True, ssr_mode=False, show_error=True)
