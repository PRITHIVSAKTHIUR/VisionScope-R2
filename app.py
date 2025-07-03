import os
import random
import uuid
import json
import requests
import time
import asyncio
from threading import Thread

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

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Behemoth-3B-070225-post0.1
MODEL_ID_N = "prithivMLmods/Behemoth-3B-070225-post0.1"
processor_n = AutoProcessor.from_pretrained(MODEL_ID_N, trust_remote_code=True)
model_n = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_N,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load SkyCaptioner-V1
MODEL_ID_M = "Skywork/SkyCaptioner-V1"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Space Thinker
MODEL_ID_Z = "remyxai/SpaceThinker-Qwen2.5VL-3B"
processor_z = AutoProcessor.from_pretrained(MODEL_ID_Z, trust_remote_code=True)
model_z = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Z,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load coreOCR-7B-050325-preview
MODEL_ID_K = "prithivMLmods/coreOCR-7B-050325-preview"
processor_k = AutoProcessor.from_pretrained(MODEL_ID_K, trust_remote_code=True)
model_k = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_K,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load remyxai/SpaceOm
MODEL_ID_Y = "remyxai/SpaceOm"
processor_y = AutoProcessor.from_pretrained(MODEL_ID_Y, trust_remote_code=True)
model_y = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Y,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

#video sampling
def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
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

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for image input.
    """
    if model_name == "SkyCaptioner-V1":
        processor = processor_m
        model = model_m
    elif model_name == "Behemoth-3B-070225-post0.1":
        processor = processor_n
        model = model_n
    elif model_name == "SpaceThinker-3B":
        processor = processor_z
        model = model_z
    elif model_name == "coreOCR-7B-050325-preview":
        processor = processor_k
        model = model_k
    elif model_name == "SpaceOm-3B":
        processor = processor_y
        model = model_y
    else:
        yield "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
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
        yield buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for video input.
    """
    if model_name == "SkyCaptioner-V1":
        processor = processor_m
        model = model_m
    elif model_name == "Behemoth-3B-070225-post0.1":
        processor = processor_n
        model = model_n
    elif model_name == "SpaceThinker-3B":
        processor = processor_z
        model = model_z
    elif model_name == "coreOCR-7B-050325-preview":
        processor = processor_k
        model = model_k
    elif model_name == "SpaceOm-3B":
        processor = processor_y
        model = model_y
    else:
        yield "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video."
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
        truncation=False,
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
        yield buffer

# Define examples for image and video inference
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

css = """
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[VisionScope R2](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=image_examples,
                        inputs=[image_query, image_upload]
                    )
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=video_examples,
                        inputs=[video_query, video_upload]
                    )
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
        with gr.Column():
            output = gr.Textbox(label="Output", interactive=False, lines=2, scale=2)
            model_choice = gr.Radio(
                choices=["SkyCaptioner-V1", "Behemoth-3B-070225-post0.1", "SpaceThinker-3B", "coreOCR-7B-050325-preview", "SpaceOm-3B"],
                label="Select Model",
                value="SkyCaptioner-V1"
            )
            
            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/VisionScope-R2/discussions)")
            gr.Markdown("> [SkyCaptioner-V1](https://huggingface.co/Skywork/SkyCaptioner-V1):  structural video captioning model designed to generate high-quality, structural descriptions for video data. It integrates specialized sub-expert models.")
            gr.Markdown("> [SpaceThinker-Qwen2.5VL-3B](https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B): thinking/reasoning multimodal/vision-language model (VLM) trained to enhance spatial reasoning.")
            gr.Markdown("> [coreOCR-7B-050325-preview](https://huggingface.co/prithivMLmods/coreOCR-7B-050325-preview): model is a fine-tuned version of qwen/qwen2-vl-7b, optimized for document-level optical character recognition (ocr), long-context vision-language understanding.")  
            gr.Markdown("> [SpaceOm](https://huggingface.co/remyxai/SpaceOm): SpaceOm, the reasoning traces in the spacethinker dataset average ~200 thinking tokens, so now included longer reasoning traces in the training data to help the model use more tokens in reasoning.")             

    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=output
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=output
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)
