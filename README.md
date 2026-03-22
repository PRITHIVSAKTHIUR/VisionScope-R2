# **VisionScope-R2**

VisionScope-R2 is an experimental, highly versatile vision suite designed for advanced image inference, spatial reasoning, and complex scene understanding. Built upon the powerful Qwen2.5-VL and Qwen2-VL architectures, this application offers a modern, interactive web interface for processing a wide variety of visual inputs, including documents, receipts, screenshots, and complex real-world scenes. By integrating a diverse roster of state-of-the-art vision-language models—ranging from specialized captioners to spatial thinkers and OCR engines—the tool allows users to seamlessly extract text, estimate distances, and generate highly detailed scene descriptions. Fully GPU-accelerated and optimized for performance, VisionScope-R2 provides researchers and developers with granular control over text generation parameters, creating an ideal environment for testing and deploying next-generation vision-based AI workflows.

<img width="1920" height="1798" alt="Screenshot 2026-03-22 at 12-09-32 VisionScope-R2 - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/a89ef395-02ea-405e-9294-39641a990ba1" />

### **Key Features**

* **Multi-Model Architecture:** Seamlessly switch between specialized vision-language models directly from the interface. Supported models include `DeepCaption-VLA-7B`, `SkyCaptioner-V1`, `SpaceThinker-3B`, `coreOCR-7B-050325-preview`, and `SpaceOm-3B`.
* **Custom User Interface:** Features a bespoke, responsive Gradio frontend built with custom HTML, CSS, and JavaScript. It includes a drag-and-drop media zone, real-time output streaming, and an integrated advanced settings panel.
* **Granular Inference Controls:** Fine-tune the AI's output by adjusting generation parameters such as Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Output Management:** Built-in actions allow users to instantly copy the raw output text to their clipboard or save the generated response directly as a `.txt` file.
* **Flash Attention 2 Integration:** Utilizes `kernels-community/flash-attn2` for optimized, memory-efficient inference on compatible GPUs.

### **Repository Structure**

```text
├── images/
│   ├── 1.jpg
│   ├── 2.jpeg
│   ├── 3.png
│   ├── 4.png
│   └── 5.jpg
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run VisionScope-R2 locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.8.0
torchvision
matplotlib
requests
kernels
hf_xet
spaces
pillow
gradio
av
```

### **Usage**

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the models. Note that the selected models will be downloaded and loaded into VRAM upon their first invocation.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/VisionScope-R2.git](https://github.com/PRITHIVSAKTHIUR/VisionScope-R2.git)
