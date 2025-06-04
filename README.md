# VisionScope-R2

A comprehensive multi-modal AI application that integrates four specialized vision-language models for advanced image and video analysis. VisionScope-R2 offers OCR capabilities, spatial reasoning, handwriting recognition, and structural video captioning through an intuitive Gradio interface.

## Features

- **Multi-Model Architecture**: Four specialized models optimized for different vision tasks
- **Image Analysis**: Advanced OCR, handwriting recognition, and spatial reasoning
- **Video Processing**: Structural video captioning and scene analysis
- **Real-time Streaming**: Progressive response generation for immediate feedback
- **Advanced Controls**: Fine-tunable parameters for optimal performance
- **Comprehensive Examples**: Pre-loaded examples for quick testing

## Supported Models

### 1. SkyCaptioner-V1 (Skywork)
- **Purpose**: Structural video captioning with specialized sub-expert models
- **Best for**: High-quality video descriptions and scene understanding
- **Model**: `Skywork/SkyCaptioner-V1`
- **Architecture**: Qwen2.5-VL based

### 2. SpaceThinker-3B (RemyxAI)
- **Purpose**: Enhanced spatial reasoning and multimodal thinking
- **Best for**: Distance estimation, spatial relationships, and geometric analysis
- **Model**: `remyxai/SpaceThinker-Qwen2.5VL-3B`
- **Architecture**: Qwen2.5-VL 3B parameters

### 3. CoreOCR-7B (Preview)
- **Purpose**: Document-level optical character recognition
- **Best for**: Long-context document understanding and text extraction
- **Model**: `prithivMLmods/coreOCR-7B-050325-preview`
- **Architecture**: Qwen2-VL 7B based

### 4. Imgscope-OCR-2B
- **Purpose**: Specialized handwriting and mathematical content recognition
- **Best for**: Messy handwriting, mathematical equations with LaTeX formatting
- **Model**: `prithivMLmods/Imgscope-OCR-2B-0527`
- **Architecture**: Qwen2-VL 2B Instruct

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- 25GB+ free disk space for models

### Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install gradio
pip install spaces
pip install opencv-python
pip install pillow
pip install numpy
pip install requests
```

### Clone Repository

```bash
git clone https://github.com/PRITHIVSAKTHIUR/DocScope-R1.git
cd DocScope-R1
```

## Usage

### Running the Application

```bash
python app.py
```

The application will start and provide you with a local URL (typically `http://127.0.0.1:7860`) to access the web interface.

### Image Analysis

1. Select the "Image Inference" tab
2. Enter your query in the text box
3. Upload an image
4. Choose your preferred model based on your task
5. Adjust advanced parameters if needed
6. Click "Submit"

**Example Use Cases:**
- **Handwriting Recognition**: "Type out the messy hand-writing as accurately as you can"
- **Object Counting**: "Count the number of birds and explain the scene in detail"
- **Spatial Reasoning**: "How far is the Goal from the penalty taker in this image?"
- **Distance Estimation**: "Approximately how many meters apart are the chair and bookshelf?"
- **Complex Scene Analysis**: "How far is the man in the red hat from the pallet of boxes in feet?"

### Video Analysis

1. Select the "Video Inference" tab
2. Enter your query describing what you want to analyze
3. Upload a video file
4. Select the appropriate model (SkyCaptioner-V1 recommended for videos)
5. Configure generation parameters
6. Click "Submit"

**Example Use Cases:**
- **Movie Scene Analysis**: "Give the highlights of the movie scene video"
- **Advertisement Analysis**: "Explain the advertisement in detail"
- **Action Recognition**: "Describe the sequence of events in the video"

## Model Selection Guide

| Task Type | Recommended Model | Use Case |
|-----------|------------------|----------|
| Handwritten Text | Imgscope-OCR-2B | Messy handwriting, math equations |
| Document OCR | CoreOCR-7B | Clean text, documents, long context |
| Spatial Analysis | SpaceThinker-3B | Distance, positioning, geometry |
| Video Content | SkyCaptioner-V1 | Scene description, video analysis |

## Configuration

### Advanced Parameters

- **Max New Tokens** (1-2048): Maximum length of generated response
- **Temperature** (0.1-4.0): Controls creativity and randomness
- **Top-p** (0.05-1.0): Nucleus sampling for diverse outputs
- **Top-k** (1-1000): Vocabulary limitation per generation step
- **Repetition Penalty** (1.0-2.0): Prevents repetitive content

### Environment Variables

- `MAX_INPUT_TOKEN_LENGTH`: Maximum input context length (default: 4096)

## Technical Architecture

### Video Processing Pipeline

Videos are automatically processed through the following steps:
1. Frame extraction (10 evenly spaced frames)
2. Timestamp annotation for each frame
3. Sequential processing with context preservation
4. Comprehensive scene understanding across temporal dimension

### Model Architecture Details

- **Mixed Precision**: All models use float16 for memory efficiency
- **GPU Acceleration**: CUDA optimization with automatic fallback to CPU
- **Streaming Generation**: Real-time text streaming for immediate feedback
- **Memory Management**: Efficient GPU memory utilization across multiple models

### Performance Optimizations

- Single model loading at startup to reduce initialization time
- Automatic device detection and optimal resource allocation
- Streaming responses for better user experience
- Smart buffer management to prevent token overflow

## System Requirements

### Minimum Requirements
- **GPU**: 12GB VRAM (RTX 3060 Ti or equivalent)
- **RAM**: 16GB system memory
- **Storage**: 30GB free space (SSD recommended)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

### Recommended Requirements
- **GPU**: 16GB+ VRAM (RTX 4080 or better)
- **RAM**: 32GB system memory
- **Storage**: SSD with 50GB free space
- **CPU**: High-performance processor (Intel i7/AMD Ryzen 7 or better)

## File Structure

```
VisionScope-R2/
├── app.py              # Main application file
├── README.md           # This documentation
├── requirements.txt    # Python dependencies
├── images/            # Example images
│   ├── 1.jpg          # Handwriting sample
│   ├── 2.jpeg         # Bird counting example
│   ├── 3.png          # Sports field analysis
│   ├── 4.png          # Indoor scene measurement
│   └── 5.jpg          # Distance estimation
└── videos/            # Example videos
    ├── 1.mp4          # Movie scene
    └── 2.mp4          # Advertisement sample
```

## Advanced Features

### Multi-Modal Understanding
- Simultaneous processing of text and visual information
- Context-aware responses based on image content
- Cross-modal reasoning capabilities

### Specialized OCR Capabilities
- Handwritten text recognition with high accuracy
- Mathematical equation parsing with LaTeX output
- Document structure understanding
- Multi-language text support

### Spatial Intelligence
- Distance estimation between objects
- Geometric relationship analysis
- 3D scene understanding from 2D images
- Perspective-aware measurements

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce max_new_tokens to 512 or lower
- Use smaller models (Imgscope-OCR-2B, SpaceThinker-3B)
- Enable CPU inference mode
- Close other GPU-intensive applications

**Model Loading Errors**
- Verify internet connection for initial downloads
- Check Hugging Face Hub access
- Ensure sufficient disk space (30GB+)
- Clear Hugging Face cache if corrupted

**Poor OCR Performance**
- Use CoreOCR-7B for clean document text
- Use Imgscope-OCR-2B for handwritten content
- Ensure image resolution is adequate (minimum 300 DPI recommended)
- Check image quality and contrast

**Video Processing Issues**
- Supported formats: MP4, AVI, MOV, MKV
- Maximum recommended video length: 5 minutes
- Ensure video file is not corrupted
- Check available system memory during processing

### Performance Optimization Tips

1. **Model Selection**: Choose the smallest suitable model for your task
2. **Image Preprocessing**: Resize large images before upload
3. **Batch Processing**: Process multiple similar images with the same model
4. **Memory Management**: Restart application periodically for long sessions

## API Integration

The application can be extended with API endpoints for programmatic access:

```python
# Example API call structure
response = generate_image(
    model_name="SpaceThinker-3B",
    text="How far apart are these objects?",
    image=your_image,
    max_new_tokens=512
)
```

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper testing
4. Update documentation as needed
5. Submit a pull request with detailed description

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include error handling for edge cases
- Test with multiple model configurations

## Future Enhancements

- **Additional Models**: Integration of more specialized vision models
- **Batch Processing**: Support for multiple image/video processing
- **API Endpoints**: RESTful API for external integrations
- **Model Quantization**: Support for INT8 quantization for faster inference
- **Cloud Integration**: Support for cloud-based model hosting

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **Skywork Team**: For the SkyCaptioner-V1 model
- **RemyxAI**: For the SpaceThinker spatial reasoning model
- **Qwen Team**: For the foundational architecture
- **Hugging Face**: For the transformers library and model hosting
- **Gradio Team**: For the user interface framework

## Citation

If you use VisionScope-R2 in your research, please cite:

```bibtex
@software{visionscope_r2,
  title={VisionScope-R2: Multi-Modal Vision-Language Analysis Platform},
  author={PRITHIVSAKTHIUR},
  year={2024},
  url={https://github.com/PRITHIVSAKTHIUR/DocScope-R1}
}
```

## Contact

For questions, issues, or collaborations:
- **GitHub Issues**: Open an issue for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact the maintainer through GitHub profile

---

**Note**: This application requires significant computational resources. Ensure your system meets the minimum requirements and has adequate cooling for extended usage.
