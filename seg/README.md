# Iris Segmentation - Improved Version

This improved project includes:
- Multi-dataset balanced training (WeightedRandomSampler) to handle domain gaps.
- Stronger augmentations with `albumentations`.
- CBAM attention and an ASPP-like module in the decoder to improve robustness.
- Combined BCEWithLogits + Dice loss + Boundary loss.
- Cosine scheduler with linear warmup and optional mixed precision training (AMP).
- Optionally freeze backbone for initial epochs to stabilize training across datasets.

## Installation

Install the required dependencies:
```
pip install -r requirements.txt
```

To use ONNX models for inference, you also need to install ONNX Runtime:

For CPU-only inference:
```
pip install onnxruntime
```

For GPU-accelerated inference (requires CUDA):
```
pip install onnxruntime-gpu==1.20.1
```

**Note for CUDA/cuDNN Requirements:**
- This project requires CUDA 12.x and cuDNN 9.x for GPU acceleration with ONNX Runtime.
- You must have compatible NVIDIA GPU drivers installed.
- If you encounter `LoadLibrary failed with error 126` or `Failed to create CUDAExecutionProvider`, please ensure you have the correct versions of CUDA and cuDNN installed.
- You may also need to install or update the latest Microsoft Visual C++ Redistributable packages.

If you encounter an error like `ImportError: ONNX model specified but onnxruntime is not installed`, please ensure you have installed the appropriate onnxruntime package:

For CPU-only:
```
pip install onnxruntime==1.20.1
```

For GPU-accelerated (requires CUDA 12.x and cuDNN 9.x):
```
pip install onnxruntime-gpu==1.20.1
```

If you are using a virtual environment, make sure it's activated before installing onnxruntime.

## Usage
Train:
```
python main.py train --data-roots /path/datasetA /path/datasetB /path/datasetC --batch-size 16 --epochs 50 --img-size 256 --amp
```

Test:
```
python main.py test --weights checkpoints/best.pth --data /path/to/dataset --output results --img-size 256
```

Infer:
```
python main.py infer --weights checkpoints/best.pth --input /path/to/images --output test_results --img-size 256
```

Infer with overlay (generates both mask and overlay images):
```
python main.py infer --weights checkpoints/best.pth --input /path/to/images --output results --img-size 256 --overlay
```

Infer with ONNX model (CPU):
```
python main.py infer --weights checkpoints/best.onnx --input /path/to/images --output results --img-size 256 --device cpu
```

Infer with ONNX model (GPU/CUDA):
```
python main.py infer --weights checkpoints/best.onnx --input /path/to/images --output results --img-size 256 --device cuda
```

Notes:
- Each dataset root must contain `train/images` and `train/masks` (or `images` and `masks`).
- The scripts use WeightedRandomSampler to balance sampling among datasets.
- By default, inference only outputs mask files with the same names as input images.
- Use `--overlay` flag to generate both mask files (with _mask suffix) and overlay images (with _overlay suffix).
- After training, the best.pth model will be automatically converted to ONNX format (best.onnx).
- Both PyTorch (.pth) and ONNX (.onnx) models are supported for inference and testing.
