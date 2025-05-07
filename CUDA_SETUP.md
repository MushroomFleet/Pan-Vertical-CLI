# CUDA Acceleration Setup Guide for Video Panner

This guide provides instructions for setting up NVIDIA CUDA for GPU acceleration with Video Panner. To use GPU acceleration, you'll need:

1. An NVIDIA GPU with CUDA support
2. NVIDIA CUDA Toolkit (tested with v11.8)
3. NVIDIA cuDNN library

## Already Installed

The installation scripts (`install-uv.bat` and `install-uv.ps1`) handle:
- Creating a Python virtual environment
- Installing required Python packages
- Setting up CUDA environment variables if CUDA is found in the standard location

## Missing Dependencies

If you encounter errors related to CUDA when running the video panner, particularly:
- `ModuleNotFoundError: No module named 'fastrlock'` 
- `cudnn64_8.dll -> not found`
- `cuTENSOR.dll -> not found`

Follow these steps:

## Step 1: Reinstall Virtual Environment

Run the installation script again and select "y" when asked if you want to recreate the virtual environment. This will ensure all dependencies including `fastrlock` are installed:

```
.\install-uv.bat
```

or 

```
.\install-uv.ps1
```

## Step 2: Install cuDNN (Required for full GPU acceleration)

1. **Register with NVIDIA Developer Program**: Visit [NVIDIA Developer Portal](https://developer.nvidia.com/cudnn) and register/login
2. **Download cuDNN**: Download the cuDNN package that matches your CUDA version (e.g., cuDNN v8.x.x for CUDA 11.8)
3. **Extract and Install**:
   - Extract the downloaded archive
   - Copy the following files to your CUDA installation directories:
     - Copy `cuda\bin\cudnn*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
     - Copy `cuda\include\cudnn*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
     - Copy `cuda\lib\x64\cudnn*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64`

## Step 3: Install TensorRT (Optional)

If you require `cuTENSOR.dll` for specific operations:

1. **Download TensorRT**: Visit [NVIDIA TensorRT page](https://developer.nvidia.com/tensorrt) 
2. **Download and install** the appropriate version for your CUDA installation
3. **Add to PATH**: Ensure the TensorRT bin directory is in your system PATH

## Step 4: Verify Installation

After installing the missing components:

1. Run the test script to verify CUDA acceleration is working:
   ```
   .\start-uv.bat
   ```
   or
   ```
   .\start-uv.ps1
   ```

2. Check the console output for confirmation of CUDA acceleration:
   ```
   CUDA acceleration enabled via CuPy!
   ```

## Troubleshooting

If you still encounter issues:

- **GPU Driver**: Ensure your NVIDIA GPU drivers are up to date
- **Path Variables**: Check that CUDA and cuDNN paths are correctly set
- **System Compatibility**: Verify your GPU supports the CUDA version you've installed
- **CUDA Version**: Make sure the cuDNN and TensorRT versions match your CUDA version

For more detailed information, refer to the official documentation:
- [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
