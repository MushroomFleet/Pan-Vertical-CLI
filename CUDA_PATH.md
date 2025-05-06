# Setting CUDA_PATH in a Virtual Environment

Here's a step-by-step guide to set up the CUDA_PATH environment variable while using a Python virtual environment:

## For Windows:

1. **Install CUDA Toolkit** first if you haven't already. Download from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

2. **Find your CUDA installation path**. It's typically something like:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
   ```
   (where 12.x is your CUDA version)

3. **Create environment variable files** in your virtual environment:
   - Navigate to your virtual environment's directory
   - For activation, create/edit `venv\Scripts\activate.bat` to include:
   ```bat
   @echo off
   set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
   set "PATH=%CUDA_PATH%\bin;%PATH%"
   ```

4. **For deactivation** (optional), edit `venv\Scripts\deactivate.bat`:
   ```bat
   @echo off
   set CUDA_PATH=
   ```

5. **Reactivate your virtual environment**:
   ```
   deactivate
   venv\Scripts\activate
   ```

6. **Verify the setup**:
   ```
   echo %CUDA_PATH%
   ```

## For Linux/macOS:

1. **Install CUDA Toolkit** if needed.

2. **Create/edit activation script** in your venv:
   - Find `bin/activate` in your venv folder
   - Add these lines at the end:
   ```bash
   # Store old path to restore later
   export _OLD_CUDA_PATH="$CUDA_PATH"
   
   # Set CUDA path (adjust version as needed)
   export CUDA_PATH="/usr/local/cuda-12.x"
   export PATH="$CUDA_PATH/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
   ```

3. **Update deactivation** in the same file:
   ```bash
   # Add this to the deactivate() function
   if [ -n "${_OLD_CUDA_PATH+_}" ] ; then
       CUDA_PATH="$_OLD_CUDA_PATH"
       export CUDA_PATH
       unset _OLD_CUDA_PATH
   else
       unset CUDA_PATH
   fi
   ```

4. **Reactivate your environment**:
   ```bash
   deactivate
   source venv/bin/activate
   ```

5. **Verify with**:
   ```bash
   echo $CUDA_PATH
   ```

Would you like me to help troubleshoot any specific issues with this setup?