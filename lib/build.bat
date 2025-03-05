@echo off
echo Checking Python environment...
python --version || exit /b 1

echo Checking CUDA installation...
where nvcc || (
    echo CUDA not found in PATH
    echo Please install CUDA and add it to PATH
    exit /b 1
)

echo Building NMS modules...
cd nms

echo Cleaning previous builds...
del /F /Q *.pyd 2>nul
del /F /Q *.so 2>nul
rmdir /S /Q build 2>nul

echo Running setup...
python setup_windows.py build_ext --inplace

if %ERRORLEVEL% NEQ 0 (
    echo Build failed
    exit /b 1
)

echo Cleaning build directory...
rmdir /S /Q build 2>nul

echo Build completed successfully
cd ..