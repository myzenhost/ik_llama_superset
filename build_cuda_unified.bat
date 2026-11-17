@echo off
echo === Building ik_llama.cpp Unified CUDA Build ===
echo === RTX 2080 Ti (Turing, sm_75) + MSVC + CUDA 13.2 ===
echo.

REM Setup MSVC environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

echo === MSVC ready ===
where cl.exe
where nvcc.exe

set SRC_DIR=%~dp0.
set BUILD_DIR=%SRC_DIR%\build_cuda

REM Clean previous build
echo === Cleaning previous build ===
if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

echo === Configuring CMake ===
cmake "%SRC_DIR%" ^
    -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER="cl.exe" ^
    -DCMAKE_CXX_COMPILER="cl.exe" ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_GRAPHS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="75" ^
    -DGGML_NATIVE=ON ^
    -DGGML_AVX2=ON ^
    -DGGML_FMA=ON ^
    -DGGML_F16C=ON ^
    -DLLAMA_CURL=OFF ^
    -DCMAKE_CXX_FLAGS="/O2 /Ob2 /DNDEBUG" ^
    -DCMAKE_C_FLAGS="/O2 /Ob2 /DNDEBUG"

if errorlevel 1 (
    echo === CMake configuration FAILED ===
    pause
    exit /b 1
)

echo === Building (this will take 20-40 minutes for 378 CUDA kernels) ===
nmake

if errorlevel 1 (
    echo === Build FAILED ===
    pause
    exit /b 1
)

echo === Build complete ===
echo.
echo === Executables ===
dir /b "%BUILD_DIR%\bin\*.exe" "%BUILD_DIR%\bin\*.dll" 2>nul
echo.
echo === File sizes ===
dir "%BUILD_DIR%\bin\*.exe"
pause
