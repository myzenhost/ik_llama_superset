@echo off
echo === Building ik_llama.cpp with Ninja (PARALLEL CUDA) ===
echo === RTX 2080 Ti (Turing, sm_75) + MSVC + CUDA 13.2 ===

:: Setup MSVC
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

echo === MSVC ready ===
where cl.exe
where nvcc.exe

:: Add Ninja to PATH
set PATH=%~dp0..\ninja;%PATH%
ninja --version

:: Clean and create build directory
echo === Cleaning previous build ===
cd /d %~dp0
rd /s /q build_cuda 2>nul
mkdir build_cuda
cd build_cuda

:: Configure with Ninja generator (KEY DIFFERENCE from nmake)
echo === Configuring CMake with Ninja generator ===
cmake .. ^
    -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=75 ^
    -DCMAKE_C_COMPILER="cl.exe" ^
    -DCMAKE_CXX_COMPILER="cl.exe" ^
    -DCMAKE_MAKE_PROGRAM="%~dp0../ninja/ninja.exe"

if errorlevel 1 (
    echo CMAKE FAILED
    pause
    exit /b 1
)

:: Build with Ninja (PARALLEL - uses all CPU cores by default)
echo === Building with Ninja (parallel CUDA compilation) ===
echo === This should be 4-8x faster than nmake ===
ninja -j4

if errorlevel 1 (
    echo BUILD FAILED
    pause
    exit /b 1
)

echo === Build complete ===
dir /s bin\*.exe 2>nul
pause
