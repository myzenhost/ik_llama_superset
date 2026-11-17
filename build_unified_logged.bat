@echo off
echo === ik_llama Unified Build (TurboQuant + TriAttention) ===
echo === Started: %date% %time% ===
echo === Log: %~dp0..\..\temp\_scratch\build_unified_2026-04-13.log ===

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set PATH=%~dp0..\ninja;%PATH%

cd /d %~dp0
rd /s /q build_cuda 2>nul
mkdir build_cuda
cd build_cuda

echo === CMake Configure ===
cmake .. -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=75 ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCMAKE_MAKE_PROGRAM=%~dp0../ninja/ninja.exe

if errorlevel 1 (
    echo CMake FAILED
    exit /b 1
)

echo === Ninja Build -j8 ===
%~dp0..\ninja\ninja.exe -j8

if errorlevel 1 (
    echo Build FAILED
    exit /b 1
)

echo === BUILD SUCCESS: %date% %time% ===
dir bin\*.exe 2>nul
dir bin\*.dll 2>nul
