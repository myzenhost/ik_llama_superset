@echo off
echo === ik_llama Ninja Build (MAX PARALLEL -j8) ===
echo === Started: %date% %time% ===

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set PATH=%~dp0..\ninja;%PATH%

cd /d %~dp0
rd /s /q build_cuda 2>nul
mkdir build_cuda
cd build_cuda

echo === CMake Configure ===
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_MAKE_PROGRAM=%~dp0../ninja/ninja.exe

if errorlevel 1 (
    echo CMAKE FAILED
    pause
    exit /b 1
)

echo === Ninja Build -j8 (MAX PARALLEL) ===
%~dp0..\ninja\ninja.exe -j8

echo === DONE: %date% %time% ===
dir bin\*.exe 2>nul
dir bin\*.dll 2>nul
pause
