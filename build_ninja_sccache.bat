@echo off
echo === ik_llama Ninja + sccache Build (FASTEST) ===
echo === RTX 2080 Ti (sm_75) + MSVC + CUDA 13.2 + sccache ===
echo === Started: %date% %time% ===

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set PATH=%~dp0..\ninja;%~dp0..\sccache;%PATH%
echo Ninja: & ninja --version
echo sccache: & sccache --version

cd /d %~dp0

:: Only clean if --clean flag passed, otherwise incremental
if "%1"=="--clean" (
    echo === Cleaning build ===
    rd /s /q build_cuda 2>nul
    mkdir build_cuda
) else (
    if not exist build_cuda mkdir build_cuda
)

cd build_cuda

:: Check if already configured (skip cmake if build.ninja exists)
if exist build.ninja (
    echo === Skipping CMake (already configured) ===
    echo === To reconfigure, run with --clean flag ===
) else (
    echo === CMake Configure with sccache ===
    cmake .. -G "Ninja" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DGGML_CUDA=ON ^
        -DGGML_CUDA_FA_ALL_QUANTS=ON ^
        -DCMAKE_CUDA_ARCHITECTURES=75 ^
        -DCMAKE_C_COMPILER=cl.exe ^
        -DCMAKE_CXX_COMPILER=cl.exe ^
        -DCMAKE_C_COMPILER_LAUNCHER=%~dp0../sccache/sccache.exe ^
        -DCMAKE_CXX_COMPILER_LAUNCHER=%~dp0../sccache/sccache.exe ^
        -DCMAKE_CUDA_COMPILER_LAUNCHER=%~dp0../sccache/sccache.exe ^
        -DCMAKE_MAKE_PROGRAM=%~dp0../ninja/ninja.exe ^
        -DGGML_CCACHE=OFF

    if errorlevel 1 (
        echo CMAKE FAILED
        pause
        exit /b 1
    )
)

echo === Ninja Build -j8 with sccache ===
%~dp0..\ninja\ninja.exe -j8

if errorlevel 1 (
    echo BUILD FAILED
    pause
    exit /b 1
)

echo === sccache stats ===
%~dp0..\sccache\sccache.exe --show-stats

echo === DONE: %date% %time% ===
dir bin\*.exe 2>nul
dir bin\*.dll 2>nul
pause
