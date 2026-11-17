@echo off
echo [%TIME%] Setting up MSVC environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 ( echo VCVARSALL FAILED & exit /b 1 )

rem Add CUDA to PATH so cmake finds nvcc
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

echo [%TIME%] MSVC + CUDA ready:
where cl.exe
where nvcc.exe

echo [%TIME%] Running cmake configure (SM120, CUDA enabled, ccache OFF)...
cd /d "%~dp0build_sm120"

"%~dp0..\mingw64\bin\cmake.exe" "%~dp0." ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DGGML_CUDA_GRAPHS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=120 ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" ^
    -DGGML_NATIVE=ON ^
    -DGGML_AVX2=ON ^
    -DGGML_FMA=ON ^
    -DGGML_F16C=ON ^
    -DLLAMA_CURL=OFF ^
    -DGGML_CCACHE=OFF ^
    -DCMAKE_MAKE_PROGRAM="%~dp0..\ninja\ninja.exe"

if errorlevel 1 ( echo CMAKE_FAILED & exit /b 1 )
echo [%TIME%] CMAKE_OK

echo [%TIME%] Building with Ninja...
"%~dp0..\ninja\ninja.exe"
if errorlevel 1 ( echo NINJA_FAILED & exit /b 1 )

echo [%TIME%] BUILD_SUCCESS
