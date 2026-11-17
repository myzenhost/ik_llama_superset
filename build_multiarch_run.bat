@echo off
echo [%TIME%] Setting up MSVC environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 ( echo VCVARSALL FAILED & exit /b 1 )

rem Add CUDA to PATH so cmake finds nvcc
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

echo [%TIME%] MSVC + CUDA ready
where cl.exe
where nvcc.exe

rem Wipe and recreate build dir (clean state)
if exist "%~dp0build_multiarch" (
    echo [%TIME%] Removing old build_multiarch...
    rmdir /s /q "%~dp0build_multiarch"
)
mkdir "%~dp0build_multiarch"

echo [%TIME%] Running cmake configure (SM75+SM86+SM89+SM90+SM120, CUDA enabled)...
cd /d "%~dp0build_multiarch"

"%~dp0..\mingw64\bin\cmake.exe" "%~dp0." ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DGGML_CUDA_GRAPHS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89;90;120" ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" ^
    -DGGML_NATIVE=OFF ^
    -DGGML_AVX_VNNI=OFF ^
    -DGGML_AVX2=ON ^
    -DGGML_FMA=ON ^
    -DGGML_F16C=ON ^
    -DLLAMA_CURL=OFF ^
    -DGGML_CCACHE=OFF ^
    -DCMAKE_MAKE_PROGRAM="%~dp0..\ninja\ninja.exe"

if errorlevel 1 ( echo CMAKE_FAILED & exit /b 1 )
echo [%TIME%] CMAKE_OK

echo [%TIME%] Building multi-arch with Ninja (SM75+SM86+SM89+SM90+SM120)...
echo [%TIME%] Note: multi-arch build takes 2-4x longer than single-arch
"%~dp0..\ninja\ninja.exe"
if errorlevel 1 ( echo NINJA_FAILED & exit /b 1 )

echo [%TIME%] MULTIARCH_BUILD_SUCCESS
dir /b bin\*.exe 2>nul | find /c ".exe"
