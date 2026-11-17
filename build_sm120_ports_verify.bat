@echo off
REM SM120 build to verify the five #247 MTP/KV ports actually compile and link.
REM
REM -j 32 of 128 cores ON PURPOSE. The host hard-power-lost twice on 2026-07-26 (Event 41), and the
REM sentinel's final samples showed the GPU calm (393 W of a 480 W cap, no throttle bits) while
REM cpu_max_core repeatedly hit 100%. A 128-way nvcc fan-out is the largest CPU transient this machine
REM ever produces. Capping it trades wall-clock for not re-triggering the thing under investigation.
REM Raise only after the Event-41 cause is actually identified.

REM All paths script-relative. %~dp0 is this script's own directory (…\portable\ik_llama_fork\,
REM with a trailing backslash), so %~dp0.. is the portable root. These were absolute D: paths,
REM which would break the moment the tree moved -- and the tree is moving.
REM 2026-08-30: the example path above said ik_llama_src, left over from when this script lived in
REM that tree. Harmless here because this script already does cd /d "%~dp0" (its own directory),
REM but its four siblings in this tree did NOT -- they carried `cd /d %~dp0..\ik_llama_src`, which
REM is a self-cd in ik_llama_src and a CROSS-TREE jump here, so canonical's own build scripts were
REM building the inert fork. Fixed in the same commit as this comment.
call "%~dp0..\msvc\activate_x64.bat" >nul 2>&1
cd /d "%~dp0"

set BD=%~dp0build_sm120_verify
set CMAKE=%~dp0..\mingw64\bin\cmake.exe
set NINJA=%~dp0../ninja/ninja.exe

echo === CONFIGURE (%DATE% %TIME%) ===
"%CMAKE%" -S . -B "%BD%" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=ON
REM Capture before the block. The branch below is correct (`if errorlevel 1` is a runtime test),
REM but %ERRORLEVEL% read INSIDE the block expands at parse time and printed a stale value -- so a
REM genuine configure failure reported "rc=0", which reads like a success next to the word FAILED.
REM Same capture-first pattern the BUILD step below already uses.
set CFG_RC=%ERRORLEVEL%
if errorlevel 1 (
  echo CONFIGURE_FAILED rc=%CFG_RC%
  exit /b 1
)
echo CONFIGURE_OK

echo === BUILD -j 32 (%DATE% %TIME%) ===
"%CMAKE%" --build "%BD%" --parallel 32
set RC=%ERRORLEVEL%
echo BUILD_EXIT rc=%RC% (%DATE% %TIME%)
exit /b %RC%
