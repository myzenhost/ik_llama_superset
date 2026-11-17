@echo off
rem Try loading and decoding the generated random-weight deepseek4 GGUF: ours CPU-only saves base logits, ours GPU compares (differential, no stock build involved).
rem GATE-CLAIM-OK: test run, wrapped in host_slot by the caller. SETUP-VERIFIED: flags as in _tca_kld2_1004.bat (-dev none --no-op-offload for CPU-only); GGML_CUDA_NO_TF32 as the arch matrix harness uses it.
set B=build\bin
set MDL=%1
set T=tmp/_tca_ppl_text_1004.txt
set L=tmp/_tca_dsv4rw_%2_base_1004.bin
%B%\llama-perplexity.exe -m %MDL% -f %T% -c 256 -b 256 -ub 256 --chunks 2 -dev none --no-op-offload --kl-divergence-base %L% > tmp/_tca_dsv4rw_%2_cpu_1004.txt 2>&1
%B%\llama-perplexity.exe -m %MDL% -f %T% -c 256 -b 256 -ub 256 --chunks 2 -ngl 99 --kl-divergence-base %L% --kl-divergence > tmp/_tca_dsv4rw_%2_gpu_1004.txt 2>&1
set GGML_CUDA_NO_TF32=1
%B%\llama-perplexity.exe -m %MDL% -f %T% -c 256 -b 256 -ub 256 --chunks 2 -ngl 99 --kl-divergence-base %L% --kl-divergence > tmp/_tca_dsv4rw_%2_gpu_notf32_1004.txt 2>&1
set GGML_CUDA_NO_TF32=
exit /b 0
