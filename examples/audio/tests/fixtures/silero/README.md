Fixtures for tests/test-asr-silero.cpp (MIT, ggml-org/whisper.cpp @ 7e26d3b0d5bced6c8217ef09710ec2f969001241):
- silero-v6.2.0-for-tests.ggml : models/for-tests-silero-v6.2.0-ggml.bin (renamed: `*.bin` is gitignored here)
- jfk.wav                     : samples/jfk.wav (16 kHz mono, 11 s)
- jfk-whispercpp-probs.txt    : per-512-sample speech probabilities printed by whisper.cpp's own
                                whisper_vad_probs() on jfk.wav, plus its segments (SEG lines: t0 t1 as returned by whisper_vad_segments_get_segment_t0/t1, centiseconds)
