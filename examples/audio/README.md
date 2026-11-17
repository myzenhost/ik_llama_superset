# Unified audio for ik_llama.cpp

Listen, think and speak from one binary.

```
llama-audio --task asr   -m model.gguf --mmproj audio-mmproj.gguf --audio in.wav
llama-audio --task tts   -m outetts.gguf --vocoder-model wavtok.gguf -p "hello" -o out.wav
llama-audio --task chat  -m model.gguf -p "hello"
llama-audio --task s2s   -m model.gguf --mmproj audio-mmproj.gguf --audio in.wav \
                         --vocoder-model wavtok.gguf -o reply.wav
llama-tts   -m outetts.gguf --vocoder-model wavtok.gguf -p "hello" -o out.wav --metrics
```

## What this is, and what it deliberately is not

The design is adopted from [audio.cpp](https://github.com/0xShug0/audio.cpp)
(Apache-2.0): one native runtime owning every audio task behind a single
`--task` selector, over a shared audio-utility layer, instead of a separate
demo binary per model family.

It is **not** a vendored copy of audio.cpp. audio.cpp is a standalone engine of
~4800 files that carries **its own ggml** under `external/ggml`. ik_llama.cpp is
deliberately divergent from upstream ggml — its IQK / Trellis / MXFP4 quants and
its CPU prompt-processing work depend on that divergence — so importing another
project's ggml would destroy exactly the thing this fork exists for. audio.cpp's
own README makes the same point from the other side: *"GGUF is a container, not
a universal architecture adapter."*

So what was adopted is the architecture and the concrete pieces that fit:

| from | what was taken |
|---|---|
| audio.cpp | the unified `--task` CLI, the shared "Audio Utilities" layer (WAV I/O, resampling, VAD chunking, STFT/ISTFT), the `model_specs/*.json` catalogue format, `--metrics` / `--inspect` / `--list-*` |
| llama.cpp b4404 `examples/tts` (MIT) | the OuteTTS prompt format, the text front-end, the ISTFT + `fold` vocoder head |
| llama.cpp `src/models/wavtokenizer-dec.cpp` (MIT) | the WavTokenizer decoder graph, re-expressed as a standalone ggml module |
| this fork, already present | the mtmd audio projectors and the whisper-style log-mel front-end in `examples/mtmd/mtmd-audio.cpp` |

**No core file was touched.** `src/llama.cpp`, `src/llama-model.cpp`,
`ggml/include/ggml.h`, `ggml/src/ggml.c`, `examples/mtmd/mtmd.cpp` and
`examples/mtmd/clip.cpp` are all unmodified. Upstream carries the vocoder as
`LLM_ARCH_WAVTOKENIZER_DEC` **inside libllama**; here it is a self-contained
graph in `examples/tts/wavtokenizer.cpp` instead, which is both safer for a
divergent fork and closer to how audio.cpp organises a model family.

## Layout

```
examples/audio/
  audio-common.{h,cpp}   WAV I/O, mono/resample, VAD + chunking, hann/irfft/fold ISTFT
  audio.cpp              llama-audio: the unified asr/tts/chat/s2s CLI
  model_specs/*.json     declarative model catalogue (audio.cpp schema)
examples/tts/
  wavtokenizer.{h,cpp}   standalone ggml WavTokenizer decoder
  tts.cpp                llama-tts: OuteTTS -> codes -> vocoder -> WAV
```

`audio-common` and `wavtokenizer` are libraries, so a future `llama-server`
audio endpoint can link them without going through either CLI.

## Two ggml gaps, and how they were handled

The decoder graph needs two ops this fork's ggml does not export:

* `ggml_pad_reflect_1d`, `ggml_cos`, `ggml_sin` — **not actually required.** The
  WavTokenizer decoder graph does not use them; they were added upstream for
  other work. Nothing to do.
* `ggml_conv_1d_dw_ph` — genuinely missing. Rather than edit `ggml/src/ggml.c`
  (owned by another workstream, and a core file in a fork whose whole value is
  its divergence), the depthwise convolution is expressed in
  `wt_conv_1d_dw_ph()` with primitives that do exist: symmetric zero padding via
  `ggml_concat`, then a tap-by-tap accumulation
  `out[l][c] = sum_k w[k][c] * padded[l+k][c]`. That is the definition of a
  depthwise convolution, and it avoids depending on `ggml_im2col`'s depthwise
  layout conventions.

The decoder also materialises every weight as F32 on load. The model is tens of
millions of parameters, so the cost is small, and a single dtype removes an
entire class of "elementwise op got an F16 operand" failure. Quantised decoder
weights are refused with an explicit error rather than silently producing noise.

## Verified configuration

Use **OuteTTS 0.2** with `WavTokenizer-Large-75-F16`, not 0.3. Through identical
code, 0.3 produced -46 dBFS audio whose words did not match the request; 0.2
produces -16 dBFS and text that transcribes back verbatim.

Round trip, checked with this fork's own ASR judging its own TTS:

```sh
llama-tts -m OuteTTS-0.2-500M-Q8_0.gguf --vocoder-model WavTokenizer-Large-75-F16.gguf \
          -p "The fork can speak." -o out.wav --tts-normalize

llama-audio --task asr -m Voxtral-Mini-3B-2507-Q4_K_M.gguf \
          --mmproj mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf --audio out.wav --temp 0
  -> "The fork can speak."
```

ASR accuracy against LibriSpeech ground truth: **WER 2.0%** over 51 words
(0.0% on the 8-word clip, 2.3% on the 39-word clip).

## Speech to speech needs two backbones, in one process

`-m` is the model that LISTENS -- it carries the audio projector. An
audio-understanding LLM has no acoustic-code tokens to speak with, so `--task s2s`
also takes `--tts-model`. Both load into the SAME process:

```sh
llama-audio --task s2s -m Voxtral-Mini-3B-2507-Q4_K_M.gguf \
    --mmproj mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf --audio in.wav \
    --tts-model OuteTTS-0.2-500M-Q8_0.gguf \
    --vocoder-model WavTokenizer-Large-75-F16.gguf -o reply.wav --tts-normalize
```

Measured: speech in -> "Concord returned to its place amidst the tents." ->
resynthesised -> transcribes back identically. Round trip lossless.

## Models

See `model_specs/`. Every repo and filename there was read from the Hugging Face
API, not from a search result. Three families run today (`outetts` +
`wavtokenizer` for speech synthesis, `voxtral` for transcription once its mmproj
is fetched, and `parakeet_tdt`); `qwen3_asr` and `muse` are recorded as **wanted**,
each with the exact file to fetch and a `blocked_by` block saying precisely what
is missing.

### Parakeet-TDT (transducer ASR, no text model)

`convert-parakeet-tdt-to-gguf.py` turns NVIDIA's `.nemo` checkpoint into one GGUF:
the FastConformer encoder under the mtmd `parakeet` tensor names, and the
Token-and-Duration Transducer decoder under `tdt.*`. `--task asr --mmproj` that
file (no `-m`) runs the encoder in mtmd and decodes greedily on the CPU
(`asr-tdt.{h,cpp}`); `--words-out` writes word times taken from the duration head.
Measured (`tests/test-asr-tdt-cli.py`): jfk.wav exact, 0/51 word errors on the two
LibriSpeech fixtures, and an empty transcript on 10 s of silence and 20 s of
-50 dBFS noise.

### Live transcription (`--task stream`)

```
llama-audio --task stream --mmproj parakeet-tdt.gguf                 # default microphone
llama-audio --task stream --mmproj parakeet-tdt.gguf --audio in.wav  # a file, one step at a time
llama-audio --list-capture-devices
```

This is whisper.cpp's `examples/stream` sliding window, run over the transducer above.

- **Windowing.** Every `--stream-step` ms (default 3000) the newest audio is
  transcribed together with the tail of the previous window, up to
  `--stream-length` ms (default 10000) in total. Every `length/step - 1` steps the
  line is committed and printed as `[t0 --> t1] text`. Only `--stream-keep` ms
  (default 200) carry into the next line.
- **Capture.** Audio is captured through the miniaudio already vendored in this
  tree, so no SDL2 dependency is added. `--stream-capture null` opens miniaudio's
  null device, which is silent and needs no hardware; the tests use it.
- **File input.** `--audio` with `--stream-speed X` plays a file into the same
  ring at X times real time. When the decoder falls behind, the newest step is
  kept and the skipped audio is reported, as whisper.cpp warns and drops.
- **Model support.** Streaming needs a transducer GGUF. The audio-LLM route
  would re-prefill the whole window at every step and is refused
  (row S69-WHISPERCPP-STREAM-LLM-ROUTE-2026-10-20-01).
