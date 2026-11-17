# audio model specs

Declarative catalogue of the audio models `llama-audio` / `llama-tts` can run,
in the JSON shape audio.cpp uses for `model_specs/*.json` (Apache-2.0):
`schema_version`, `family`, `category`, `tasks`, `languages`, `runtime`,
`packages[]` with a `download` block, and `ui.recommended_package`.

The point of the format is that a model is described in DATA rather than in a
per-model binary: the family says which loader handles it, the package says
exactly which file to fetch and from where. Every `repo` + `files` pair below
was read from the Hugging Face API on 2026-08-31, not from a search snippet.

Two subsets are tracked:

* `status: "supported"` -- runs today with the code in this tree.
* `status: "wanted"` -- the model this fork should gain next; the spec records
  the exact file to fetch and what is still missing, so nothing has to be
  re-researched.

Nothing here downloads automatically. Fetch a package by hand, e.g.

```sh
curl -L -o WavTokenizer-Large-75-F16.gguf \
  https://huggingface.co/ggml-org/WavTokenizer/resolve/main/WavTokenizer-Large-75-F16.gguf
```
