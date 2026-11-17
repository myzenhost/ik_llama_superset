# In-Place TTT oracle (vendored, unmodified)

Third-party code, Apache-2.0 (see `LICENSE` in this directory). Copied byte-for-byte from
https://github.com/ByteDance-Seed/In-Place-TTT at commit `be2324829b0e91c8fd10a74d4b43714fde6676e1`
(2026-04-21, "update README"), path `inference_model/hf_qwen3/`. Git blob shas were checked against the
commit's tree when fetched. NOT modified; `tests/test-ttt-inplace.py` refuses to run if a sha256 differs:

| file | sha256 |
|---|---|
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `hf_qwen3/__init__.py` | `a38d76873e560e9c034258408e1c67fdfc08f2be846da1937cea1434f6e10e80` |
| `hf_qwen3/configuration_qwen3.py` | `39ef69d7c79114ae9bd2710320c4260796fc5dbed9f725e18b5a0f5771114ec2` |
| `hf_qwen3/modeling_qwen3.py` | `98ec94bcf1780635d7817b993a0bf4fa1f1a7e3a35415dce8a46c345a52a2858` |

Used only as the reference implementation that `tests/test-ttt-inplace.py` runs (fp32, CPU, batch 1) to
produce the logits the fork must match. Requires torch, transformers (4.57.x; written against 4.57.3), einops, opt_einsum.
