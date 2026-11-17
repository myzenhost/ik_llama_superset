# Position-gated LoRA adapters

Row `S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01`. A gated adapter applies its low-rank delta only on some rows of a
batch; the base weights run unchanged on every other row. The mask is a per-token F32 graph input multiplied into the delta
(`scale * mask * B(Ax)`), so gated and ungated adapters share one graph and one decode batch.

## GGUF keys

| key | type | meaning |
|---|---|---|
| `adapter.lora.gate` | string | `none` (default), `draft_rows` or `mask_token`. Any other value fails the load. `invocation` is not a gate here: aLoRA uses `adapter.alora.invocation_tokens`. |
| `adapter.lora.gate_token` | u32 | required for `mask_token`: rows whose input token equals it get the adapter. |

`convert_lora_to_gguf.py --gate {none,draft_rows,mask_token} [--gate-token ID]` writes them (a `gate` / `gate_token` field in
the PEFT adapter config is the fallback).

## API

`llama_set_draft_start(ctx, seq_id, pos)`: for `draft_rows` adapters, positions `>= pos` of that sequence are draft rows (the
adapter is on there). `pos < 0` clears. Sticky until changed; position based, so a rewind needs no extra call.

## Tests

`tests/test-adapter-peft-parity.py` cases `gated_draft_rows`, `gated_mask_token` compare against a PEFT forward that applies the
adapter on exactly the gated rows (max-abs < 1e-4 F32, adapter effect > 1e-1); the negative controls are the adapter removed and
the same adapter converted without the gate.

## Not done

The server speculative path does not mark draft rows yet, so the `--lora-hetero-batch` + speculative refusal in
`tools/server/server-context.cpp` (`init_lora_pool`) stays. Follow-up row filed at close.
