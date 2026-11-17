# REWRITE-SAFE: writes only two small vocab-only GGUF files under the scratch dir it is given; reads the Gemma 4 GGUF header only (no tensors).
"""Gemma 4 GGUFs that name their tokenizer 'llama' (row S69-OLLAMA-PICK-GEMMA4-TOKENIZER-LLAMA-TO-GEMMA4-2026-09-29-01).

Some published Gemma 4 GGUFs (Ollama registry pulls) write tokenizer.ggml.model = "llama" (SentencePiece) although the model needs the BPE "gemma4"
tokenizer; with the SPM tokenizer the control tokens <|turn> / <turn|> / <|channel> are split into subword pieces and come back as raw text.

Usage: pythonw check_gemma4_llama_tokenizer.py <gemma4.gguf> <llama-tokenize> <scratch-dir> [--expect-fixed|--expect-defect]
  builds two vocab-only GGUFs from the real file's metadata, one with tokenizer.ggml.model rewritten to "llama" and one as shipped, tokenizes
  '<|turn>user\\nHi<turn|>' with both and compares the ids.
  --expect-defect  (fail-before) exit 0 only if the two DIFFER, i.e. the defect is present
  --expect-fixed   (pass-after)  exit 0 only if the two are IDENTICAL and '<|turn>' is one token
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf  # noqa: E402

SRC, TOK, SCRATCH = sys.argv[1:4]
MODE = sys.argv[4] if len(sys.argv) > 4 else "--expect-fixed"
# spaces, doubled spaces, a tab, non-ASCII and a newline-indent: text where a SentencePiece-style tokenizer and the BPE one differ
TEXT = "<|turn>user\nHello world,  how are you?  你好\tx = [1, 2]\n  indented<turn|>\n<|turn>model\n"
os.makedirs(SCRATCH, exist_ok=True)

reader = gguf.GGUFReader(SRC, "r")
arch = reader.get_field("general.architecture").contents()
assert arch == "gemma4", "this check is for arch gemma4, got %r" % arch
tokens = reader.get_field("tokenizer.ggml.tokens").contents()
assert "<|turn>" in tokens, "the source vocabulary has no <|turn> token"
turn_id = tokens.index("<|turn>")
assert reader.get_field("tokenizer.ggml.merges") is not None, "the source file carries no merges: the promotion cannot apply"


def write_vocab_only(path, model_name):
    w = gguf.GGUFWriter(path, arch)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vtype = field.types[0]
        sub = field.types[1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if key == "tokenizer.ggml.model":
            val = model_name
        w.add_key_value(key, val, vtype, sub)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tokenize(model_path):
    p = subprocess.run([TOK, "-m", model_path, "-p", TEXT, "--ids", "--no-bos"], capture_output=True, text=True, timeout=300, creationflags=0x08000000)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.findall(r"\[([0-9, ]+)\]", p.stdout or "")
    ids = [int(x) for x in m[-1].replace(" ", "").split(",") if x] if m else None
    return p.returncode, ids, out


as_llama = os.path.join(SCRATCH, "g4_vocab_llama.gguf")
as_gemma4 = os.path.join(SCRATCH, "g4_vocab_gemma4.gguf")
write_vocab_only(as_llama, "llama")
write_vocab_only(as_gemma4, "gemma4")
rc_a, ids_a, out_a = tokenize(as_llama)
rc_b, ids_b, out_b = tokenize(as_gemma4)
print("shipped as gemma4 : rc=%s ids=%s" % (rc_b, ids_b))
print("rewritten 'llama' : rc=%s ids=%s" % (rc_a, ids_a))
print("<|turn> is token %d in the vocabulary" % turn_id)
ok_ctrl = rc_b == 0 and ids_b is not None and ids_b[0] == turn_id
print("CONTROL (gemma4 file starts with the <|turn> id): %s" % ("PASS" if ok_ctrl else "FAIL"))
if not ok_ctrl:
    sys.exit(3)
same = rc_a == 0 and ids_a == ids_b
if MODE == "--expect-defect":
    print("DEFECT PRESENT (ids differ): %s" % (not same))
    sys.exit(0 if not same else 1)
print("FIXED (ids identical): %s" % same)
sys.exit(0 if same else 1)
