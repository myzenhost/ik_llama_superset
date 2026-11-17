"""Shared ASR word-error-rate helpers with a Whisper-style English text normaliser.

Row S69-ENGREV-WHISPERCPP-WERNORM-2026-10-27-01.  The audio tests used hand-rolled copies that
only lower-cased and stripped punctuation, so "twenty million dollars" and "$20 million" counted
as errors and numbers/titles/spelling moved WER by points.  This module is the one place they
now share.

SOURCE-READ: the behaviour of the OpenAI Whisper English normaliser (MIT), measured in
temp/arex1014/engrev_1027/ (_wernorm_unit_1028.py).  This is an independent, dependency-free
rewrite of the part number-heavy legal speech needs; no code copied.
  lower-case, drop [bracketed]/(parenthesised) asides, strip thousands commas, "$N" -> "N dollars",
  "%" -> "percent", titles (mr -> mister), a few British spellings, contractions, and number words
  -> digits including scales ("twenty million" -> 20000000) and ordinals ("third" -> 3rd).
Both sides of a comparison go through the same function, so oddities cancel.

wer_counts() returns (errors, reference_words); wer() the rate.
"""
import re

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALE = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
        "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
        "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
        "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
        "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90, "hundredth": 100}
_TITLES = {"mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor", "jr": "junior",
           "sr": "senior"}
_SPELL = {"colour": "color", "colours": "colors", "favour": "favor", "honour": "honor",
          "labour": "labor", "neighbour": "neighbor", "behaviour": "behavior", "centre": "center",
          "defence": "defense", "offence": "offense", "licence": "license", "judgement": "judgment",
          "organise": "organize", "recognise": "recognize", "analyse": "analyze",
          "programme": "program", "grey": "gray", "theatre": "theater"}
_CONTR = [(r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bshan't\b", "shall not"),
          (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"),
          (r"\bi'm\b", "i am")]


def _suffix(n):
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _is_num_start(tokens, i):
    t = tokens[i]
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    return (t in _ONES or t in _TENS or t in _SCALE or t in _ORD
            or (t.isdigit() and nxt in _SCALE))


def _numbers(tokens):
    """Collapse runs of number words (and a digit token followed by a scale word) into digits."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        if not _is_num_start(tokens, i):
            out.append(tokens[i])
            i += 1
            continue
        total = cur = 0
        ordinal = False
        while i < n:
            t = tokens[i]
            if t in _ONES:
                cur += _ONES[t]
            elif t in _TENS:
                cur += _TENS[t]
            elif t in _ORD:
                cur += _ORD[t]
                ordinal = True
                i += 1
                break
            elif (t == "and" and (total or cur >= 100) and i + 1 < n
                  and (tokens[i + 1] in _ONES or tokens[i + 1] in _TENS)):
                pass    # "one hundred and fifty": 'and' joins the run
            elif t.isdigit() and i + 1 < n and tokens[i + 1] in _SCALE:
                cur += int(t)
            elif t in _SCALE:
                if t == "hundred":
                    cur = max(cur, 1) * 100
                else:
                    total += max(cur, 1) * _SCALE[t]
                    cur = 0
            else:
                break
            i += 1
        value = total + cur
        out.append(f"{value}{_suffix(value)}" if ordinal else str(value))
    return out


def normalise(s, english=True):
    """-> list of words. english=False is the old lower-case + strip-punctuation behaviour."""
    s = s.lower()
    if not english:
        return [w for w in (re.sub(r"[^a-z0-9']", "", t) for t in s.split()) if w]
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = re.sub(r"\$\s?(\d+(?:\.\d+)?)(\s*(?:hundred|thousand|million|billion)\b)?",
               lambda m: m.group(1) + (m.group(2) or "") + " dollars", s)
    s = s.replace("%", " percent ").replace("&", " and ").replace("’", "'")
    for pat, rep in _CONTR:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9' .]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    toks = []
    for t in s.split():
        t = t.strip("'")
        if t:
            toks.append(_SPELL.get(_TITLES.get(t, t), _TITLES.get(t, t)))
    return [t for t in _numbers(toks) if t]


def wer_counts(ref, hyp, english=True):
    """-> (word errors, reference word count): Levenshtein over normalised words."""
    r, h = normalise(ref, english), normalise(hyp, english)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def wer(ref, hyp, english=True):
    errors, n = wer_counts(ref, hyp, english)
    if n == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / float(n)
