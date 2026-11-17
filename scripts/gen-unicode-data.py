from __future__ import annotations

import argparse
import array
import sys
import unicodedata

import requests


def out(line=""):
    print(line, end='\n')  # noqa


def normalization_tables():
    out(f"// Canonical normalization tables: Unicode {unicodedata.unidata_version}.")
    out("const std::unordered_map<uint32_t, uint8_t> unicode_combining_class = {")
    for cpt in range(0x110000):
        if value := unicodedata.combining(chr(cpt)):
            out(f"{{0x{cpt:06X}, {value}}},")
    out("};\n")
    out("const std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> unicode_decomposition = {")
    compositions = []
    for cpt in range(0x110000):
        if 0xAC00 <= cpt < 0xAC00 + 11172:
            continue
        raw = unicodedata.decomposition(chr(cpt))
        if not raw or raw.startswith("<"):
            continue
        parts = [int(part, 16) for part in raw.split()]
        assert 1 <= len(parts) <= 2
        first, second = parts[0], parts[1] if len(parts) == 2 else 0
        out(f"{{0x{cpt:06X}, {{0x{first:06X}, 0x{second:06X}}}}},")
        if second and unicodedata.normalize("NFC", chr(first) + chr(second)) == chr(cpt):
            compositions.append((first, second, cpt))
    out("};\n")
    out("const std::unordered_map<uint64_t, uint32_t> unicode_composition = {")
    for first, second, cpt in compositions:
        out(f"{{0x{((first << 32) | second):016X}ULL, 0x{cpt:06X}}},")
    out("};")


# Generate only canonical tables, without downloading the UCD data.
if sys.argv[1:] == ["--normalization-only"]:
    normalization_tables()
    sys.exit(0)


MAX_CODEPOINTS = 0x110000

# Pinned, not "latest": the pre-tokenizer regexes must classify code points the way HF tokenizers (Oniguruma / the
# Rust regex crate) do, and those are on Unicode 16.0 as of 2026-10. "latest" made the output depend on the day the
# script ran (the previous table was 15.1.0) and would now pull 17.0.0, whose new blocks HF does not know.
DEFAULT_UCD_VERSION = "16.0.0"

parser = argparse.ArgumentParser(description="Generate src/unicode-data.cpp from the Unicode Character Database")
parser.add_argument("--ucd-version", default=DEFAULT_UCD_VERSION, help="UCD version to download (default %(default)s)")
parser.add_argument("--ucd-file", default=None, help="local UnicodeData.txt to use instead of downloading; pass the matching --ucd-version, it is written into the header")
args = parser.parse_args()

UNICODE_DATA_URL = f"https://www.unicode.org/Public/{args.ucd_version}/ucd/UnicodeData.txt"


def unicode_data_text() -> str:
    if args.ucd_file:
        with open(args.ucd_file, encoding="utf-8") as f:
            return f.read()
    res = requests.get(UNICODE_DATA_URL)
    res.raise_for_status()
    return res.content.decode()


UNICODE_DATA = unicode_data_text()

# Canonical decompositions (field 5 without a <tag>), used for NFD below. Taken from the same UnicodeData.txt rather
# than Python's unicodedata module, whose UCD version is that of the interpreter (14.0.0 for Python 3.11), so the NFD
# table always matches the version of the flags table.
canonical_decomposition: dict[int, list[int]] = {}
for _line in UNICODE_DATA.splitlines():
    _f = _line.split(";")
    if _f[5] and not _f[5].startswith("<"):
        canonical_decomposition[int(_f[0], 16)] = [int(x, 16) for x in _f[5].split()]


def nfd_first(cpt: int) -> int:
    # Hangul syllables decompose algorithmically (Unicode 3.12); the first jamo is the leading consonant
    if 0xAC00 <= cpt <= 0xD7A3:
        return 0x1100 + (cpt - 0xAC00) // (21 * 28)
    while cpt in canonical_decomposition:
        cpt = canonical_decomposition[cpt][0]
    return cpt


# see https://www.unicode.org/L2/L1999/UnicodeData.html
def unicode_data_iter():
    data = UNICODE_DATA

    prev = []

    for line in data.splitlines():
        # ej: 0000;<control>;Cc;0;BN;;;;;N;NULL;;;;
        line = line.split(";")

        cpt = int(line[0], base=16)
        assert cpt < MAX_CODEPOINTS

        cpt_lower = int(line[-2] or "0", base=16)
        assert cpt_lower < MAX_CODEPOINTS

        cpt_upper = int(line[-3] or "0", base=16)
        assert cpt_upper < MAX_CODEPOINTS

        categ = line[2].strip()
        assert len(categ) == 2

        bidir = line[4].strip()
        assert len(categ) == 2

        name = line[1]
        if name.endswith(", First>"):
            prev = (cpt, cpt_lower, cpt_upper, categ, bidir)
            continue
        if name.endswith(", Last>"):
            assert prev[1:] == (0, 0, categ, bidir)
            for c in range(prev[0], cpt):
                yield (c, cpt_lower, cpt_upper, categ, bidir)

        yield (cpt, cpt_lower, cpt_upper, categ, bidir)


# see definition in unicode.h
CODEPOINT_FLAG_UNDEFINED   = 0x0001  #
CODEPOINT_FLAG_NUMBER      = 0x0002  # \p{N}
CODEPOINT_FLAG_LETTER      = 0x0004  # \p{L}
CODEPOINT_FLAG_SEPARATOR   = 0x0008  # \p{Z}
CODEPOINT_FLAG_MARK        = 0x0010  # \p{M}
CODEPOINT_FLAG_PUNCTUATION = 0x0020  # \p{P}
CODEPOINT_FLAG_SYMBOL      = 0x0040  # \p{S}
CODEPOINT_FLAG_CONTROL     = 0x0080  # \p{C}

UNICODE_CATEGORY_TO_FLAG = {
    "Cn": CODEPOINT_FLAG_UNDEFINED,    # Undefined
    "Cc": CODEPOINT_FLAG_CONTROL,      # Control
    "Cf": CODEPOINT_FLAG_CONTROL,      # Format
    "Co": CODEPOINT_FLAG_CONTROL,      # Private Use
    "Cs": CODEPOINT_FLAG_CONTROL,      # Surrrogate
    "Ll": CODEPOINT_FLAG_LETTER,       # Lowercase Letter
    "Lm": CODEPOINT_FLAG_LETTER,       # Modifier Letter
    "Lo": CODEPOINT_FLAG_LETTER,       # Other Letter
    "Lt": CODEPOINT_FLAG_LETTER,       # Titlecase Letter
    "Lu": CODEPOINT_FLAG_LETTER,       # Uppercase Letter
    "L&": CODEPOINT_FLAG_LETTER,       # Cased Letter
    "Mc": CODEPOINT_FLAG_MARK,         # Spacing Mark
    "Me": CODEPOINT_FLAG_MARK,         # Enclosing Mark
    "Mn": CODEPOINT_FLAG_MARK,         # Nonspacing Mark
    "Nd": CODEPOINT_FLAG_NUMBER,       # Decimal Number
    "Nl": CODEPOINT_FLAG_NUMBER,       # Letter Number
    "No": CODEPOINT_FLAG_NUMBER,       # Other Number
    "Pc": CODEPOINT_FLAG_PUNCTUATION,  # Connector Punctuation
    "Pd": CODEPOINT_FLAG_PUNCTUATION,  # Dash Punctuation
    "Pe": CODEPOINT_FLAG_PUNCTUATION,  # Close Punctuation
    "Pf": CODEPOINT_FLAG_PUNCTUATION,  # Final Punctuation
    "Pi": CODEPOINT_FLAG_PUNCTUATION,  # Initial Punctuation
    "Po": CODEPOINT_FLAG_PUNCTUATION,  # Other Punctuation
    "Ps": CODEPOINT_FLAG_PUNCTUATION,  # Open Punctuation
    "Sc": CODEPOINT_FLAG_SYMBOL,       # Currency Symbol
    "Sk": CODEPOINT_FLAG_SYMBOL,       # Modifier Symbol
    "Sm": CODEPOINT_FLAG_SYMBOL,       # Math Symbol
    "So": CODEPOINT_FLAG_SYMBOL,       # Other Symbol
    "Zl": CODEPOINT_FLAG_SEPARATOR,    # Line Separator
    "Zp": CODEPOINT_FLAG_SEPARATOR,    # Paragraph Separator
    "Zs": CODEPOINT_FLAG_SEPARATOR,    # Space Separator
}


codepoint_flags = array.array('H', [CODEPOINT_FLAG_UNDEFINED]) * MAX_CODEPOINTS
table_whitespace = []
table_lowercase = []
table_uppercase = []
table_nfd = []

for (cpt, cpt_lower, cpt_upper, categ, bidir) in unicode_data_iter():
    # codepoint category flags
    codepoint_flags[cpt] = UNICODE_CATEGORY_TO_FLAG[categ]

    # lowercase conversion
    if cpt_lower:
        table_lowercase.append((cpt, cpt_lower))

    # uppercase conversion
    if cpt_upper:
        table_uppercase.append((cpt, cpt_upper))

    # NFD normalization
    norm = nfd_first(cpt)
    if cpt != norm:
        table_nfd.append((cpt, norm))


# whitespaces, see "<White_Space>" https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt
table_whitespace.extend(range(0x0009, 0x000D + 1))
table_whitespace.extend(range(0x2000, 0x200A + 1))
table_whitespace.extend([0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000])


# sort by codepoint
table_whitespace.sort()
table_lowercase.sort()
table_uppercase.sort()
table_nfd.sort()


# group ranges with same flags
ranges_flags: list[tuple[int, int]] = [(0, codepoint_flags[0])]  # start, flags
for codepoint, flags in enumerate(codepoint_flags):
    if flags != ranges_flags[-1][1]:
        ranges_flags.append((codepoint, flags))
ranges_flags.append((MAX_CODEPOINTS, 0x0000))


# group ranges with same nfd
ranges_nfd: list[tuple[int, int, int]] = [(0, 0, 0)]  # start, last, nfd
for codepoint, norm in table_nfd:
    start = ranges_nfd[-1][0]
    if ranges_nfd[-1] != (start, codepoint - 1, norm):
        ranges_nfd.append((0, 0, 0))  # dummy, will be replaced below
        start = codepoint
    ranges_nfd[-1] = (start, codepoint, norm)


# Generate 'unicode-data.cpp':
#   python ./scripts//gen-unicode-data.py > unicode-data.cpp

def out(line=""):
    print(line, end='\n')  # noqa


out(f"""\
// generated with scripts/gen-unicode-data.py from UCD {args.ucd_version}

#include "unicode-data.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
""")

out("const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {  // start, flags // last=next_start-1")
for codepoint, flags in ranges_flags:
    out("{0x%06X, 0x%04X}," % (codepoint, flags))
out("};\n")

out("const std::unordered_set<uint32_t> unicode_set_whitespace = {")
for codepoint in table_whitespace:
    out("0x%06X," % codepoint)
out("};\n")

out("// list is always in ascending order, to enable binary search")
out("const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase = {")
for tuple_lw in table_lowercase:
    out("{0x%06X, 0x%06X}," % tuple_lw)
out("};\n")

out("// list is always in ascending order, to enable binary search")
out("const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase = {")
for tuple_up in table_uppercase:
    out("{0x%06X, 0x%06X}," % tuple_up)
out("};\n")

out("const std::initializer_list<range_nfd> unicode_ranges_nfd = {  // start, last, nfd")
for triple in ranges_nfd:
    out("{0x%06X, 0x%06X, 0x%06X}," % triple)
out("};\n")

normalization_tables()
