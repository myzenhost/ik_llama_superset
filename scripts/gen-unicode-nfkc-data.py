# Generates src/unicode-nfkc-data.cpp: the tables unicode_cpts_normalize_nfkc() needs (UAX #15).
#   - full compatibility decomposition (NFKD) of every code point that has one, Hangul syllables excluded
#     (those decompose algorithmically);
#   - canonical combining class ranges (ccc != 0);
#   - primary composite pairs: (first, second) -> composite for every canonical two-code-point decomposition
#     that NFC recomposes (composition exclusions and non-starter decompositions drop out by that test).
# The source is Python's unicodedata, so the Unicode version is the interpreter's (printed into the header).
#
# usage: python scripts/gen-unicode-nfkc-data.py > src/unicode-nfkc-data.cpp

from __future__ import annotations

import sys
import unicodedata

MAX_CODEPOINTS = 0x110000
S_BASE, S_COUNT = 0xAC00, 11172


def is_scalar(cpt: int) -> bool:
    return not (0xD800 <= cpt <= 0xDFFF)


def main() -> None:
    nfkd: list[tuple[int, list[int]]] = []
    ccc_ranges: list[list[int]] = []
    pairs: list[tuple[int, int, int]] = []

    for cpt in range(MAX_CODEPOINTS):
        if not is_scalar(cpt):
            continue
        ch = chr(cpt)

        if not (S_BASE <= cpt < S_BASE + S_COUNT):
            d = unicodedata.normalize("NFKD", ch)
            if d != ch:
                nfkd.append((cpt, [ord(c) for c in d]))

        cc = unicodedata.combining(ch)
        if cc:
            if ccc_ranges and ccc_ranges[-1][1] == cpt - 1 and ccc_ranges[-1][2] == cc:
                ccc_ranges[-1][1] = cpt
            else:
                ccc_ranges.append([cpt, cpt, cc])

        dec = unicodedata.decomposition(ch)
        if dec and not dec.startswith("<"):
            parts = [int(p, 16) for p in dec.split()]
            if len(parts) == 2 and unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == ch:
                pairs.append((parts[0], parts[1], cpt))

    pairs.sort()

    out = sys.stdout
    out.write("// generated with scripts/gen-unicode-nfkc-data.py from Python unicodedata, Unicode %s\n\n"
              % unicodedata.unidata_version)
    out.write('#include "unicode-nfkc-data.h"\n\n')

    flat: list[int] = []
    out.write("const unicode_nfkd_entry unicode_nfkd_map[] = {  // cpt, offset into unicode_nfkd_data, length\n")
    for cpt, seq in nfkd:
        out.write("{0x%06X, %d, %d},\n" % (cpt, len(flat), len(seq)))
        flat.extend(seq)
    out.write("};\n")
    out.write("const size_t unicode_nfkd_map_size = sizeof(unicode_nfkd_map) / sizeof(unicode_nfkd_map[0]);\n\n")

    out.write("const uint32_t unicode_nfkd_data[] = {\n")
    for i in range(0, len(flat), 12):
        out.write(", ".join("0x%06X" % c for c in flat[i:i + 12]) + ",\n")
    out.write("};\n\n")

    out.write("const unicode_ccc_range unicode_ccc_ranges[] = {  // first, last, ccc\n")
    for first, last, cc in ccc_ranges:
        out.write("{0x%06X, 0x%06X, %d},\n" % (first, last, cc))
    out.write("};\n")
    out.write("const size_t unicode_ccc_ranges_size = sizeof(unicode_ccc_ranges) / sizeof(unicode_ccc_ranges[0]);\n\n")

    out.write("const unicode_compose_entry unicode_compose_pairs[] = {  // first, second, composite (sorted)\n")
    for a, b, c in pairs:
        out.write("{0x%06X, 0x%06X, 0x%06X},\n" % (a, b, c))
    out.write("};\n")
    out.write("const size_t unicode_compose_pairs_size = sizeof(unicode_compose_pairs) / sizeof(unicode_compose_pairs[0]);\n")

    print("nfkd %d entries (%d code points), ccc %d ranges, compose %d pairs"
          % (len(nfkd), len(flat), len(ccc_ranges), len(pairs)), file=sys.stderr)


if __name__ == "__main__":
    main()
