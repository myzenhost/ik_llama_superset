# simplewebp (vendored)

Single-header WebP decoder (lossy VP8, lossless VP8L, ALPH alpha; no animation, no encode).
Used by tools/mtmd/mtmd-helper.cpp to decode webp images in-process (no ffmpeg needed).

- Upstream: https://github.com/MikuAuahDark/simplewebp
- Upstream HEAD at fetch (2026-11-10): 679ee727829df44962e3de0cb8fbcc3db9dab962 (header SIMPLEWEBP_VERSION 20260718)
- Licence: BSD-3-Clause + WebM patent grant (header tail block read in full, matches LICENSE.md / PATENTS.md).
- Files copied unmodified from raw.githubusercontent.com/.../master/:
  - simplewebp.h  sha256 aa0ec53ec3131fe276eaf0cf28e48249d4d6eace177e4ea22e18a79e56831e90
  - LICENSE.md    sha256 5e2dcd8fc2457501234b2cd2288304c7acdd22318bec96ebfd6a3876de4c518a
  - PATENTS.md    sha256 d578b7a471d48406a16c64e3c6a8685e66292f3c6709214c49ca55b2f22ed7cd
- SIMPLEWEBP_IMPLEMENTATION is defined in exactly one TU: tools/mtmd/mtmd-helper.cpp.
