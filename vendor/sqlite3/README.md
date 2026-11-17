SQLite (amalgamation)

VERSION
  3.49.1 -- SQLITE_SOURCE_ID "2025-02-18 13:38:58 873d4e274b4988d260ba8354a9718324a1c26187a4ab4c1cc0227c03d0f10e70"
  sha256 sqlite3.c ff80c36ef1bb44eb357c7ff1d15be77540d41c28fb671088215a6cd12785c5d3
  sha256 sqlite3.h 88da6f1963bc192dfa18a3a48b423cf1fbbb04f903202efe9a78e3a597473e18

ORIGIN
  Official amalgamation 3.49.1, taken byte-identical from the libsqlite3-sys 0.33.0 crate
  (crates.io, crate checksum 947e6816f7825b2b45027c2c32e7085da9934defa535de4a6a46b10a4d5257fa,
  verified by cargo against the Cargo.lock of sqlite-zstd v0.3.5), directory sqlite3/. That crate
  bundles the unmodified amalgamation (its upgrade.sh fetches it from sqlite.org).

WHY 3.49.1 (was 3.47.1 until 2026-09-27)
  The sqlite-zstd loadable extension (G4, --sqlite-zstd-ext-file) is built on libsqlite3-sys 0.33,
  whose loadable_extension init refuses an older host library. MEASURED with 3.47.1:
    failed to load '...sqlite_zstd.dll': error during initialization: SQLite version mismatch:
    3047001 < 3049001
  The previous 3.47.1 copy came from llamafile's third_party/sqlite and carried llamafile's
  "#includez" mangling (4 sites, one de-mangled here). This copy is the plain amalgamation, so no
  local edits are carried any more.

LICENSE
  Public domain (SQLite's own declaration; see the header of sqlite3.c/.h).

WHAT IS AND IS NOT VENDORED HERE
  sqlite3.c + sqlite3.h only (the amalgamated core engine + its public C API). NOT vendored:
  shell.c (the sqlite3 CLI shell, unneeded), sqlite3ext.h (needed only by code that IS a loadable
  extension, not by an application loading one -- sqlite3_load_extension/_enable_load_extension
  are declared in sqlite3.h itself).

USED BY
  tools/server's G3 SQL session store and G4 zstd routes (rows
  S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01,
  S69-G4-ZSTD-ROUTES-PORT-USER-APPROVED-2026-09-27-01), gated behind the CMake option
  LLAMA_SERVER_SQLITE3 (default OFF) via the sqlite_modern_cpp wrapper at
  tools/server/sqlite_modern_cpp/.

BUILD FLAGS
  Core engine only (no FTS5/zlib flags). A future consumer that needs one should add the matching
  -D as a target_compile_definitions on the vendor-sqlite3 target, not re-vendor.
