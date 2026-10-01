# Local browser compiler toolchain

Source: https://github.com/binji/wasm-clang
Pinned revision: `648c4a89997a351eef75cdaec3ef5b89d4937dec`

`clang`, `lld`, `memfs`, `sysroot.tar`, `shared.js`, `LICENSE` and `LICENSE.llvm`
are copied without modification. Their Git blob hashes are verified by
`fetch-toolchain.cjs`. Do not replace them with moving-branch downloads.

The upstream JavaScript shim is Apache-2.0; LLVM artifacts and their exceptions
are covered by the retained `LICENSE.llvm`. This project does not ship the
unrelated 6502 assembler or the upstream Ace/xterm UI.

The product-specific Worker adapter is `/wasm-worker.js`. It wraps the unmodified
API with C++17 flags, UTF-8 source/output handling, phase reporting, bounded output
and separate compilation/execution workers. No canvas is connected to the page.

This is an old experimental toolchain, not a complete modern C++ environment.
The known `std::sort` / `__lttf2` sysroot defect remains visible as a link error.
