# DOSBox Staging for POWER

**DOSBox Staging POWER** is the restoration of dropped Linux PPC64LE JIT backend with the goal of creating downstream patch for Linux packagers to consume. If you run into bugs with PPC64LE, please file [bug report](https://github.com/runlevel5/dosbox-staging-power/issues) here.

## History

- The PPC64LE dynamic recompiler backend was initially written by Cameron
  Kaiser, documented in
  [DOSBox JIT on ppc64le and how you can too](https://www.talospace.com/2020/01/dosbox-jit-on-ppc64le-and-how-you-can.html).
- It was then upstreamed into DOSBox Staging via
  [pull request #2828](https://github.com/dosbox-staging/dosbox-staging/pull/2828).
- The DOSBox Staging project later decided to drop PPC64LE support, citing
  maintenance concerns and a lack of ppc64 expertise
  ([PR #4796](https://github.com/dosbox-staging/dosbox-staging/pull/4796)).

This fork restores that backend and maintains it as a downstream patch.

## Generating the downstream patch

Linux packagers (Fedora, Debian, Ubuntu, ...) can restore PPC64LE support on
top of a pristine upstream DOSBox Staging release with a single patch generated
from this repository.

Run the generator from inside a clone of this repository:

```sh
./scripts/packaging/generate-ppc64le-patch.sh [output.patch]
```

With no argument it writes `dosbox-staging-restore-ppc64le.patch` in the current
directory. For example:

```sh
git clone https://github.com/runlevel5/dosbox-staging-power.git
cd dosbox-staging-power
./scripts/packaging/generate-ppc64le-patch.sh /tmp/dosbox-staging-ppc64le.patch
```

The script collects the complete current state of the backend — the restored
`src/cpu/core_dynrec/risc_ppc64le.h`, the CMake build-system hooks, and every
later fix and optimisation — and emits a single self-contained unified diff with
a [DEP-3](https://dep-team.pages.debian.net/deps/dep3/) header. It touches only
five files and deliberately excludes fork-specific bits (CI workflows,
cross-compilation package lists). Set `RESTORE_COMMIT=<sha>` to override the
auto-detected restoration commit.

Apply the patch to an unpacked upstream release tree with any of:

```sh
patch -p1 < dosbox-staging-restore-ppc64le.patch
# or
git apply dosbox-staging-restore-ppc64le.patch
# or, for Debian quilt
quilt import dosbox-staging-restore-ppc64le.patch && quilt push
```

### Build notes for packagers

- Upstream is CMake-only; configure with the system-libraries preset, e.g.
  `cmake --preset release-linux`.
- The backend runs on **POWER8 and later**. POWER10 (PowerISA 3.1) instructions
  are emitted only when detected at run time (`getauxval(AT_HWCAP2) &
  PPC_FEATURE2_ARCH_3_1`), so a single binary built with a POWER8 baseline
  (`-mcpu=power8`) stays portable and still uses the POWER10 fast path where
  available.
