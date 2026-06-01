#!/bin/sh

# SPDX-FileCopyrightText:  2020-2026 Cameron Kaiser, Trung Lê
# SPDX-License-Identifier: GPL-2.0-or-later

#
# Generate a downstream patch that restores PPC64LE (POWER) dynamic
# recompiler support on top of an upstream DOSBox Staging source tree.
#
# Upstream PR #4796 removed the PPC64LE dynrec backend along with all other
# "unsupported" platforms. This script extracts only the source and build-system
# changes needed to put it back, producing a single self-contained patch that
# downstream packagers (Fedora, Debian, Ubuntu, ...) can apply to a pristine
# upstream release tarball with `git apply`, `patch -p1`, or quilt.
#
# Upstream is now CMake-only (Meson was removed), so the patch touches only the
# CMake build system and the C++ sources.
#
# The patch deliberately EXCLUDES fork-specific changes that do not belong in a
# downstream tree:
#   - CI workflow rewrites (.github/...)
#   - cross-compilation package lists (packages/...)
#

set -e

# --- Configuration -----------------------------------------------------------

# Commit that restored the PPC64LE backend. Override with the RESTORE_COMMIT
# env var, or let the script locate it by commit message.
RESTORE_COMMIT="${RESTORE_COMMIT:-}"

# Output patch file. Override with the first positional argument.
OUTPUT="${1:-dosbox-staging-restore-ppc64le.patch}"

# Files that make up the PPC64LE restoration (CMake-only).
PATCH_PATHS="
CMakeLists.txt
src/cpu/core_dynrec.cpp
src/cpu/core_dynrec/risc_ppc64le.h
src/dosbox_config.h.in.cmake
src/misc/support.h
"

# Upstream reference, recorded in the patch header for provenance.
UPSTREAM_PR="https://github.com/dosbox-staging/dosbox-staging/pull/4796"

# --- Helpers -----------------------------------------------------------------

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

# --- Locate the repo and the restoration commit ------------------------------

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "must be run inside the dosbox-staging-power git repository"

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

if [ -z "$RESTORE_COMMIT" ]; then
    RESTORE_COMMIT=$(git log --format=%H \
        --grep='Restore PPC64LE dynamic recompiler backend' -1)
    [ -n "$RESTORE_COMMIT" ] \
        || die "could not find the PPC64LE restoration commit; set RESTORE_COMMIT"
fi

git cat-file -e "${RESTORE_COMMIT}^{commit}" 2>/dev/null \
    || die "commit '$RESTORE_COMMIT' not found"

# The restoration commit's parent is the pristine upstream state for these
# files; diffing it against HEAD captures the restoration *and* every later
# fix/optimisation to the backend (mask/sign-extension fixes, the CMP/TEST
# fix, the POWER10 ISA 3.1 fast path, ...), not just the first commit.
BASE_COMMIT="${RESTORE_COMMIT}^"
BASE_SHORT=$(git rev-parse --short "$BASE_COMMIT")
HEAD_SHORT=$(git rev-parse --short HEAD)
AUTHOR=$(git show -s --format='%an <%ae>' "$RESTORE_COMMIT")
DATE=$(git show -s --format='%ad' --date=short HEAD)

# --- Sanity-check that all expected paths exist at HEAD ----------------------

for path in $PATCH_PATHS; do
    git cat-file -e "HEAD:${path}" 2>/dev/null \
        || die "expected file '$path' not present at HEAD; refusing to emit incomplete patch"
done

# --- Emit the patch ----------------------------------------------------------

{
    # DEP-3 header (https://dep-team.pages.debian.net/deps/dep3/). Lines before
    # the first "diff --git" are ignored by `git apply` and `patch`, so this is
    # safe for every consumer while giving Debian/quilt a structured header.
    printf 'Description: Restore PPC64LE (POWER) dynamic recompiler backend\n'
    printf ' Restores the little-endian PowerPC64 dynrec backend that upstream\n'
    printf ' removed in PR #4796 when it pruned "unsupported" platforms. Adds the\n'
    printf ' PPC64LE target to the CMake build system and restores\n'
    printf ' src/cpu/core_dynrec/risc_ppc64le.h. PPC64LE is little-endian, so it\n'
    printf ' does not need the big-endian code paths that PR also removed.\n'
    printf 'Author: %s\n' "$AUTHOR"
    printf 'Origin: backport, dosbox-staging-power, %s..%s\n' "$BASE_SHORT" "$HEAD_SHORT"
    printf 'Bug: %s\n' "$UPSTREAM_PR"
    printf 'Forwarded: not-needed\n'
    printf 'Last-Update: %s\n' "$DATE"
    printf '\n'

    # Plain unified diff (-p1) of the complete backend state against pristine
    # upstream. Applies with:
    #   git apply <patch>      patch -p1 < <patch>      quilt import / push
    git diff "$BASE_COMMIT" HEAD -- $PATCH_PATHS
} > "$OUTPUT"

# --- Report ------------------------------------------------------------------

LINES=$(wc -l < "$OUTPUT" | tr -d ' ')
printf 'Wrote %s (%s lines) from %s..%s\n' "$OUTPUT" "$LINES" "$BASE_SHORT" "$HEAD_SHORT"
printf 'Apply with: patch -p1 < %s   (or: git apply %s)\n' "$OUTPUT" "$OUTPUT"
