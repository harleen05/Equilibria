#!/usr/bin/env bash
# Download PPO checkpoints (HF Spaces reject binary files in git pushes).
#
# SECURITY NOTE: previously this fetched model checkpoints from a
# third-party fork over plain curl with NO integrity verification.
# stable-baselines3's PPO.load() deserializes via cloudpickle, so loading
# a tampered checkpoint is a real code-execution risk if that upstream
# repo (or its commit history) is ever compromised. Every checkpoint is
# now verified against a SHA256 pinned to the default commit ref below --
# computed and verified directly against the real files at that ref
# (confirmed to be genuine, loadable stable-baselines3 PPO checkpoints
# with the expected Discrete(25) action space for this environment)
# before being pinned here.
#
# If you override MODELS_GIT_REF to point at a different commit, checksum
# verification is skipped for that run (the pinned hashes only apply to
# the default ref) -- you'll see a clear warning either way.
#
# NOTE ON PORTABILITY: this deliberately avoids `declare -A` (associative
# arrays), since macOS's default /bin/bash is bash 3.2 (Apple hasn't
# shipped a newer bash since a GPLv3 licensing change) and does not
# support them at all -- using one here previously broke on any Mac that
# hadn't separately installed a newer bash via Homebrew. A case statement
# works identically on bash 3.2 through 5.x.
set -euo pipefail

DEFAULT_REF="8b4e416"
REF="${MODELS_GIT_REF:-$DEFAULT_REF}"
BASE="https://raw.githubusercontent.com/harleen05/Equilibria/${REF}/models"

mkdir -p models/best/easy models/best/medium models/best/hard

# SHA256 checksums pinned to commit 8b4e416. Verified directly against the
# real files at that ref: valid stable-baselines3 PPO zip archives
# (data, policy.pth, policy.optimizer.pth, ...) that load successfully
# with the expected observation/action space for this environment.
expected_checksum_for() {
  case "$1" in
    "ppo_easy_final.zip")
      echo "1422baafdfc0c205057cdc737fd7a4e811305ae63b4f8b0d1a08f6907d3b0261" ;;
    "ppo_medium_final.zip")
      echo "1c44bcef59ab784e542ad8fceb7adf3f9bc9525b113cd0865eb51d05a3c31dd7" ;;
    "ppo_hard_final.zip")
      echo "1d1ec434d8ed1e697af0679b6b007e8d8290ef43c908ba7f7c59e347e4311cff" ;;
    "best/easy/best_model.zip")
      echo "d76545367729a1a74c2dc7db049517c6919c24b22dda9c2e4413e50930409e9d" ;;
    "best/medium/best_model.zip")
      echo "af4955f62d0fd7a92b02203bf1cce1dc3f6de57294feecd15e45c1e1685767b6" ;;
    "best/hard/best_model.zip")
      echo "d2a55aa95678c3d7283425a4352a9251012c79f1e0cade23cd026fa7a53aaa98" ;;
    *)
      echo "" ;;
  esac
}

verify_checksum() {
  local dest="$1"
  local name="$2"

  if [ "${REF}" != "${DEFAULT_REF}" ]; then
    echo "  ⚠ MODELS_GIT_REF overridden (${REF}) -- skipping checksum verification for ${name}"
    return 0
  fi

  local expected
  expected="$(expected_checksum_for "${name}")"
  if [ -z "${expected}" ]; then
    echo "  ⚠ No pinned checksum for ${name} -- skipping verification" >&2
    return 0
  fi

  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${dest}" | cut -d' ' -f1)"
  else
    actual="$(shasum -a 256 "${dest}" | cut -d' ' -f1)"
  fi

  if [ "${actual}" != "${expected}" ]; then
    echo "  ✗ CHECKSUM MISMATCH for ${dest}" >&2
    echo "      expected: ${expected}" >&2
    echo "      actual:   ${actual}" >&2
    echo "    Removing untrusted file. Refusing to continue." >&2
    rm -f "${dest}"
    exit 1
  fi
  echo "  ✓ checksum verified"
}

fetch() {
  local dest="$1"
  local name="$2"
  echo "→ ${dest}"
  curl -fsSL "${BASE}/${name}" -o "${dest}"
  verify_checksum "${dest}" "${name}"
}

fetch models/ppo_easy_final.zip ppo_easy_final.zip
fetch models/ppo_medium_final.zip ppo_medium_final.zip
fetch models/ppo_hard_final.zip ppo_hard_final.zip
fetch models/best/easy/best_model.zip best/easy/best_model.zip
fetch models/best/medium/best_model.zip best/medium/best_model.zip
fetch models/best/hard/best_model.zip best/hard/best_model.zip

echo "✓ Models ready under models/"