#!/usr/bin/env bash
# Re-copy the benchmarked code from the private Efficient-Training checkout.
#
#   ./sync_from_upstream.sh [UPSTREAM_CHECKOUT] [BRANCH]
#
# Cloud.ru clones over public https from github.com only, and Efficient-Training is
# private, so the code the benchmark measures has to live here. This mirror carries
# the model, the three optimizers and the FP8 machinery -- not the training entry
# point, the datasets, the evaluation stack, the run scripts or the history.
#
# Run it after the upstream branch moves; the diff it leaves is the review.
set -euo pipefail

upstream=${1:-$HOME/Programming/Efficient-Training}
branch=${2:-codex/cloud-fp8-optimizer-states}
here=$(cd "$(dirname "$0")" && pwd)

[[ -d $upstream/.git ]] || { echo "not a checkout: $upstream" >&2; exit 2; }
git -C "$upstream" rev-parse --verify --quiet "$branch" >/dev/null || {
  echo "no such branch in $upstream: $branch" >&2
  exit 2
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git -C "$upstream" archive "$branch" | tar -x -C "$work"

copy() {
  local path=$1
  if [[ ! -e $work/$path ]]; then
    echo "  MISSING upstream: $path"
    return
  fi
  mkdir -p "$here/$(dirname "$path")"
  rm -rf "${here:?}/$path"
  cp -a "$work/$path" "$here/$path"
  echo "  $path"
}

echo "syncing $branch ($(git -C "$upstream" rev-parse --short "$branch")) from $upstream"

rm -rf "$here/src" "$here/third_party"

# The model and its FP8 forward path.
copy src/models

# The optimizer state quantizer, and the three optimizers the sweep measures.
# `src/optim/sota_opt/__init__.py` is deliberately NOT copied: importing it pulls in
# MARS, Adan, SWAN and Shampoo, none of which this benchmark builds. Without it
# `optim.sota_opt` resolves as a namespace package and the two SOAP modules import
# exactly as they do upstream. AdamW and Muon come from third_party, not from here.
copy src/optim/__init__.py
copy src/optim/fp8_state.py
copy src/optim/sota_opt/soap
copy src/optim/sota_opt/fp8_soap.py

# COAT: FP8 activations, the weight-scale cache and the FP8 AdamW. MuonLite: Muon.
copy third_party/__init__.py
copy third_party/coat
copy third_party/lite

find "$here/src" "$here/third_party" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$here/src" "$here/third_party" -name '*.pyc' -delete 2>/dev/null || true

printf '%s\n' \
  "upstream: $(git -C "$upstream" remote get-url origin)" \
  "branch:   $branch" \
  "commit:   $(git -C "$upstream" rev-parse "$branch")" \
  "synced:   $(date -u +%FT%TZ)" > "$here/UPSTREAM.txt"
cat "$here/UPSTREAM.txt"
