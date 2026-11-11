#!/usr/bin/env bash
# run_all.sh <input_dir> <output_dir>: add reBruce fake-data weights to every sBruce file.
# Re-runs itself in a clean env so cafpyana's python/ROOT don't shadow reBruce's.
[ -z "$_CLEAN" ] && exec env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin _CLEAN=1 \
    bash --norc "$(readlink -f "$0")" "$(realpath -m "$1")" "$(realpath -m "$2")"

cd /exp/sbnd/app/users/nrowe/reBruce || exit 1
source /cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.1.1/setup-env.sh
spack load --first root@6.28.12 ^python@3.11   # pin python: --first alone can pick a py3.10 build
source venv/bin/activate

mkdir -p "$2"
for f in "$1"/*.root; do
    out="$2/$(basename "$f" .root)_fakedata.root"
    python reweight.py configs/all_calculators.yaml --input "$f" --output "$out" \
        --stl-vectors --skip-incomplete || cp "$f" "$out"
done
