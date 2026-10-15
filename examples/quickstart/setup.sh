#!/usr/bin/env bash
# Builds a self-contained playground: a local git remote, a local DVC remote, a
# shared model store, a producer project and a consumer app. Nothing leaves
# your machine.   Usage: ./setup.sh [target-folder]   (default ~/mldna-quickstart)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$HOME/mldna-quickstart}"
if [ -e "$TARGET" ]; then
    echo "$TARGET already exists. Remove it, or pass another folder: ./setup.sh <folder>" >&2
    exit 1
fi

mkdir -p "$TARGET/remotes/dvc-store" "$TARGET/model-store"
TARGET="$(cd "$TARGET" && pwd)"
git init -q --bare "$TARGET/remotes/producer.git"

mldna init churn --dir "$TARGET/producer" --namespace quickstart \
    --tracking-source filesystem --filesystem-path "$TARGET/model-store" \
    --dvc-backend local --dvc-local-path "$TARGET/remotes/dvc-store"
cp "$HERE/producer/train.py" "$TARGET/producer/"
cp "$HERE/producer/dataset/customers.csv" "$TARGET/producer/dataset/"
printf 'tracking:\n  key_metric: accuracy\n' >> "$TARGET/producer/mldna.yaml"
git -C "$TARGET/producer" remote add origin "$TARGET/remotes/producer.git"

mldna init --consumer --dir "$TARGET/consumer" \
    --tracking-source filesystem --filesystem-path "$TARGET/model-store"
cp "$HERE/consumer/app.py" "$TARGET/consumer/"

echo
echo "Ready in $TARGET - next: cd \"$TARGET/producer\" and follow step 3 of the README."
