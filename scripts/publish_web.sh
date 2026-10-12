#!/usr/bin/env bash
# Publish web/dist (built by scripts/export_web.py) to the gh-pages branch.
# The model only lives on gh-pages, so main stays free of large binaries.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f web/dist/index.html ] || { echo "Run: python scripts/export_web.py"; exit 1; }
remote=$(git remote get-url origin)
tmp=$(mktemp -d)
cp -R web/dist/. "$tmp"
touch "$tmp/.nojekyll"  # serve files as-is, no Jekyll build
cd "$tmp"
git init -q -b gh-pages
git add -A
git commit -q -m "Deploy web demo from $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push -q -f "$remote" gh-pages
rm -rf "$tmp"
echo "Pushed. Live at https://prathameshh-patil.github.io/intentbench/ in a minute or two."
