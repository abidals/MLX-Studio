#!/bin/bash
#
# Build a SELF-CONTAINED "Alis Studio.app" and wrap it in a drag-to-Applications .dmg.
#
# The bundle ships its own standalone Python interpreter AND all runtime deps (mlx, mflux,
# transformers, pywebview, the Krea 2 Turbo backend, …) inside Contents/Resources. Double-click
# to run — no system Python, no `pip install`, nothing to set up.
#
# Caveat that no DMG can avoid: the *model weights* (10–50 GB) are NOT bundled — they download
# from Hugging Face on the first image generation. The app runs offline; the first picture needs
# network + disk.
#
#   bash packaging/build_dmg.sh
#   → dist/Alis-Studio-<version>.dmg   (and dist/Alis Studio.app)
#
# Requires uv (https://docs.astral.sh/uv/) to fetch a relocatable CPython and resolve deps fast.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
APP="$DIST/Alis Studio.app"
RES="$APP/Contents/Resources"
BUNDLE_ID="com.avlp12.alis-studio"
PYVER="${ALIS_PY_VERSION:-3.13}"

command -v uv >/dev/null 2>&1 || { echo "error: uv is required — https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }

# --- version: single source of truth is studio/__version__ (pipefail-safe, first match) ------
VERSION="$(awk -F'"' '/^__version__[[:space:]]*=/{print $2; exit}' "$ROOT/studio/__init__.py")"
[ -n "$VERSION" ] || { echo "error: could not read __version__ from studio/__init__.py" >&2; exit 1; }
echo "Alis Studio $VERSION → building self-contained .app + .dmg (this downloads ~1 GB of deps the first time)"

# --- clean + scaffold ------------------------------------------------------------------------
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/app"

# --- 1) bundle a standalone, relocatable CPython --------------------------------------------
export UV_PYTHON_PREFERENCE=only-managed
uv python install "$PYVER" >/dev/null
PYBIN="$(uv python find "$PYVER")"
# resolve symlinks: the real install root is two levels up from the real executable
SRC_PY="$(python3 -c "import os;print(os.path.dirname(os.path.dirname(os.path.realpath('$PYBIN'))))")"
echo "  • bundling CPython from $SRC_PY"
cp -R "$SRC_PY" "$RES/python"
BPY="$RES/python/bin/python3"
"$BPY" -c 'import sys; assert sys.prefix.endswith("/python"), sys.prefix' \
  || { echo "error: bundled python did not relocate" >&2; exit 1; }
# this copy is now ours, not uv's — drop the "managed, don't modify" marker so we can install into it
find "$RES/python/lib" -name 'EXTERNALLY-MANAGED' -delete 2>/dev/null || true

# --- 2) install runtime deps INTO the bundled interpreter -----------------------------------
# Same layout as requirements.txt: mflux pinned to the main-HEAD commit that carries
# Qwen-Image-2.1 LoRA support (the Viggle Turbo backend; PRs #756/#768, after 0.20.0), then the
# Krea 2 backend WITHOUT its own dependency pins (they cap mflux<0.19 / mlx<0.32 and would
# downgrade the pin away — v0.3.1 is verified to run fine on mflux 0.20 / mlx 0.32), plus the
# desktop window and the optional prompt-enhancer. UV visits the Krea 2 package with --no-deps
# by splitting the install into a second invocation that lists krea2's remaining light deps.
echo "  • installing runtime deps into the bundle (mlx, mflux@HEAD, transformers, pywebview, mlx-lm)…"
uv pip install --python "$BPY" \
  "mflux @ git+https://github.com/mflux-community/mflux.git@0db686992d65cbd84bd925dfce23b77ad0bedf7d" \
  "pywebview>=5,<7" \
  "mlx-lm>=0.20" \
  "requests>=2.28,<3" \
  "numpy>=1.24,<3" \
  "pillow>=10,<13"
echo "  • installing the Krea 2 Turbo backend (no-deps) into the bundle…"
uv pip install --python "$BPY" --no-deps \
  "krea2-alis-mlx @ git+https://github.com/avlp12/krea2_alis_mlx.git@v0.3.1"
"$BPY" - <<'CHECK'
import importlib.util
missing = []
for mod in ("mflux.models.qwen21.weights.qwen21_lora_mapping",   # Viggle Turbo: 2.1 LoRA loader
            "krea2.pipeline",                                   # Krea 2 backend
            "webview",                                          # native desktop window
            "mlx_lm"):                                          # prompt enhancer extra
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)
assert not missing, f"bundle is missing runtime deps: {missing}"
print("bundle deps check: OK (mflux 2.1 LoRA + krea2 + pywebview + mlx-lm all importable)")
CHECK

# --- 3) the app code (pure-python studio/ + web/), beside the bundled interpreter ------------
cp -R "$ROOT/studio" "$RES/app/studio"
cp -R "$ROOT/web"    "$RES/app/web"
find "$RES/app" -name '__pycache__' -type d -prune -exec rm -rf {} +

# --- 4) icon (optional — degrade gracefully if Pillow OR icon generation fails) -------------
ICON_ARG=""
if python3 -c 'import PIL' >/dev/null 2>&1 \
   && python3 "$ROOT/packaging/make_icon.py" "$RES/AppIcon.icns" >/dev/null; then
  ICON_ARG="    <key>CFBundleIconFile</key>
    <string>AppIcon</string>"
else
  rm -f "$RES/AppIcon.icns" 2>/dev/null || true   # no dangling unreferenced icon
  echo "note: building without a custom icon (needs Pillow + iconutil; \`pip install pillow\`)" >&2
fi

# --- 5) Info.plist --------------------------------------------------------------------------
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Alis Studio</string>
    <key>CFBundleDisplayName</key>
    <string>Alis Studio</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleExecutable</key>
    <string>Alis Studio</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
$ICON_ARG
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.graphics-design</string>
</dict>
</plist>
PLIST

printf 'APPL????' > "$APP/Contents/PkgInfo"

# --- 6) launcher → the BUNDLED interpreter (no system Python involved) -----------------------
cat > "$APP/Contents/MacOS/Alis Studio" <<'LAUNCH'
#!/bin/bash
# Self-contained launcher: run the bundled interpreter against the bundled app code.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$(cd "$HERE/../Resources" && pwd)"
LOG="$HOME/Library/Logs/Alis Studio.log"
mkdir -p "$(dirname "$LOG")"
cd "$RES/app"
# Never write .pyc into the (signed, read-only) bundle — that would invalidate the code seal.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$HOME/Library/Caches/Alis Studio/pycache"
export PYTHONPATH="$RES/app"
exec "$RES/python/bin/python3" -m studio.desktop >>"$LOG" 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/Alis Studio"

# --- 7) sign (ad-hoc, deep — seals the interpreter + every bundled dylib) --------------------
# Strip any pyc the install wrote into app code; the python's own stdlib/site-packages pyc are
# fine (present at sign time, never rewritten at runtime thanks to PYTHONDONTWRITEBYTECODE).
find "$RES/app" -name '__pycache__' -type d -prune -exec rm -rf {} +
xattr -cr "$APP" 2>/dev/null || true
echo "  • ad-hoc signing the bundle (many nested binaries — takes a moment)…"
if codesign --force --deep --sign - --timestamp=none "$APP" 2>/dev/null; then
  codesign --verify --deep --strict "$APP" >/dev/null 2>&1 && echo "  • signed + strict-verified."
else
  echo "note: ad-hoc deep-sign failed — a locally built app still runs unsigned (no quarantine on a local build)." >&2
fi

# --- 8) DMG ---------------------------------------------------------------------------------
DMG="$DIST/Alis-Studio-$VERSION.dmg"
rm -f "$DMG"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"   # drag-to-install target
hdiutil create -volname "Alis Studio $VERSION" -srcfolder "$STAGE" \
  -ov -format UDZO "$DMG" >/dev/null

echo ""
echo "✓ $APP  ($(du -sh "$APP" | cut -f1))"
echo "✓ $DMG  ($(du -h "$DMG" | cut -f1))"
echo "  Double-click the .app (or the DMG → drag to Applications). First image downloads model weights."
