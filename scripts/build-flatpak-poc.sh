#!/usr/bin/env bash
set -euo pipefail

# Local proof-of-concept builder:
# 1) Build/reuse a PyInstaller Linux executable
# 2) Wrap it into a Flatpak payload + manifest
# 3) Build a local Flatpak repo and bundle (.flatpak)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="OpenGS-MapTool"
APP_ID="org.opengs.MapTool"
APP_SLUG="opengs-maptool"
PROJECT_PACKAGE="opengs_maptool"
CONFIG_PATH="${PROJECT_PACKAGE}/config.py"
ENTRYPOINT_PATH="${PROJECT_PACKAGE}/main.py"
ICON_SOURCE_PATH="${ROOT_DIR}/ogs-mt-icon-master.ico"
ICON_THEME_REL_DIR="icons/hicolor/256x256/apps"
ICON_FILENAME="${APP_ID}.png"
ICON_PAYLOAD_REL_PATH="${ICON_THEME_REL_DIR}/${ICON_FILENAME}"
ICON_INSTALL_PATH="/app/share/icons/hicolor/256x256/apps/${ICON_FILENAME}"
APP_SHARE_PATH="/app/share/${APP_SLUG}"
RUNTIME="org.freedesktop.Platform"
SDK="org.freedesktop.Sdk"
RUNTIME_VERSION="${RUNTIME_VERSION:-24.08}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
KEEP_TEMP="${KEEP_TEMP:-0}"

WORK_DIR="${ROOT_DIR}/.flatpak-poc"
PAYLOAD_DIR="${WORK_DIR}/payload"
BUILD_DIR="${WORK_DIR}/build-dir"
REPO_DIR="${WORK_DIR}/repo"
OUT_DIR="${ROOT_DIR}/dist-flatpak"
MANIFEST_PATH="${WORK_DIR}/${APP_ID}.json"
DESKTOP_PATH="${PAYLOAD_DIR}/${APP_ID}.desktop"
BUNDLE_PATH="${OUT_DIR}/${APP_NAME}-PLACEHOLDER.flatpak"
ICON_PAYLOAD_PATH="${PAYLOAD_DIR}/${ICON_PAYLOAD_REL_PATH}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

cleanup_temp() {
  if [[ "${KEEP_TEMP}" == "1" ]]; then
    echo "Keeping temporary files (KEEP_TEMP=1)."
    return 0
  fi

  rm -rf "${WORK_DIR}" "${ROOT_DIR}/.flatpak-builder"
  echo "Cleaned temporary files: ${WORK_DIR} and ${ROOT_DIR}/.flatpak-builder"
}

ensure_pip() {
  if "${PYTHON_CMD}" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "${PYTHON_CMD} has no pip; bootstrapping with ensurepip..."
  if "${PYTHON_CMD}" -m ensurepip --upgrade >/dev/null 2>&1; then
    return 0
  fi

  return 1
}

pip_install() {
  if "${PYTHON_CMD}" -m pip --version >/dev/null 2>&1; then
    "${PYTHON_CMD}" -m pip "$@"
    return 0
  fi

  echo "pip is unavailable for interpreter ${PYTHON_CMD}." >&2
  exit 1
}

pick_python_with_pip() {
  local candidate
  for candidate in "${PYTHON_CMD}" /usr/bin/python3 python3; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" -m pip --version >/dev/null 2>&1; then
      PYTHON_CMD="${candidate}"
      return 0
    fi
  done

  # Keep the user-selected/default interpreter; ensure_pip() will attempt bootstrap.
  return 0
}

# Interactive pause point for step-by-step execution.
# Disable globally by setting NO_BREAKPOINTS=1.
breakpoint() {
  local name="$1"
  if [[ "${NO_BREAKPOINTS:-0}" == "1" ]]; then
    return 0
  fi

  echo
  echo "[breakpoint] ${name}"
  echo "Press Enter to continue, or type 'skip' to disable remaining breakpoints."
  local answer
  read -r answer
  if [[ "$answer" == "skip" ]]; then
    export NO_BREAKPOINTS=1
    echo "Breakpoints disabled for the rest of this run."
  fi
}

for cmd in flatpak flatpak-builder python3; do
  require_cmd "$cmd"
done

pick_python_with_pip
echo "Using Python interpreter: ${PYTHON_CMD}"

# Read the release version directly from config.py so this output mirrors CI.
VERSION="$(${PYTHON_CMD} - <<'PY'
import pathlib
import re
import sys

text = pathlib.Path('opengs_maptool/config.py').read_text(encoding='utf-8')
match = re.search(r'^VERSION\s*=\s*"([^"]+)"', text, re.M)
if not match:
  sys.exit('Could not read VERSION from config.py')
print(match.group(1))
PY
)"

BUNDLE_PATH="${OUT_DIR}/${APP_NAME}-${VERSION}.flatpak"

# Phase 1: ensure we have a Linux executable to package.

echo "Using version: ${VERSION}"
breakpoint "Version resolved and paths configured"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

if [[ ! -x "dist/${APP_NAME}" ]]; then
  breakpoint "About to build Linux executable with PyInstaller"
  echo "No Linux PyInstaller executable found in dist/. Building it now..."
  if ensure_pip; then
    pip_install install --upgrade pip
    pip_install install . pyinstaller
    "${PYTHON_CMD}" -m PyInstaller --noconfirm --clean --onefile --windowed \
      --name "${APP_NAME}" \
      --collect-all qtawesome \
      --exclude-module tkinter --exclude-module tkinterdnd2 --exclude-module pytest \
      "${ENTRYPOINT_PATH}"
  elif command -v uv >/dev/null 2>&1; then
    echo "pip unavailable; falling back to uv for PyInstaller build..."
    uv run --with pyinstaller pyinstaller --noconfirm --clean --onefile --windowed \
      --name "${APP_NAME}" \
      --collect-all qtawesome \
      --exclude-module tkinter --exclude-module tkinterdnd2 --exclude-module pytest \
      "${ENTRYPOINT_PATH}"
  else
    echo "Could not bootstrap pip via ensurepip and 'uv' is not installed." >&2
    echo "Install python3-pip or uv, or rerun with dist/${APP_NAME} prebuilt." >&2
    exit 1
  fi
fi

# Phase 2: prepare payload files that will be copied into /app in Flatpak.
echo "Preparing Flatpak build context in ${WORK_DIR}"
breakpoint "About to prepare Flatpak payload directory"
rm -rf "${WORK_DIR}"
mkdir -p "${PAYLOAD_DIR}/examples" "${PAYLOAD_DIR}/${ICON_THEME_REL_DIR}"

# Main executable + sample inputs + version marker for inspection/debugging.
cp "dist/${APP_NAME}" "${PAYLOAD_DIR}/${APP_NAME}"
cp -R examples/input "${PAYLOAD_DIR}/examples/"
echo "${VERSION}" > "${PAYLOAD_DIR}/VERSION"

# Desktop entry lets Flatpak expose the app in launchers.
cat > "${DESKTOP_PATH}" <<DESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=OpenGS MapTool
Comment=Create province maps and related files
Keywords=province;map;generator;gfx;
Exec=${APP_NAME}
Icon=${APP_ID}
Categories=Graphics;Utility;
Terminal=false
StartupNotify=true
DESKTOP

# Optional icon conversion when Pillow is available.
if [[ -f "${ICON_SOURCE_PATH}" ]]; then
  if "${PYTHON_CMD}" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('PIL') else 1)
PY
  then
    export ICON_SOURCE_PATH ICON_PAYLOAD_PATH
    "${PYTHON_CMD}" - <<'PY'
import os
from PIL import Image

ico = Image.open(os.environ['ICON_SOURCE_PATH'])
icon_out = os.environ['ICON_PAYLOAD_PATH']
ico.resize((256, 256), Image.Resampling.LANCZOS).save(icon_out, 'PNG')
PY
  fi
fi

# Manifest describes runtime, permissions, and install commands.
cat > "${MANIFEST_PATH}" <<MANIFEST
{
  "id": "${APP_ID}",
  "runtime": "${RUNTIME}",
  "runtime-version": "${RUNTIME_VERSION}",
  "sdk": "${SDK}",
  "command": "${APP_NAME}",
  "finish-args": [
    "--share=ipc",
    "--socket=fallback-x11",
    "--socket=wayland",
    "--filesystem=home"
  ],
  "modules": [
    {
      "name": "${APP_SLUG}",
      "buildsystem": "simple",
      "build-commands": [
        "install -Dm755 ${APP_NAME} /app/bin/${APP_NAME}",
        "install -Dm644 ${APP_ID}.desktop /app/share/applications/${APP_ID}.desktop",
        "mkdir -p ${APP_SHARE_PATH}/examples",
        "cp -R examples/input ${APP_SHARE_PATH}/examples/",
        "install -Dm644 VERSION ${APP_SHARE_PATH}/VERSION",
        "if [ -f ${ICON_PAYLOAD_REL_PATH} ]; then install -Dm644 ${ICON_PAYLOAD_REL_PATH} ${ICON_INSTALL_PATH}; fi"
      ],
      "sources": [
        {
          "type": "dir",
          "path": "${PAYLOAD_DIR}"
        }
      ]
    }
  ]
}
MANIFEST

breakpoint "Manifest and payload prepared"

# Phase 3: ensure runtime/SDK are installed for the selected branch.
echo "Ensuring Flathub remote and runtime/sdk are available"
breakpoint "About to install/update Flatpak runtime and SDK"
flatpak remote-add --if-not-exists --user flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak install --user -y flathub "${RUNTIME}//${RUNTIME_VERSION}" "${SDK}//${RUNTIME_VERSION}"

# Phase 4: build app repo, then produce a single-file distributable bundle.
echo "Building Flatpak"
breakpoint "About to run flatpak-builder"
flatpak-builder \
  --force-clean \
  --user \
  --install-deps-from=flathub \
  --repo="${REPO_DIR}" \
  "${BUILD_DIR}" \
  "${MANIFEST_PATH}"

echo "Creating bundle: ${BUNDLE_PATH}"
breakpoint "About to run flatpak build-bundle"
flatpak build-bundle \
  "${REPO_DIR}" \
  "${BUNDLE_PATH}" \
  "${APP_ID}" \
  --runtime-repo=https://dl.flathub.org/repo/flathub.flatpakrepo

echo "Done: ${BUNDLE_PATH}"
echo "Install with: flatpak install --user \"${BUNDLE_PATH}\""
echo "Run with: flatpak run ${APP_ID}"

# Final step: remove temporary build state unless KEEP_TEMP=1.
cleanup_temp
