# OpenGS-MapTool Flatpak Integration: Complete Strategy

## Part 1: How the Current Build Workflow Works

### 1.1 Version Management (Single Source of Truth)

Your release process is architected around **one authoritative version**:

```
┌─────────────────────────────────┐
│ opengs_maptool/config.py        │
│ VERSION = "0.4"                 │ ← Source of truth
└────────────┬────────────────────┘
             │
             ├─→ pyproject.toml must match
             │   (version = "0.4")
             │
             ├─→ GitHub tag check
             │   (v0.4 must NOT exist)
             │
             └─→ Release trigger
                 (builds only if new version)
```

**The check-version job does:**
1. Extract VERSION from `config.py`
2. Extract version from `pyproject.toml`
3. Error if they don't match (prevents accidental version skew)
4. Output `version=0.4` and `tag=v0.4`
5. Check if tag exists on GitHub
6. Set `release-needed=true/false`

This means: **Change VERSION in one place, everything cascades**. Perfect design for single-source-of-truth.

### 1.2 The Build Matrix

Three parallel platform builds (no dependencies between them):

| Platform | Runner | PyInstaller | Output | Notes |
|----------|--------|-------------|--------|-------|
| **Windows** | windows-latest | With .ico icon | `OpenGS-MapTool.exe` | Windowed GUI |
| **Linux** | ubuntu-22.04 | No icon equivalent | `OpenGS-MapTool` | Portable binary |
| **macOS** | macos-14 (Apple Silicon) | With .icns icon | `OpenGS-MapTool.app` bundle | Signed/bundled |

**Key insight:** Each platform gets a **standalone executable** with Python 3.13 + all deps + Qt runtime bundled inside. No system Python needed.

### 1.3 Linux Build Step-by-Step

```
1. Checkout repository
   └─ Ubuntu 22.04 runner

2. Setup Python 3.13 (cached by actions/setup-python)
   └─ Uses pip cache for fast dependency installation

3. Install Qt system libraries (NOT in pip)
   ├─ libegl1, libgl1, libdbus-1-3, libxkbcommon-x11-0
   ├─ libxcb-* (10+ X11 protocol libraries)
   └─ Required: PyInstaller + PyQt6 need these to link at build time

4. pip install . (installs app + all dependencies)
   ├─ Installs pyproject.toml dependencies:
   │  ├─ numpy, pillow, PyQt6, scipy
   │  ├─ qtawesome, PyYAML, platformdirs, etc.
   │  └─ pytest (for testing)
   └─ App code is now installed as editable package

5. pytest -q (run tests)
   ├─ QT_QPA_PLATFORM=offscreen (headless Qt, no display needed)
   └─ Fails build if any test fails

6. PyInstaller builds executable
   ├─ Input: opengs_maptool/main.py
   ├─ Command flags:
   │  ├─ --onefile (single executable, not directory)
   │  ├─ --windowed (GUI app, no console)
   │  ├─ --name "OpenGS-MapTool"
   │  ├─ --collect-all qtawesome (include icon fonts)
   │  └─ --exclude-module tkinter,pytest,... (shrink size)
   ├─ Process:
   │  ├─ Analyzes imports recursively
   │  ├─ Bundles Python runtime
   │  ├─ Bundles all installed packages
   │  ├─ Bundles PyQt6 + Qt libraries (+ platform-specific libs)
   │  └─ Creates UPX-compressed executable
   └─ Output: dist/OpenGS-MapTool (~150-200 MB)

7. Stage release files
   ├─ Create staging/OpenGS-MapTool/ directory
   ├─ Copy: dist/OpenGS-MapTool → staging/OpenGS-MapTool/
   ├─ Copy: examples/input/ → staging/OpenGS-MapTool/examples/
   ├─ Copy: README.md, LICENSE → staging/OpenGS-MapTool/
   └─ Result:
      staging/OpenGS-MapTool/
      ├── OpenGS-MapTool (executable)
      ├── examples/
      │   └── input/
      │       ├── density.png
      │       ├── bound.png
      │       ├── terrain.png
      │       └── land.png
      ├── README.md
      └── LICENSE

8. Package as tar.gz
   └─ tar -czf OpenGS-MapTool-linux-x86_64.tar.gz -C staging OpenGS-MapTool

9. Upload artifact
   └─ Workflow artifact storage (used by release job later)
```

### 1.4 The Release Job (Final Step)

```
After all 3 platforms complete:

1. Download all artifacts from matrix builds
   ├─ build-windows-latest → OpenGS-MapTool-windows-x86_64.zip
   ├─ build-ubuntu-22.04 → OpenGS-MapTool-linux-x86_64.tar.gz
   └─ build-macos-14 → OpenGS-MapTool-macos-arm64.zip

2. gh release create "$TAG" --generate-notes assets/*
   ├─ Creates GitHub Release with tag v0.4
   ├─ Generates release notes from commits since last tag
   ├─ Uploads all 3 asset files
   └─ Publishes automatically (visible on GitHub Releases page)
```

---

## Part 2: Flatpak Integration Strategy

### 2.1 Why Reuse PyInstaller

Your plan to wrap the PyInstaller executable in Flatpak is **the right approach** because:

| Aspect | Traditional | Your Strategy |
|--------|-------------|---------------|
| **Python in Flatpak** | Rebuilt via Flatpak SDK | Already compiled in executable ✓ |
| **Dependencies** | Resolved inside Flatpak | Already bundled in executable ✓ |
| **Qt libraries** | Installed via flatpak SDK | Already linked in executable ✓ |
| **Build time** | 10+ minutes | ~2-5 minutes ✓ |
| **Manifest complexity** | 50+ lines (Python tooling) | ~20 lines (just wrapper) ✓ |
| **Single source of truth** | Version hardcoded in 2 places | Injected at build time ✓ |

### 2.2 Flatpak Architecture

```
Your Application Stack
═════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────┐
│ Flatpak Container (org.opengs.MapTool)                   │
│                                                          │
│ ┌────────────────────────────────────────────────────┐   │
│ │ /app/                                              │   │
│ │ ├── bin/                                           │   │
│ │ │   └── OpenGS-MapTool ← PyInstaller binary       │   │
│ │ │       (Contains: Python 3.13 + all deps + Qt)   │   │
│ │ │                                                  │   │
│ │ ├── share/                                         │   │
│ │ │   ├── applications/                             │   │
│ │ │   │   └── org.opengs.MapTool.desktop            │   │
│ │ │   │       (Desktop launcher entry)              │   │
│ │ │   │                                              │   │
│ │ │   ├── icons/hicolor/                            │   │
│ │ │   │   ├── 16x16/apps/org.opengs.MapTool.png    │   │
│ │ │   │   ├── 32x32/apps/org.opengs.MapTool.png    │   │
│ │ │   │   ├── 64x64/apps/org.opengs.MapTool.png    │   │
│ │ │   │   ├── 128x128/apps/org.opengs.MapTool.png  │   │
│ │ │   │   └── 256x256/apps/org.opengs.MapTool.png  │   │
│ │ │   │                                              │   │
│ │ │   └── opengs-maptool/                           │   │
│ │ │       ├── VERSION                               │   │
│ │ │       └── examples/input/                       │   │
│ │ │           ├── density.png                       │   │
│ │ │           ├── bound.png                         │   │
│ │ │           ├── terrain.png                       │   │
│ │ │           └── land.png                          │   │
│ │ │                                                  │   │
│ │ └── (other /usr stuff from runtime)               │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ Sandbox Permissions (for X11/Wayland/home access)       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**What Flatpak provides (Freedesktop 24.08 runtime):**
- /usr/lib, /usr/include (standard Linux libraries)
- /usr/bin/env (standard tools)
- Nothing app-specific

**What the PyInstaller executable brings:**
- Python 3.13 runtime (embedded)
- All Python packages (embedded)
- Qt 6.10 libraries (embedded)
- Platform dependencies it linked against

**Result:** The .flatpak file is **just a container** around something that already works.

### 2.3 Build Integration in CI

**Current Linux build produces:**
```
OpenGS-MapTool-linux-x86_64.tar.gz
```

**Modified Linux build will produce:**
```
OpenGS-MapTool-linux-x86_64.tar.gz
OpenGS-MapTool-0.4.flatpak
```

**Additional steps after PyInstaller:**

```python
# Pseudocode for CI additions

# Step 1: Extract version (same logic as check-version job)
VERSION = extract_from_config("opengs_maptool/config.py", r'VERSION\s*=\s*"([^"]+)"')

# Step 2: Prepare Flatpak build context
mkdir -p flatpak-build/{app,share/applications,share/icons,share/opengs-maptool}
cp dist/OpenGS-MapTool flatpak-build/app/bin/
cp -r examples/input flatpak-build/app/share/opengs-maptool/
cp flatpak/org.opengs.MapTool.desktop flatpak-build/share/applications/
cp -r flatpak/icons/* flatpak-build/share/icons/
echo VERSION > flatpak-build/app/share/opengs-maptool/VERSION

# Step 3: Build Flatpak
# Generate manifest with VERSION injected
generate_manifest("flatpak/org.opengs.MapTool.json", VERSION=VERSION)

# Run Flatpak builder
flatpak-builder \
  --repo=flatpak-repo \
  build-dir \
  flatpak/org.opengs.MapTool.json

# Step 4: Create .flatpak bundle
flatpak build-bundle \
  flatpak-repo \
  "OpenGS-MapTool-${VERSION}.flatpak" \
  org.opengs.MapTool

# Step 5: Verify and upload
test -f "OpenGS-MapTool-${VERSION}.flatpak" || exit 1
echo "Created OpenGS-MapTool-${VERSION}.flatpak"
```

---

## Part 3: What Needs to Be Defined Before Implementation

### 3.1 Mandatory: Application ID

**Decision:** `org.opengs.MapTool`

**Why this ID:**
- Reverse DNS convention (Java-style, Linux standard)
- Guarantees uniqueness across all Flatpak apps
- Used in all Flatpak metadata
- Example: GNOME uses `org.gnome.Files`, KDE uses `org.kde.Konsole`

**Used in:**
- Flatpak manifest: `"app-id": "org.opengs.MapTool"`
- Desktop entry: `Icon=org.opengs.MapTool`
- Icon paths: `/app/share/icons/.../org.opengs.MapTool.png`

### 3.2 Mandatory: Desktop Entry

**File to create:** `flatpak/org.opengs.MapTool.desktop`

```ini
[Desktop Entry]
Type=Application
Name=OpenGS MapTool
Comment=Create province maps and related files
Exec=/app/bin/OpenGS-MapTool
Icon=org.opengs.MapTool
Categories=Graphics;Utility;
Terminal=false
```

**Why:**
- Enables app to appear in app menus (GNOME Activities, KDE Kickoff, etc.)
- Tells desktop environment how to launch the app
- Standard Linux desktop convention (XDG Base Directory spec)

**Install location:** `/app/share/applications/org.opengs.MapTool.desktop`

### 3.3 Mandatory: Application Icons (PNG)

**Current state:**
- `ogs-mt-icon-master.ico` (Windows)
- `ogs-mt-icon-master.icns` (macOS)
- No PNG available

**Need to create:**
```
flatpak/icons/
├── 16x16/apps/org.opengs.MapTool.png
├── 32x32/apps/org.opengs.MapTool.png
├── 64x64/apps/org.opengs.MapTool.png
├── 128x128/apps/org.opengs.MapTool.png
└── 256x256/apps/org.opengs.MapTool.png
```

**How to extract from .ico:**
```bash
# Using ImageMagick
convert ogs-mt-icon-master.ico icon-%d.png

# Then manually:
# - Identify which sizes are available
# - Create folders for each size
# - Rename and place correctly

# Or using Python + Pillow:
from PIL import Image
import os

ico = Image.open("ogs-mt-icon-master.ico")
sizes = [16, 32, 64, 128, 256]

for size in sizes:
    # Try to get that size, or resize closest
    resized = ico.resize((size, size), Image.Resampling.LANCZOS)
    os.makedirs(f"flatpak/icons/{size}x{size}/apps", exist_ok=True)
    resized.save(f"flatpak/icons/{size}x{size}/apps/org.opengs.MapTool.png")
```

### 3.4 Mandatory: Flatpak Manifest

**File to create:** `flatpak/org.opengs.MapTool.json`

**Minimal template:**
```json
{
  "app-id": "org.opengs.MapTool",
  "runtime": "org.freedesktop.Platform",
  "runtime-version": "24.08",
  "sdk": "org.freedesktop.Sdk",
  "command": "/app/bin/OpenGS-MapTool",
  "finish-args": [
    "--share=ipc",
    "--socket=fallback-x11",
    "--socket=wayland",
    "--filesystem=home"
  ],
  "modules": [
    {
      "name": "opengs-maptool",
      "buildsystem": "simple",
      "build-commands": [
        "mkdir -p /app/bin /app/share/opengs-maptool/examples",
        "cp OpenGS-MapTool /app/bin/",
        "cp -r examples/input /app/share/opengs-maptool/examples/",
        "mkdir -p /app/share/applications /app/share/icons/hicolor",
        "cp org.opengs.MapTool.desktop /app/share/applications/",
        "cp -r icons/* /app/share/icons/hicolor/"
      ],
      "sources": [
        {
          "type": "dir",
          "path": "."
        }
      ]
    }
  ]
}
```

**What this does:**
- Specifies app ID, runtime (Freedesktop 24.08 has Qt 6.x and standard libs)
- Declares sandbox permissions (IPC, X11/Wayland, home directory access)
- Single module: copies PyInstaller executable, examples, desktop entry, and icons

**Permissions explained:**
| Permission | Why |
|------------|-----|
| `--share=ipc` | Qt needs shared memory for Wayland/X11 |
| `--socket=fallback-x11` | X11 support (for compatibility) |
| `--socket=wayland` | Wayland support (modern, preferred) |
| `--filesystem=home` | Allow access to user's home (for .gsmap files) |

### 3.5 Example Files in Flatpak

**Current situation:**
- App code doesn't hardcode path to examples/
- README says "download includes examples" (distribution expectation)
- Users unzip and have examples/ in same folder

**In Flatpak:**
- Place in: `/app/share/opengs-maptool/examples/input/`
- **App doesn't need to find them programmatically**
- Users can access via file dialog to `/app/share/opengs-maptool/`
- Or document: "Examples are in the installation directory"

**For future expansion (if app wants to find them):**
- Use `platformdirs` (already in dependencies!) to check:
  - `/app/share/opengs-maptool/` (Flatpak)
  - `~/.local/share/opengs-maptool/` (user data)
  - `./examples/` (relative, tar.gz)

### 3.6 Version Propagation Strategy

**Requirement:** Flatpak .flatpak filename MUST include version:
```
OpenGS-MapTool-0.4.flatpak  ← version "0.4" from config.py
```

**Approach: Template manifest + injection**

1. **Create template:** `flatpak/org.opengs.MapTool.json.template`
   - Placeholder: `{VERSION}` or parameterized

2. **At build time (in CI):**
   ```bash
   VERSION=$(python3 -c "import sys; sys.path.insert(0, '.'); from opengs_maptool.config import VERSION; print(VERSION)")
   
   # Generate manifest from template
   sed "s/{VERSION}/${VERSION}/g" flatpak/org.opengs.MapTool.json.template > flatpak/org.opengs.MapTool.json
   
   # Build Flatpak with injected version
   flatpak-builder ... flatpak/org.opengs.MapTool.json
   
   # Name output with version
   flatpak build-bundle ... "OpenGS-MapTool-${VERSION}.flatpak" org.opengs.MapTool
   ```

3. **Result:**
   - Flatpak filename matches VERSION from config.py
   - Single source of truth preserved
   - No manual version updates needed

---

## Part 4: Proposed Workflow Modifications

### 4.1 Changes to release.yml

**Current Linux build (lines ~90-125):**
```yaml
- name: Build (Linux)
  if: runner.os == 'Linux'
  run: >
    pyinstaller --noconfirm --clean --onefile --windowed
    --name "${{ env.APP_NAME }}"
    --collect-all qtawesome
    --exclude-module tkinter --exclude-module tkinterdnd2 --exclude-module pytest
    opengs_maptool/main.py

- name: Stage release files
  shell: bash
  run: |
    # ... tar.gz packaging ...

- name: Package (Linux)
  if: runner.os == 'Linux'
  run: tar -czf "${{ matrix.asset }}" -C staging "${{ env.APP_NAME }}"
```

**Add after Package (Linux) step:**

```yaml
- name: Extract version for Flatpak
  if: runner.os == 'Linux'
  id: version
  run: |
    VERSION=$(python3 -c "import sys; sys.path.insert(0, '.'); from opengs_maptool.config import VERSION; print(VERSION)")
    echo "version=${VERSION}" >> "$GITHUB_OUTPUT"

- name: Install Flatpak tools
  if: runner.os == 'Linux'
  run: |
    sudo apt-get update
    sudo apt-get install -y flatpak-builder

- name: Prepare Flatpak build
  if: runner.os == 'Linux'
  run: |
    VERSION="${{ steps.version.outputs.version }}"
    
    # Create Flatpak build directory structure
    mkdir -p flatpak-build/app/bin flatpak-build/app/share/applications flatpak-build/app/share/icons flatpak-build/app/share/opengs-maptool/examples
    
    # Copy PyInstaller executable
    cp dist/OpenGS-MapTool flatpak-build/app/bin/
    chmod +x flatpak-build/app/bin/OpenGS-MapTool
    
    # Copy examples
    cp -r examples/input flatpak-build/app/share/opengs-maptool/examples/
    
    # Copy desktop entry and icons
    cp flatpak/org.opengs.MapTool.desktop flatpak-build/app/share/applications/
    cp -r flatpak/icons/* flatpak-build/app/share/icons/
    
    # Store version
    echo "${VERSION}" > flatpak-build/app/share/opengs-maptool/VERSION

- name: Build Flatpak
  if: runner.os == 'Linux'
  run: |
    VERSION="${{ steps.version.outputs.version }}"
    
    # Build Flatpak
    flatpak-builder \
      --repo=flatpak-repo \
      --force-clean \
      build-dir \
      flatpak/org.opengs.MapTool.json
    
    # Create .flatpak bundle
    flatpak build-bundle \
      flatpak-repo \
      "OpenGS-MapTool-${VERSION}.flatpak" \
      org.opengs.MapTool
    
    # Verify
    test -f "OpenGS-MapTool-${VERSION}.flatpak" || { echo "::error::Flatpak build failed"; exit 1; }
    echo "Built OpenGS-MapTool-${VERSION}.flatpak ($(du -h "OpenGS-MapTool-${VERSION}.flatpak" | cut -f1))"

- name: Check all Linux assets
  if: runner.os == 'Linux'
  shell: bash
  run: |
    VERSION="${{ steps.version.outputs.version }}"
    test -f "OpenGS-MapTool-linux-x86_64.tar.gz" || { echo "::error::tar.gz not found"; exit 1; }
    test -f "OpenGS-MapTool-${VERSION}.flatpak" || { echo "::error::flatpak not found"; exit 1; }
    ls -lh OpenGS-MapTool-*
```

### 4.2 Matrix Asset Update

**Current (lines ~40-52):**
```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - os: windows-latest
        label: Windows
        asset: OpenGS-MapTool-windows-x86_64.zip
      - os: ubuntu-22.04
        label: Linux
        asset: OpenGS-MapTool-linux-x86_64.tar.gz  # ← ONLY tar.gz
      - os: macos-14
        label: macOS (Apple Silicon)
        asset: OpenGS-MapTool-macos-arm64.zip
```

**Problem:** Matrix with single asset per OS breaks when Linux produces 2 assets

**Solution 1: Upload both as "Linux" artifact**
```yaml
- uses: actions/upload-artifact@v4
  if: runner.os == 'Linux'
  with:
    name: build-${{ matrix.os }}
    path: |
      OpenGS-MapTool-linux-x86_64.tar.gz
      OpenGS-MapTool-*.flatpak
    if-no-files-found: error
```

**Solution 2: Separate upload for Flatpak (cleaner)**
```yaml
# After tar.gz upload:
- uses: actions/upload-artifact@v4
  if: runner.os == 'Linux'
  with:
    name: build-linux-flatpak
    path: OpenGS-MapTool-*.flatpak
    if-no-files-found: error
```

---

## Part 5: Implementation Checklist

### Phase 1: Preparation (Before modifying CI)

- [ ] **Define Application ID**
  - Decision: `org.opengs.MapTool` ✓

- [ ] **Create Desktop Entry**
  - File: `flatpak/org.opengs.MapTool.desktop`
  - Based on template above

- [ ] **Extract/Create Icons**
  - Extract PNG from .ico or recreate
  - Create directory structure: `flatpak/icons/{size}x{size}/apps/`
  - 5 sizes: 16, 32, 64, 128, 256

- [ ] **Create Flatpak Manifest**
  - File: `flatpak/org.opengs.MapTool.json`
  - Based on template above
  - Test locally with `flatpak-builder`

- [ ] **Test Flatpak locally**
  - Install flatpak-builder on dev machine
  - Build: `flatpak-builder --repo=repo build-dir flatpak/org.opengs.MapTool.json`
  - Create bundle: `flatpak build-bundle repo OpenGS-MapTool-0.4.flatpak org.opengs.MapTool`
  - Test run: `flatpak install OpenGS-MapTool-0.4.flatpak && flatpak run org.opengs.MapTool`

### Phase 2: CI Integration

- [ ] **Modify release.yml**
  - Add version extraction step
  - Add flatpak-builder install
  - Add build Flatpak steps
  - Add verification

- [ ] **Test on main branch**
  - Trigger workflow manually (without VERSION change)
  - Verify tar.gz still builds
  - Verify Flatpak builds without releasing
  - Check artifact output

- [ ] **Create release with new VERSION**
  - Update config.py and pyproject.toml to v0.5 (or next version)
  - Push to main
  - Watch GitHub Actions
  - Verify both assets appear in release

### Phase 3: Polish & Documentation

- [ ] **Create AppData file** (optional, for Flathub listing)
  - File: `flatpak/org.opengs.MapTool.appdata.xml`
  - Contains: description, screenshots, release notes

- [ ] **Document Flatpak support**
  - Update README.md installation section
  - Add: "On Linux, install from Flathub (coming soon)"
  - Explain how to install .flatpak manually: `flatpak install OpenGS-MapTool-0.4.flatpak`

- [ ] **Consider Flathub submission** (future)
  - Submit manifest + appdata + screenshots
  - Enables `flatpak install --from-file` support
  - Reduces download friction

---

## Quick Reference: Files to Create/Modify

### New Files
```
flatpak/
├── org.opengs.MapTool.json        (manifest)
├── org.opengs.MapTool.desktop     (desktop entry)
└── icons/
    ├── 16x16/apps/org.opengs.MapTool.png
    ├── 32x32/apps/org.opengs.MapTool.png
    ├── 64x64/apps/org.opengs.MapTool.png
    ├── 128x128/apps/org.opengs.MapTool.png
    └── 256x256/apps/org.opengs.MapTool.png
```

### Modified Files
```
.github/workflows/release.yml
├── Add version extraction step
├── Add flatpak-builder install
├── Add Flatpak build steps
└── Update artifact upload logic

README.md
├── Add Linux/Flatpak installation instructions
└── Update "How to install" section
```

---

## Next Steps

**Immediate:**
1. ✅ Understand current workflow (DONE)
2. **Extract icons from .ico** (see Python script above)
3. **Create `flatpak/` directory structure**
4. **Write desktop entry and manifest**
5. **Test locally** before committing CI changes

**Then:**
6. **Modify release.yml** with Flatpak build steps
7. **Test on GitHub** with manual workflow dispatch
8. **Create new release** to verify both assets build

**Questions to clarify:**
- Do you want to start with just creating the static files first?
- Should I generate the icon sizes for you?
- Prefer to test Flatpak locally before CI changes?
