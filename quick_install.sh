#!/bin/bash
# Quick install script - run this if the full install.sh hangs
set -e

echo "Quick MineRL 0.3.7 Install"
echo "=========================="

cd "$(dirname "$0")"

# Step 1: Create venv
echo "[1/7] Creating venv..."
python3 -m venv agent_env
source agent_env/bin/activate

# Step 2: Install minerl with compatible versions
echo "[2/7] Installing minerl==0.3.7 with compatible gym/numpy (this takes ~2 min)..."
pip install -q --upgrade pip setuptools wheel
pip install -q gym==0.23.1 numpy==1.23.5
pip install -q minerl==0.3.7

# Step 3: Get path
echo "[3/7] Finding MineRL installation..."
MINERL_PATH=$(python -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'minerl'))")
echo "Found at: $MINERL_PATH"

# Step 4: Fix spaces.py
echo "[4/7] Patching spaces.py..."
sed -i '202d' "$MINERL_PATH/herobraine/hero/spaces.py"
echo "✓ Line 202 (self.shape = ()) removed"

# Step 5: Fix core.py
echo "[5/7] Patching core.py..."
sed -i 's/collections\.Mapping/collections.abc.Mapping/g' "$MINERL_PATH/env/core.py"
echo "✓ collections.Mapping → collections.abc.Mapping"

# Step 5b: Fix np.int references
echo "[5b/7] Fixing np.int references..."
sed -i 's/dtype=np\.int)/dtype=int)/g' "$MINERL_PATH/herobraine/hero/handlers/observables.py"
echo "✓ np.int → int"

# Step 5c: Fix resolution to 640x360 (JarvisVLA training resolution)
echo "[5c/7] Setting observation resolution to 640x360..."
sed -i 's/self\.resolution = tuple((64, 64))/self.resolution = tuple((360, 640))/' "$MINERL_PATH/herobraine/env_specs/simple_env_spec.py"
echo "✓ Resolution: 64x64 → 360x640 (height, width)"

# Step 5d: Fix FOV to 70° (JarvisVLA training FOV)
echo "[5d/7] Setting Minecraft FOV to 70° (Normal)..."
sed -i 's/fov:1\.5/fov:0.0/' "$MINERL_PATH/data/assets/template_minecraft/options.txt"
echo "✓ FOV: 1.5 (105° fish-eye) → 0.0 (70° Normal)"

# Step 5e: Fix XML VideoProducer resolution
echo "[5e/7] Patching mission XML files for 640x360 video..."
find "$MINERL_PATH/herobraine/env_specs/missions" -name "*.xml" -exec sed -i 's/<Width>64<\/Width>/<Width>640<\/Width>/g;s/<Height>64<\/Height>/<Height>360<\/Height>/g' {} \;
echo "✓ Mission XMLs: VideoProducer 64x64 → 640x360"

# Step 6: Build MixinGradle
echo "[6/7] Building MixinGradle..."
if [ ! -d "/tmp/MixinGradle" ]; then
    git clone -q https://github.com/SpongePowered/MixinGradle.git /tmp/MixinGradle
fi
cd /tmp/MixinGradle
git checkout -q dcfaf61
gradle build -q 2>&1 | grep -E "BUILD|FAILED" || true

MALMO_DIR="$MINERL_PATH/env/Malmo/Minecraft"
mkdir -p "$MALMO_DIR/libs"
cp build/libs/mixingradle-0.6-SNAPSHOT.jar "$MALMO_DIR/libs/"
echo "✓ MixinGradle built and copied"

# Step 7: Update build.gradle
echo "[7/7] Updating build.gradle..."
cd "$MALMO_DIR"
cp build.gradle build.gradle.backup

# Simple replacement - add flatDir before jitpack
sed -i "/maven { url 'https:\/\/jitpack.io' }/i\\        flatDir { dirs 'libs' }" build.gradle
sed -i "s/classpath 'com.github.SpongePowered:MixinGradle:dcfaf61'/classpath name: 'mixingradle-0.6-SNAPSHOT'/g" build.gradle

echo "✓ build.gradle updated"

echo ""
echo "========================================="
echo "Core patches applied successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Add UUID fix (see FIXES_APPLIED.md)"
echo "  2. Rebuild Malmo:"
echo "       cd $MALMO_DIR"
echo "       ./gradlew clean build"
echo ""
echo "Or just test if it works:"
echo "  source agent_env/bin/activate"
echo "  python test_minerl_server.py"
echo ""
