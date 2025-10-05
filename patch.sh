#!/bin/bash
# Apply UUID fix, JarvisVLA compatibility patches, and build configuration
#
# JarvisVLA Compatibility:
# - Sets FOV to 0.0 (70° Normal) to match training data (not 1.5 = 105° fish-eye)
# - Sets observation resolution to 640x360 (not 64x64) for native high-res capture
#
set -e

cd "$(dirname "$0")"
source venv/bin/activate

echo "Applying JarvisVLA compatibility patches..."

# Fix FOV to 70° (Normal) - JarvisVLA was trained with normal FOV, not fish-eye
echo "[0/6] Setting Minecraft FOV to 70° (Normal)..."
OPTIONS_FILES=(
    "venv/minerl/data/assets/template_minecraft/options.txt"
    "venv/minerl/env/Malmo/Minecraft/run/options.txt"
    "venv/lib/python3.10/site-packages/minerl/data/assets/template_minecraft/options.txt"
)

for OPTIONS_FILE in "${OPTIONS_FILES[@]}"; do
    if [ -f "$OPTIONS_FILE" ]; then
        sed -i 's/fov:1\.5/fov:0.0/' "$OPTIONS_FILE"
        echo "  ✓ Updated $OPTIONS_FILE (fov: 1.5 → 0.0)"
    else
        echo "  ⚠ $OPTIONS_FILE not found (will be created on first run)"
    fi
done

# Fix resolution to 640x360 - JarvisVLA training resolution
echo "[0/6] Setting observation resolution to 640x360..."
SIMPLE_ENV_SPEC="venv/minerl/herobraine/env_specs/simple_env_spec.py"
if [ -f "$SIMPLE_ENV_SPEC" ]; then
    sed -i 's/self\.resolution = tuple((64, 64))/self.resolution = tuple((640, 360))/' "$SIMPLE_ENV_SPEC"
    echo "  ✓ Updated $SIMPLE_ENV_SPEC (64x64 → 640x360)"
else
    echo "  ✗ $SIMPLE_ENV_SPEC not found!"
    exit 1
fi

echo ""
echo "Applying Malmo patches..."

MALMO_DIR="$(pwd)/venv/lib/python3.10/site-packages/minerl/env/Malmo/Minecraft"

# 1. Copy MalmoEnvServer.java with UUID fix
echo "[1/6] Copying MalmoEnvServer.java with UUID fix..."
if [ ! -f "patches/MalmoEnvServer.java" ]; then
    echo "✗ patches/MalmoEnvServer.java not found!"
    exit 1
fi

cp "patches/MalmoEnvServer.java" "$MALMO_DIR/src/main/java/com/microsoft/Malmo/Client/MalmoEnvServer.java"
echo "✓ MalmoEnvServer.java applied"

# 2. Copy build.gradle
echo "[2/6] Copying build.gradle..."
if [ ! -f "patches/build.gradle" ]; then
    echo "✗ patches/build.gradle not found!"
    exit 1
fi

cp "patches/build.gradle" "$MALMO_DIR/build.gradle"
echo "✓ build.gradle applied"

# 3. Build or copy MixinGradle JAR (build-first philosophy)
echo "[3/6] Setting up MixinGradle..."
mkdir -p "$MALMO_DIR/libs"

BUILD_SUCCESS=false

# Try building from source first
echo "  → Attempting to build MixinGradle from source..."
if [ ! -d "MixinGradle" ]; then
    echo "     Cloning repository..."
    git clone -q https://github.com/SpongePowered/MixinGradle.git 2>&1 | grep -v "^Cloning" || true
fi

cd MixinGradle
git checkout -q dcfaf61 2>/dev/null || true

if gradle build 2>&1 | grep -q "BUILD SUCCESSFUL"; then
    if [ -f "build/libs/mixingradle-0.6-SNAPSHOT.jar" ]; then
        cp build/libs/mixingradle-0.6-SNAPSHOT.jar "$MALMO_DIR/libs/"
        echo "  ✓ MixinGradle built from source"
        BUILD_SUCCESS=true
    fi
else
    echo "     Build failed or gradle not available"
fi

# Fall back to backup JAR if build failed
if [ "$BUILD_SUCCESS" = false ]; then
    echo "  → Falling back to backup JAR from patches/..."
    if [ -f "patches/mixingradle-0.6-SNAPSHOT.jar" ]; then
        cp "patches/mixingradle-0.6-SNAPSHOT.jar" "$MALMO_DIR/libs/"
        echo "  ✓ MixinGradle copied from patches/ (backup)"
    else
        echo "  ✗ Backup JAR not found in patches/"
        exit 1
    fi
fi

# 4. Rebuild Malmo
echo "[4/6] Rebuilding Malmo Minecraft (takes ~2-3 min)..."
cd "$MALMO_DIR"
BUILD_OUTPUT=$(./gradlew clean build --no-daemon 2>&1)

if echo "$BUILD_OUTPUT" | grep -q "BUILD SUCCESSFUL"; then
    JAR_SIZE=$(ls -lh build/libs/MalmoMod-0.37.0.jar | awk '{print $5}')
    echo ""
    echo "========================================================================"
    echo "✓✓✓ Malmo rebuilt successfully (JAR size: $JAR_SIZE) ✓✓✓"
    echo "========================================================================"
    echo ""
    echo "Installation complete!"
    echo ""
    echo "To use:"
    echo "  cd $(dirname "$0")"
    echo "  source venv/bin/activate"
    echo "  python test_minerl_server.py"
    echo ""
else
    echo "✗ Build failed - showing errors:"
    echo "$BUILD_OUTPUT" | grep -E "error:|FAILED" | head -10
    exit 1
fi
