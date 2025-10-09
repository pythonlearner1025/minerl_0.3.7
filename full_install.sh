#!/bin/bash
# Complete MineRL 0.3.7 installation in one command
set -e

echo "========================================================================"
echo "MineRL 0.3.7 Full Installation"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Install minerl==0.3.7 with compatible dependencies"
echo "  2. Apply Python 3.10 compatibility patches"
echo "  3. Apply UUID fix to MalmoEnvServer.java"
echo "  4. Configure and build MixinGradle"
echo "  5. Rebuild Malmo Minecraft"
echo ""
echo "Time: ~5 minutes"
echo ""

cd "$(dirname "$0")"

# Step 1: Quick install (Python patches + dependencies)
echo "========================================================================"
echo "STEP 1: Python Patches"
echo "========================================================================"
./quick_install.sh

# Step 2: Apply Malmo patches and rebuild
echo ""
echo "========================================================================"
echo "STEP 2: Malmo Patches & Rebuild"
echo "========================================================================"
./patch.sh

echo ""
echo "========================================================================"
echo "✓✓✓ INSTALLATION COMPLETE ✓✓✓"
echo "========================================================================"
echo ""
echo "All fixes applied:"
echo "  ✓ Python 3.10 compatibility (collections.abc.Mapping)"
echo "  ✓ Gym compatibility (shape assignment removed)"
echo "  ✓ NumPy 2.x compatibility (np.int → int)"
echo "  ✓ JarvisVLA resolution (640x360 native capture)"
echo "  ✓ JarvisVLA FOV (70° Normal, not 105° fish-eye)"
echo "  ✓ MixinGradle configured"
echo "  ✓ UUID fix added to MalmoEnvServer.java"
echo "  ✓ Malmo Minecraft rebuilt"
echo ""
echo "Dependencies pinned:"
echo "  - minerl==0.3.7"
echo "  - gym==0.23.1"
echo "  - numpy==1.23.5"
echo ""
echo "Ready to use!"
echo ""
echo "Start server:"
echo "  cd /home/minjune/minerl_0.3.7"
echo "  source agent_env/bin/activate"
echo "  python test_minerl_server.py"
echo ""
echo "Connect client (in another terminal):"
echo "  source agent_env/bin/activate"
echo "  python -m minerl.interactor 6666"
echo ""
