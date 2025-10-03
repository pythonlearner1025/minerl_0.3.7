# Installation Instructions

## Quick Install (Recommended)

```bash
cd /home/minjune/minerl_0.3.7
./quick_install.sh
```

This applies Python patches and configures MixinGradle.

## Then Add UUID Fix

**Option A: Use your already-working environment**
Just copy the working installation:
```bash
# You already have a working minerl_env with all fixes!
# Just use that one instead of rebuilding
```

**Option B: Manually add UUID fix to new install**
The UUID fix is already in `/home/minjune/minerl_env` Malmo Minecraft.
Copy the fixed MalmoEnvServer.java:

```bash
cp /home/minjune/minerl_env/lib/python3.10/site-packages/minerl/env/Malmo/Minecraft/src/main/java/com/microsoft/Malmo/Client/MalmoEnvServer.java \
   /home/minjune/minerl_0.3.7/venv/lib/python3.10/site-packages/minerl/env/Malmo/Minecraft/src/main/java/com/microsoft/Malmo/Client/

# Then rebuild
cd /home/minjune/minerl_0.3.7/venv/lib/python3.10/site-packages/minerl/env/Malmo/Minecraft
./gradlew clean build --no-daemon
```

## What Gets Fixed

1. ✅ **spaces.py** - Line 202 removed (`self.shape = ()`)
2. ✅ **core.py** - `collections.Mapping` → `collections.abc.Mapping`
3. ✅ **build.gradle** - MixinGradle configured with local JAR
4. ✅ **MixinGradle** - Built from source (commit dcfaf61)
5. ✅ **MalmoEnvServer.java** - UUID fix added
6. ✅ **Malmo Minecraft** - Rebuilt with all fixes

## Usage

```bash
cd /home/minjune/minerl_0.3.7
source venv/bin/activate

# Terminal 1: Start server
python test_minerl_server.py

# Terminal 2: Connect client (wait for "MineRL agent is public")
python -m minerl.interactor 6666
```

## Files Summary

```
minerl_0.3.7/
├── venv/                    # Virtual environment with all patches
├── quick_install.sh         # Apply Python + MixinGradle fixes
├── add_uuid_fix.sh         # Add UUID fix (optional - use working env instead)
├── test_minerl_server.py   # Server test script
├── test_minerl_client.py   # Client test script
├── patches/                 # Patch files for reference
├── README.md               # Overview
├── FIXES_APPLIED.md        # Detailed modifications list
└── INSTALL_INSTRUCTIONS.md # This file
```

## Alternative: Use Existing Working Environment

Since you already have a fully working environment at `/home/minjune/minerl_env`, you can:

1. Copy test scripts to a workspace directory
2. Use that venv instead
3. Skip this whole installation

```bash
mkdir -p ~/minerl_workspace
cp /tmp/test_minerl_*.py ~/minerl_workspace/
cd ~/minerl_workspace
source ~/minerl_env/bin/activate
python test_minerl_server.py
```
