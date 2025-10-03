# MineRL 0.3.7 with Human Interaction Fix

**The 15-hour bug:** One line - `collections.Mapping` → `collections.abc.Mapping`

## Quick Start

```bash
cd /home/minjune/minerl_0.3.7
source venv/bin/activate
python test_minerl_server.py
```

Then in another terminal:
```bash
cd /home/minjune/minerl_0.3.7
source venv/bin/activate
python -m minerl.interactor 6666
```

## Installation Status

✅ **FULLY INSTALLED AND WORKING**

All patches applied, Malmo rebuilt, ready to use.

## Installation

One command:
```bash
./full_install.sh
```

This runs:
1. `quick_install.sh` - Install minerl + apply Python patches
2. `patch.sh` - Apply UUID fix + rebuild Malmo (builds MixinGradle from source, falls back to backup)

## What We Modified

1. **spaces.py** - Removed `self.shape = ()`
2. **core.py** - Fixed `collections.Mapping` → `collections.abc.Mapping`
3. **observables.py** - Fixed `np.int` → `int`
4. **MalmoEnvServer.java** - Added UUID generation (50+ lines)
5. **build.gradle** - Configured for local MixinGradle
6. **MalmoMod JAR** - Rebuilt with all fixes

## Required Versions

```
Python: 3.10
minerl: 0.3.7
gym: 0.23.1  ← NOT 0.26.2!
numpy: 1.23.5 ← NOT 2.x!
```

## Files

```
minerl_0.3.7/
├── venv/                      ← FULLY CONFIGURED
├── COMPLETE_INSTALL.sh        ← One-command install
├── quick_install.sh           ← Python + MixinGradle
├── copy_from_working_env.sh   ← UUID fix + config
├── test_minerl_server.py      ← Start this
├── test_minerl_client.py      ← Test script
├── requirements.txt           ← Pinned versions
├── README.md                  ← This file
├── FIXES_APPLIED.md           ← Detailed changes
└── patches/                   ← Reference patches
```

## The Bug

Python 3.10 removed `collections.Mapping`. MineRL 0.3.7 still used it. This caused:
- Silent exceptions during observation processing
- Looked like socket timeouts
- Infinite retry loops
- Mission would load then immediately crash

**One line fix, 15 hours to find.**

## Lesson

Always check Python/dependency versions first when running old repos. Could have saved 14.5 hours.
