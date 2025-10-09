# JarvisVLA Oak Log Gathering - Complete Evaluation Configuration

## Python Environment

**Environment Type**: System Python 3.10 with user-local packages (NOT conda/virtualenv)
- **Python Executable**: `/usr/bin/python3`
- **Python Version**: Python 3.10
- **Package Location**: `~/.local/lib/python3.10/site-packages/`

## Model Configuration

### Model Parameters
```yaml
model_name: "CraftJarvis/JarvisVLA-Qwen2-VL-7B"
checkpoint_path: "CraftJarvis/JarvisVLA-Qwen2-VL-7B"
base_url: "http://localhost:3000/v1"
temperature: 0.6
history_num: 2
instruction_type: "normal"
action_chunk_len: 1
```

**Model Details:**
- **Base Model**: Qwen2-VL-7B (Vision-Language Model)
- **Fine-tuned**: JarvisVLA (CraftJarvis team)
- **Model Size**: 7 billion parameters
- **Served via**: VLLM (at port 3000)

## Environment Configuration

### Minecraft Environment (mine_oak_log.yaml)
```yaml
# Base Configuration
seed: 19961103
random_tp_range: 1000
fast_reset: True
start_time: 0
time_limit: 1000

# Spawn Settings
candidate_preferred_spawn_biome:
  - "forest"
candidate_weather:
  - "clear"

# Camera Configuration
camera_config:
  camera_binsize: 1
  camera_maxval: 10
  camera_mu: 20
  camera_quantization_scheme: "mu_law"

# Resolution
origin_resolution: [640, 360]
resize_resolution: [640, 360]

# GUI and Inventory
need_crafting_table: False
need_gui: False
inventory_distraction_level: "zero"

# Task Configuration
task_conf:
  - name: "mine"
    text: "mine_block:oak_log"

# Reward Configuration
reward_conf:
  - event: "mine_block"
    max_reward_times: 1
    reward: 1.0
    objects:
      - "oak_log"
    identity: "mine oak_log"

# Starting Inventory
init_inventory:
  - slot: 0
    type: "stone_axe"
    quantity: 1

# Worker Type
worker: "mine"

# Mobs
mobs: []
```

### Evaluation Parameters
```yaml
workers: 10  # (or 100 for full evaluation)
max_frames: 500
split_number: 5
video_main_fold: "logs/"
env_config: "mine/mine_oak_log"
verbos: False
device: "cuda:1"  # (not used with VLLM)
```

## Action Space Configuration

### Tokenizer Configuration
- **Tokenizer Type**: qwen2_vl
- **Action Bases**: [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21]
  - Base 10: Forward/backward movement
  - Base 3: Left/right movement, jump, sneak, sprint, use, attack, place
  - Base 21: Camera pitch and yaw (21 bins each)

### Camera Configuration
- **Camera MU**: 20
- **Camera Bins**: 21
- **Camera Bin Size**: 1
- **Quantization Scheme**: mu_law

## Rendering Configuration

### GPU Rendering (VirtualGL)
```bash
MINESTUDIO_GPU_RENDER=1
# Falls back to CPU rendering if CUDA unavailable
```

### Actual Rendering Used
- **Mode**: CPU rendering (Xvfb)
- **Reason**: CUDA import unavailable, gracefully degraded
- **Device Output**: "cpu"

## Success Criteria

**Task**: Mine at least one oak log block from a tree

**Success Detection**:
- Event: `mine_block`
- Object: `oak_log`
- Reward: 1.0 point when triggered
- Max reward times: 1 (stops after first oak log)

**Evaluation Metrics**:
- Success/Failure: Boolean (reward > 0)
- Frames to completion: Integer (number of steps taken)
- Success rate: Percentage across all trials

## Output Files

### Directory Structure
```
logs/CraftJarvis-JarvisVLA-Qwen2-VL-7B-mine_oak_log/
├── 0/
│   └── episode_1.mp4
├── 1/
│   └── episode_1.mp4
├── 2/
│   └── episode_1.mp4
├── ...
├── end.json          # Results summary
└── image.png         # Success rate visualization
```

### Results Format (end.json)
```json
[
  [true, 175, "3"],   // [success, frames_taken, worker_id]
  [true, 251, "0"],
  [true, 274, "2"],
  [true, 316, "1"],
  [false, 500, "5"],  // Failed (timed out)
  ...
]
```

## Installed Dependencies

### Core Packages
- **jarvisvla**: 1.0 (editable install)
- **minestudio**: 1.0.6
- **ray**: 2.49.2
- **hydra-core**: 1.3.2
- **omegaconf**: 2.3.0
- **matplotlib**: 3.10.6

### Model/Inference
- **transformers**: 4.50.3
- **openai**: (for VLLM client)
- **qwen-vl-utils**: 0.0.14
- **tokenizers**: 0.21.0

### Environment
- **gym**: 0.26.2
- **gymnasium**: 1.2.1
- **opencv-python**: 4.8.0.74
- **pillow**: 10.4.0
- **av**: 14.1.0

### Visualization
- **matplotlib**: 3.10.6
- **rich**: (for logging)

## Key Configuration Files Modified

1. **`jarvisvla/evaluate/config/mine/mine_oak_log.yaml`** (CREATED)
   - Task-specific configuration for oak log gathering

2. **`jarvisvla/evaluate/evaluate.py`** (MODIFIED)
   - Line 48-50: Fixed `InitInventoryCallback` parameter from `inventory_distraction_level` to `distraction_level`

3. **`jarvisvla/evaluate/env_helper/craft_agent.py`** (MODIFIED)
   - Line 14: Changed import from `minestudio.models.shell.gui_agent` to `jarvisvla.evaluate.env_helper.gui_agent`

4. **`jarvisvla/evaluate/env_helper/smelt_agent.py`** (MODIFIED)
   - Line 6: Changed import from `minestudio.models.shell.craft_agent` to `jarvisvla.evaluate.env_helper.craft_agent`

5. **GPU Utils Files** (MODIFIED - both locations)
   - `/home/minjune/JarvisVLA-oak/oak_env/lib/python3.10/site-packages/minestudio/simulator/minerl/env/gpu_utils.py`
   - `/home/minjune/.local/lib/python3.10/site-packages/minestudio/simulator/minerl/env/gpu_utils.py`
   - Wrapped CUDA imports in try-except for graceful degradation

## Command Line Invocation

### Test (1 worker, verbose)
```bash
cd /home/minjune/JarvisVLA-oak
export MINESTUDIO_GPU_RENDER=1
python3 jarvisvla/evaluate/evaluate.py \
    --workers 0 \
    --env-config mine/mine_oak_log \
    --max-frames 500 \
    --temperature 0.6 \
    --checkpoints CraftJarvis/JarvisVLA-Qwen2-VL-7B \
    --video-main-fold "logs/" \
    --base-url "http://localhost:3000/v1" \
    --history-num 2 \
    --instruction-type normal \
    --action-chunk-len 1 \
    --verbos True
```

### Full Evaluation (100 workers)
```bash
cd /home/minjune/JarvisVLA-oak
./run_oak_log_100.sh
```

## Performance Metrics

### From 10-Worker Test Run
- **Average FPS**: 28-40 FPS
- **Frame Processing Time**: 0.03-0.04s per frame
- **Success Examples**:
  - Worker 3: 175 frames (~6 seconds)
  - Worker 0: 251 frames (~8 seconds)
  - Worker 2: 274 frames (~9 seconds)
  - Worker 1: 316 frames (~11 seconds)

## Notes

- The system gracefully falls back to CPU rendering when GPU/CUDA is unavailable
- Minecraft warnings (OptiFine, Realms, narrator) are non-critical and can be ignored
- The `can't set up init inventory` warning is expected - inventory setup happens in-game
- Ray handles parallel evaluation across multiple Minecraft instances
- Videos are saved for post-hoc analysis of agent behavior
