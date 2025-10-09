# Oak Log Gathering Evaluation Setup

This is a specialized fork of JarvisVLA configured to evaluate oak log gathering tasks.

## What's Been Set Up

1. **Configuration File**: `jarvisvla/evaluate/config/mine/mine_oak_log.yaml`
   - Task: Mine oak logs from trees
   - Spawn biome: Forest (where oak trees are common)
   - Starting inventory: Stone axe (for efficient tree chopping)
   - Success criteria: Successfully mine at least 1 oak log block

2. **Evaluation Scripts**:
   - `test_oak_log_single.sh`: Test with 1 worker (verbose mode)
   - `run_oak_log_100.sh`: Run 100 parallel evaluations

## Prerequisites

1. **VLLM Server Running**: Make sure your VLLM server is running and accessible
2. **VirtualGL Installed**: For GPU rendering (already confirmed as available)
3. **Dependencies Installed**: Install JarvisVLA dependencies

## Quick Start

### Step 1: Install Dependencies

```bash
cd /home/minjune/JarvisVLA-oak
conda activate oak_env  # Make sure you're in the oak_env environment

# Run the installation script
./install_env.sh

# OR manually install:
pip install -e .
```

### Step 2: Configure Your VLLM Server

Edit both shell scripts and update these variables:
- `base_url`: Your VLLM server URL (default: "http://localhost:3000/v1")
- `model_local_path`: Your model checkpoint name

```bash
# Edit test_oak_log_single.sh
nano test_oak_log_single.sh

# Edit run_oak_log_100.sh
nano run_oak_log_100.sh
```

### Step 3: Test with Single Evaluation

Run a single evaluation first to ensure everything works:

```bash
# Using VirtualGL for GPU rendering
MINESTUDIO_GPU_RENDER=1 ./test_oak_log_single.sh

# OR using Xvfb (CPU rendering)
./test_oak_log_single.sh
```

### Step 4: Run 100 Evaluations

Once the single test works, run the full evaluation:

```bash
# Using VirtualGL for GPU rendering (recommended)
MINESTUDIO_GPU_RENDER=1 ./run_oak_log_100.sh

# OR using Xvfb (CPU rendering)
./run_oak_log_100.sh
```

## Results

Results will be saved in `logs/<model_name>-mine_oak_log/`:

- **end.json**: Contains success/failure data for each trial
  - Format: `[[success_bool, frames_taken, trial_id], ...]`
- **image.png**: Visualization of success rate
- **Videos**: Individual trial videos in subdirectories (0/, 1/, 2/, ...)

## Configuration Details

### Task Configuration (mine_oak_log.yaml)

- **Biome**: Forest (oak trees spawn naturally)
- **Max frames**: 500 steps per evaluation
- **Starting item**: Stone axe in slot 0
- **Reward**: 1.0 points when any oak_log block is mined
- **Success**: Reward > 0 (at least 1 oak log gathered)

### Evaluation Parameters

- **Temperature**: 0.6 (controls model randomness)
- **History**: 2 frames of history context
- **Action chunk**: 1 action at a time
- **Instruction type**: "normal"

## Modifying the Configuration

To adjust the task, edit `jarvisvla/evaluate/config/mine/mine_oak_log.yaml`:

```yaml
# Example: Give better tools
init_inventory:
  - slot: 0
    type: "iron_axe"  # Faster than stone_axe
    quantity: 1

# Example: Allow multiple oak logs
reward_conf:
  - event: "mine_block"
    max_reward_times: 5  # Count up to 5 oak logs
    reward: 1.0
    objects:
      - "oak_log"
```

## Troubleshooting

1. **Import errors**: Make sure you've installed dependencies with `pip install -e .`
2. **Display errors**: Use `MINESTUDIO_GPU_RENDER=1` if VirtualGL is available
3. **VLLM connection**: Verify your `base_url` is correct and server is running
4. **Ray errors**: The script automatically retries up to 20 times with 10s delays

## Manual Evaluation Command

If you prefer to run manually without the scripts:

```bash
python jarvisvla/evaluate/evaluate.py \
    --workers 100 \
    --env-config mine/mine_oak_log \
    --max-frames 500 \
    --temperature 0.6 \
    --checkpoints <your_model_path> \
    --video-main-fold "logs/" \
    --base-url "http://localhost:3000/v1" \
    --history-num 2 \
    --action-chunk-len 1 \
    --split-number 5
```

## Success Rate Calculation

The evaluation will automatically calculate and display:
- Total number of trials
- Number of successful trials
- Success rate percentage
- Average frames to completion (for successful trials)

Check `logs/<model_name>-mine_oak_log/image.png` for a visual summary.
