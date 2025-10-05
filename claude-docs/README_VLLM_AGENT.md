# JarvisVLA Agent for minerl 0.3.7

Port of JarvisVLA's VLLM agent to work with minerl 0.3.7 for human-interactive Minecraft gameplay.

## Features

- ✅ Full JarvisVLA action/observation pipeline
- ✅ OpenAI API compatibility (connects to VLLM server)
- ✅ Conversation history tracking
- ✅ Mu-law camera quantization/dequantization
- ✅ Recipe-based and normal instruction modes
- ✅ Realtime observation sampling
- ✅ Action chunking support
- ✅ **No transformers dependency** - uses regex to parse action tokens
- ✅ **Minimal dependencies** - works with Python 3.6+ and old minerl

## Installation

```bash
# Minimal dependencies (no transformers!)
pip install -r requirements_vllm_agent.txt

# Or manually:
pip install numpy Pillow openai
```

## Quick Start

### 1. Start VLLM Server

```bash
# Start VLLM with your trained model
vllm serve /path/to/jarvisvla/model --port 8000 --trust-remote-code
```

### 2. Run Agent

```bash
cd /home/minjune/minerl_0.3.7

# Basic usage
python example_realtime_agent.py \\
    --base-url http://localhost:8000/v1 \\
    --task "craft item crafting_table" \\
    --max-steps 200

# With conversation history
python example_realtime_agent.py \\
    --base-url http://localhost:8000/v1 \\
    --task "craft item crafting_table" \\
    --history-num 3 \\
    --instruction-type recipe \\
    --temperature 0.5 \\
    --verbose
```

## Architecture

### VLLMAgentWrapper

Main agent class that handles:

1. **Prompt Construction** - Loads task instructions and recipes from `assets/`
2. **Image Preprocessing** - Resizes and encodes observations as base64
3. **VLLM Communication** - Sends multimodal messages via OpenAI API
4. **Action Decoding** - Converts VLM output tokens to minerl actions
5. **History Management** - Tracks conversation context

### Action Mapping Pipeline

```
VLM Output Tokens
    ↓
Token Parsing (find start/end tags)
    ↓
Group Actions (12 integers: hotbar, movement, camera)
    ↓
Mu-Law Inverse (camera bins → continuous angles)
    ↓
MineRL Action Dict
```

### Token Extraction (No Transformers!)

Instead of using the transformers library, we use simple regex to extract action tokens:

```python
# VLLM outputs text like: "<|reserved_special_token_178|><|reserved_special_token_180|>..."
pattern = r'<\|reserved_special_token_(\d+)\|>'
matches = re.findall(pattern, outputs)

# Convert to token IDs: 151657 + token_num
# e.g., token_178 → 151657 + 178 = 151835
token_ids = [151657 + int(m) for m in matches]
```

This works because:
- Action tokens are in range 151833-151907 (Qwen2-VL reserved tokens)
- VLLM returns them as text with `skip_special_tokens=False`
- We only care about special tokens, ignore all other text

### Key Parameters

- **camera_mu**: 20 (mu-law compression parameter)
- **camera_bins**: 21 (quantization bins per axis)
- **camera_maxval**: 10 (degrees)
- **bases**: [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21] (action group sizes)

## File Structure

```
minerl_0.3.7/
├── vllm_agent_wrapper.py      # Main agent class
├── example_realtime_agent.py  # Realtime execution loop
├── server.py                  # MineRL server (for GUI mode)
├── assets/
│   ├── instructions.json      # Task instructions
│   └── recipes/               # Minecraft recipes (858 files)
└── README_VLLM_AGENT.md       # This file
```

## Usage Examples

### Test Agent Without Environment

```python
from vllm_agent_wrapper import VLLMAgentWrapper
import numpy as np

agent = VLLMAgentWrapper(
    base_url="http://localhost:8000/v1",
    instruction_type='normal',
    temperature=0.7
)

# Create dummy observation
obs = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

# Get action
action = agent.forward(
    observation=obs,
    task_instruction="craft item crafting_table",
    verbose=True
)

print(f"Camera: {action['camera']}")
print(f"Forward: {action['forward']}")
```

### Custom Instruction Type

```python
# Recipe mode: Includes crafting recipes in prompt
agent = VLLMAgentWrapper(
    base_url="http://localhost:8000/v1",
    instruction_type='recipe',  # Include recipes
    history_num=3               # Track 3 previous frames
)

# Simple mode: Just the task thought
agent = VLLMAgentWrapper(
    base_url="http://localhost:8000/v1",
    instruction_type='simple'  # Minimal prompt
)
```

### With MineRL Interactive Mode

```python
import gym
import minerl

env = gym.make('MineRLTreechop-v0')
env.make_interactive(port=6666, realtime=True)  # Enable GUI

obs = env.reset()

for i in range(200):
    action = agent.forward(obs['pov'], task="chop trees")
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

## Troubleshooting

### Issue: "Could not tokenize output"

**Cause**: Agent can't parse VLLM response tokens

**Solution**:
- Make sure VLLM server is running with `--trust-remote-code`
- Check that model supports action tokens (151833-151907 for Qwen2-VL)
- Verify `skip_special_tokens=False` in API call

### Issue: Agent moves erratically

**Cause**: Incorrect camera dequantization

**Solution**:
- Verify mu-law parameters match training config
- Check camera bin mapping (bin 10 should = 0 degrees)
- Test with `verbose=True` to see raw camera values

### Issue: Actions don't match minerl space

**Cause**: Action dict keys mismatch

**Solution**:
- Check your environment's action space: `env.action_space`
- Update `_group_action_to_minerl_action()` if needed
- Some envs may not have all keys (e.g., no hotbar)

## Advanced: Modifying Action Space

If your minerl environment has a different action space:

```python
# In vllm_agent_wrapper.py, modify _group_action_to_minerl_action():

def _group_action_to_minerl_action(self, group_action):
    action = {
        "camera": [...],  # Keep camera mapping
        # Add/remove keys to match your env
        "craft": group_action[0],  # Example: craft action
        # ...
    }
    return action
```

## Performance Notes

- **Inference latency**: ~200-500ms per forward pass (depends on VLLM setup)
- **Sample rate**: Recommended 0.5s interval (2 fps) for realtime mode
- **Memory**: ~100MB for agent wrapper, rest depends on VLLM
- **GPU**: Agent wrapper runs on CPU, VLLM needs GPU

## Credits

- **JarvisVLA**: Original implementation and model training
- **MineStudio**: VPT action mapping and camera quantization
- **minerl 0.3.7**: Legacy Minecraft environment with GUI support

## License

Same as JarvisVLA (check original repository)
