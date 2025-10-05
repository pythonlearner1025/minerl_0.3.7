# Agent Logging System

Comprehensive logging for debugging VLLMAgentWrapper behavior.

## What Gets Logged

1. **User Commands** - Every command typed (task changes, resets)
2. **Agent State** - History length, instruction type, method at each step
3. **Observations** - RGB images saved as PNGs
4. **Prompts** - Full message history sent to VLLM (with base64 images omitted)
5. **VLLM Responses** - Raw text output and extracted token IDs
6. **Actions** - Decoded minerl action dicts
7. **Timing** - Wall clock time for VLLM calls
8. **Errors** - Any parsing or decoding failures

## Directory Structure

```
logs/
└── session_20250104_143022/
    ├── agent_log.jsonl          # Main log (one JSON per line)
    ├── summary.txt               # Human-readable summary
    ├── observations/
    │   ├── step_000000.png
    │   ├── step_000001.png
    │   └── ...
    └── prompts/
        ├── step_000000.json      # Full message history
        ├── step_000001.json
        └── ...
```

## Log Format

`agent_log.jsonl` contains one JSON object per line:

```json
{"timestamp": 1234567890.123, "datetime": "2025-01-04T14:30:22", "type": "user_command", "step": 0, "command": "chop trees"}
{"timestamp": 1234567890.456, "type": "forward_call", "step": 0, "task": "chop trees", "method": "freestyle", "instruction_type": "freestyle", "history_length": 0}
{"timestamp": 1234567891.789, "type": "vllm_response", "step": 0, "response_text": "<|reserved_special_token_178|>...", "num_tokens": 15, "wall_time_ms": 245.3}
{"timestamp": 1234567892.012, "type": "action", "step": 0, "action": {"camera": [0.0, 0.5], "forward": 1, ...}}
```

## Debugging Workflow

### Problem: Agent not responding to new tasks

**Check:**
1. `summary.txt` - Did user commands get logged?
2. Search for `"type": "user_command"` in `agent_log.jsonl`
3. Check if `agent.reset()` cleared history - look at `history_length` in next `forward_call`
4. Check `prompts/step_NNNNNN.json` - does it contain the new task?

Example:
```bash
# Find all user commands
grep '"type": "user_command"' logs/session_*/agent_log.jsonl

# Check if history was cleared after command
grep -A 1 '"type": "user_command"' logs/session_*/agent_log.jsonl | grep history_length
```

### Problem: Agent always does same action

**Check:**
1. `vllm_response` entries - is VLLM returning different tokens?
2. Compare `token_ids` across multiple steps
3. Check `response_text` - does it contain action tokens?

Example:
```bash
# Extract all token_ids
grep '"type": "vllm_response"' logs/session_*/agent_log.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    obj = json.loads(line.split('}{')[0] + '}')
    print(f\"Step {obj['step']}: {obj['token_ids'][:10]}...\")
"
```

### Problem: Wrong prompt being sent

**Check:**
1. Open `prompts/step_NNNNNN.json`
2. Look at `messages` array
3. Verify `first_message_text` contains correct task

Example prompt file:
```json
{
  "step": 5,
  "num_messages": 1,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "chop trees\nobservation: "},
        {"type": "image", "note": "[image data omitted]"}
      ]
    }
  ]
}
```

## Quick Analysis Scripts

### Count events by type
```bash
cd logs/session_YYYYMMDD_HHMMSS
cat agent_log.jsonl | python3 -c "
import json, sys
from collections import Counter
types = Counter(json.loads(line)['type'] for line in sys.stdin)
for t, count in types.most_common():
    print(f'{t}: {count}')
"
```

### Extract all task instructions
```bash
grep '"type": "forward_call"' agent_log.jsonl | \
  python3 -c "import json, sys; [print(f\"Step {json.loads(line)['step']}: {json.loads(line)['task']}\") for line in sys.stdin]"
```

### Check VLLM timing
```bash
grep '"type": "vllm_response"' agent_log.jsonl | \
  python3 -c "import json, sys; times = [json.loads(line)['wall_time_ms'] for line in sys.stdin]; print(f'Avg: {sum(times)/len(times):.1f}ms, Min: {min(times):.1f}ms, Max: {max(times):.1f}ms')"
```

## Integration

Logger is automatically created in `example_realtime_agent.py`:

```python
from agent_logger import AgentLogger

logger = AgentLogger(log_dir="logs")
agent = VLLMAgentWrapper(..., logger=logger)

# Logger automatically tracks all agent.forward() calls
```

## Disable Logging

Set `logger=None` in VLLMAgentWrapper:

```python
agent = VLLMAgentWrapper(..., logger=None)
```

## Common Issues

**Issue:** Logs directory filling up

**Solution:** Delete old sessions:
```bash
rm -rf logs/session_*
```

**Issue:** Can't open images

**Solution:** Use absolute paths or open from logs directory:
```bash
cd logs/session_YYYYMMDD_HHMMSS
eog observations/step_000000.png  # Linux
open observations/step_000000.png  # Mac
```

**Issue:** Want to analyze specific time range

**Solution:** Filter by timestamp:
```bash
# Get logs from specific time window
python3 -c "
import json
start, end = 1234567890, 1234567900
with open('agent_log.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if start <= obj['timestamp'] <= end:
            print(json.dumps(obj, indent=2))
"
```
