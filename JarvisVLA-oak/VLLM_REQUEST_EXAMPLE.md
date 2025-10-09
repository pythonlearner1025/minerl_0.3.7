# VLLM Request Structure for JarvisVLA

## Parameter Explanations

### 1. History Frames (history_num: 2)
**What it does**: Controls how many previous observation frames are included as context.

- `history_num = 0`: Only current frame → No temporal context
- `history_num = 2`: Current frame + 2 previous frames → 3 frames total

**Why it matters**: Gives the model temporal context to understand:
- Movement and velocity (e.g., "am I moving forward?")
- Changes over time (e.g., "did I just break a block?")
- Action consequences (e.g., "my last action moved me closer to the tree")

### 2. Action Chunk Length (action_chunk_len: 1)
**What it does**: How many future actions the model predicts in one inference call.

- `action_chunk_len = 1`: Predict only next action → More responsive, more API calls
- `action_chunk_len = 4`: Predict next 4 actions → Fewer API calls, less reactive

**Why it matters**: Trade-off between:
- **Responsiveness**: Shorter chunks adapt faster to changes
- **Efficiency**: Longer chunks reduce VLLM inference calls (expensive!)

---

## Example VLLM Request

### Scenario: Mining Oak Logs with history_num=2

**Step 1: First Frame (No History Yet)**
```python
# Initial state - no history available
# System creates placeholder history with null actions

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Mine the oak log.\nobservation: "},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<frame_0>"}}
        ]
    }
]
```

**Step 2: After 2 Steps (History Built Up)**

Now with `history_num=2`, the request includes the last 2 observations + actions:

```python
messages = [
    # ===== HISTORY FRAME 1 (t-2) =====
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Mine the oak log.\nobservation: "  # Instruction only on first frame
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,<frame_t-2>"}
            }
        ]
    },
    {
        "role": "assistant",
        "content": "<action_0_5_1_1_0_0_0_0_0_0_10_10>"  # Action taken at t-2
    },

    # ===== HISTORY FRAME 2 (t-1) =====
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "\nobservation: "  # No instruction repeat
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,<frame_t-1>"}
            }
        ]
    },
    {
        "role": "assistant",
        "content": "<action_0_5_1_1_0_0_0_0_0_0_12_10>"  # Action taken at t-1
    },

    # ===== CURRENT FRAME (t) =====
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "\nobservation: "
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,<frame_t>"}  # Current frame
            }
        ]
    }
]

# === API Call ===
response = client.chat.completions.create(
    messages=messages,
    model="CraftJarvis/JarvisVLA-Qwen2-VL-7B",
    temperature=0.6,
    max_tokens=1024,
    top_p=0.99,
    extra_body={
        "skip_special_tokens": False,
        "top_k": -1
    }
)

# Response from VLLM
# With action_chunk_len=1: Returns 1 action token
output = response.choices[0].message.content
# Example: "<action_0_5_1_1_0_0_0_1_0_0_11_10>"
```

---

## Complete Request Structure Breakdown

### Message Format with History

```
[User Message 1 - History Frame]
  ├─ Text: "Mine the oak log.\nobservation: "
  └─ Image: Base64-encoded frame from 2 steps ago

[Assistant Message 1 - Past Action]
  └─ Text: "<action_...>"  (the action that was taken)

[User Message 2 - History Frame]
  ├─ Text: "\nobservation: "
  └─ Image: Base64-encoded frame from 1 step ago

[Assistant Message 2 - Past Action]
  └─ Text: "<action_...>"

[User Message 3 - Current Frame]
  ├─ Text: "\nobservation: "
  └─ Image: Base64-encoded current frame

[Model predicts next action based on all context]
```

### Action Token Format

Actions are encoded as special tokens:
```
<action_0_5_1_1_0_0_0_1_0_0_11_10>
         │ │ │ │ │ │ │ │ │ │ ││ └─ Camera yaw (bin 10)
         │ │ │ │ │ │ │ │ │ │ └─── Camera pitch (bin 11)
         │ │ │ │ │ │ │ │ │ └───── Place (0=no)
         │ │ │ │ │ │ │ │ └─────── Attack (0=no)
         │ │ │ │ │ │ │ └───────── Use (1=yes, using axe)
         │ │ │ │ │ │ └─────────── Sprint (0=no)
         │ │ │ │ │ └───────────── Sneak (0=no)
         │ │ │ │ └─────────────── Jump (0=no)
         │ │ │ └───────────────── Left/Right (1)
         │ │ └─────────────────── Forward/Back (1)
         │ └───────────────────── Strafe (5)
         └─────────────────────── Movement base (0)
```

This corresponds to action space bases: `[10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21]`

---

## With action_chunk_len > 1

### Example: action_chunk_len=4

**Model Output:**
```
<action_0_5_1_1_0_0_0_1_0_0_11_10><action_0_5_1_1_0_0_0_1_0_0_11_10><action_0_5_1_1_0_0_0_1_0_0_12_10><action_0_5_1_1_0_0_0_0_0_0_12_10>
```

**Behavior:**
- Agent stores all 4 actions in memory
- Returns action 1, executes it
- Next 3 frames: No VLLM call, just returns cached actions
- After 4 frames: Makes new VLLM call with updated history

**Benefits:**
- 75% reduction in VLLM API calls
- Faster overall execution (no inference latency every frame)

**Drawbacks:**
- Less reactive to unexpected events
- Can't adjust mid-sequence if something changes

---

## Real Example from Oak Log Task

### Request at Frame 50 (with history_num=2)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Mine the oak log.\\nobservation: "},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
      ]
    },
    {
      "role": "assistant",
      "content": "<action_3_1_1_1_0_0_0_0_0_0_10_11>"
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "\\nobservation: "},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
      ]
    },
    {
      "role": "assistant",
      "content": "<action_3_1_1_1_0_0_0_0_0_0_10_12>"
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "\\nobservation: "},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
      ]
    }
  ],
  "model": "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
  "temperature": 0.6,
  "max_tokens": 1024,
  "top_p": 0.99,
  "extra_body": {
    "skip_special_tokens": false,
    "top_k": -1
  }
}
```

### Response from VLLM

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1728381234,
  "model": "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<action_5_1_1_1_0_0_0_1_0_0_11_10>"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 2847,  # 3 images + text + history
    "completion_tokens": 15,
    "total_tokens": 2862
  }
}
```

---

## Key Insights

### History Provides Temporal Context
Without history, the model sees a single snapshot and must infer:
- "Am I moving?" → Can't tell from one frame
- "Did my action work?" → No feedback

With history_num=2:
- Sees 3 consecutive frames
- Can infer velocity, action effects, and temporal patterns
- Better decision-making for dynamic tasks

### Action Chunking Trades Off Reactivity for Speed
- **action_chunk_len=1**: Reactive (re-plan every step) but slow (VLLM call per step)
- **action_chunk_len=4**: Fast (1 VLLM call per 4 steps) but commits to a sequence

### Token Efficiency
Each VLLM call with history_num=2 uses:
- ~900 tokens per image (Qwen2-VL image encoding)
- 3 images = ~2700 tokens
- Text prompts = ~100 tokens
- **Total**: ~2800 tokens per inference

With action_chunk_len=4, this cost is amortized over 4 frames.

---

## Configuration Impact

| Config | VLLM Calls | Tokens/Call | Total Tokens (500 frames) | Reactivity |
|--------|------------|-------------|---------------------------|------------|
| history=0, chunk=1 | 500 | ~1000 | 500K | Highest |
| history=2, chunk=1 | 500 | ~2800 | 1.4M | High |
| history=2, chunk=4 | 125 | ~2800 | 350K | Medium |
| history=4, chunk=8 | 62 | ~4500 | 279K | Low |

**Your Configuration** (history=2, chunk=1):
- High reactivity (can adapt every frame)
- High token usage (~1.4M tokens for 500 frames)
- Best for tasks requiring precise, adaptive control
- Ideal for oak log gathering where finding trees requires exploration
