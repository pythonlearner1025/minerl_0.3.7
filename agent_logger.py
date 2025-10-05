"""
Simple logging system for debugging VLLMAgentWrapper behavior.
Logs: user commands, observations, prompts, VLLM responses, actions, history state.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image


class AgentLogger:
    """Logs all agent interactions to timestamped files."""

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create session directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.session_dir / "observations").mkdir(exist_ok=True)
        (self.session_dir / "prompts").mkdir(exist_ok=True)

        # Main log file
        self.log_file = self.session_dir / "agent_log.jsonl"
        self.step_count = 0

        print(f"[Logger] Session directory: {self.session_dir}")

    def log_user_command(self, command: str):
        """Log user input command."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "user_command",
            "step": self.step_count,
            "command": command
        }
        self._write_log(entry)
        print(f"[Logger] User command: {command}")

    def log_env_reset(self):
        """Log environment reset event."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "env_reset",
            "step": self.step_count
        }
        self._write_log(entry)
        print(f"[Logger] Environment reset")

    def log_observation(self, obs: np.ndarray):
        """Save observation image (raw input)."""
        img_path = self.session_dir / "observations" / f"step_{self.step_count:06d}_raw.png"
        Image.fromarray(obs.astype('uint8')).save(img_path)
        return str(img_path.relative_to(self.log_dir))

    def log_preprocessed_image(self, image_pil: Image.Image):
        """Save preprocessed image that's actually sent to VLLM."""
        img_path = self.session_dir / "observations" / f"step_{self.step_count:06d}_vllm.png"
        image_pil.save(img_path)
        return str(img_path.relative_to(self.log_dir))

    def log_forward_call(self, task: str, method: str, obs_shape: tuple,
                         history_len: int, instruction_type: str):
        """Log agent.forward() call details."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "forward_call",
            "step": self.step_count,
            "task": task,
            "method": method,
            "instruction_type": instruction_type,
            "obs_shape": obs_shape,
            "history_length": history_len
        }
        self._write_log(entry)

    def log_prompt(self, messages: list, prompt_text: str = None):
        """Log the full prompt sent to VLLM."""
        # Save detailed messages to separate file
        prompt_file = self.session_dir / "prompts" / f"step_{self.step_count:06d}.json"

        # Simplify messages for storage (remove base64 images)
        simplified_messages = []
        for msg in messages:
            simple_msg = {"role": msg["role"], "content": []}
            for content in msg["content"]:
                if content["type"] == "text":
                    simple_msg["content"].append({"type": "text", "text": content["text"]})
                elif content["type"] == "image_url":
                    simple_msg["content"].append({"type": "image", "note": "[image data omitted]"})
            simplified_messages.append(simple_msg)

        with open(prompt_file, 'w') as f:
            json.dump({
                "step": self.step_count,
                "num_messages": len(messages),
                "messages": simplified_messages,
                "prompt_text": prompt_text
            }, f, indent=2)

        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "prompt",
            "step": self.step_count,
            "num_messages": len(messages),
            "prompt_file": str(prompt_file.relative_to(self.log_dir)),
            "first_message_text": simplified_messages[0]["content"][0]["text"][:200] if simplified_messages else None
        }
        self._write_log(entry)

    def log_vllm_response(self, response_text: str, token_ids: list, wall_time_ms: float):
        """Log VLLM response."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "vllm_response",
            "step": self.step_count,
            "response_text": response_text[:500],  # Truncate long responses
            "response_length": len(response_text),
            "num_tokens": len(token_ids),
            "token_ids": token_ids,
            "wall_time_ms": wall_time_ms
        }
        self._write_log(entry)

    def log_action(self, action: dict):
        """Log decoded action."""
        # Simplify action for logging
        simple_action = {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in action.items()}

        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "action",
            "step": self.step_count,
            "action": simple_action
        }
        self._write_log(entry)

    def log_history_state(self, history: list):
        """Log agent's conversation history state."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "history_state",
            "step": self.step_count,
            "history_length": len(history),
            "history": [
                {
                    "action_text": h[1][:100],  # Truncate action text
                    "thought": h[2][:100],
                    "step_idx": h[3]
                }
                for h in history
            ] if history else []
        }
        self._write_log(entry)

    def log_error(self, error_msg: str, error_type: str = "unknown"):
        """Log error."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": "error",
            "step": self.step_count,
            "error_type": error_type,
            "error_msg": error_msg
        }
        self._write_log(entry)
        print(f"[Logger] ERROR: {error_msg}")

    def increment_step(self):
        """Increment step counter."""
        self.step_count += 1

    def _write_log(self, entry: dict):
        """Write log entry to file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def generate_summary(self):
        """Generate summary of session."""
        summary_file = self.session_dir / "summary.txt"

        # Read all log entries
        entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))

        # Generate summary
        with open(summary_file, 'w') as f:
            f.write(f"Agent Session Summary\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"Session directory: {self.session_dir}\n")
            f.write(f"Total steps: {self.step_count}\n\n")

            # Count entry types
            type_counts = {}
            for entry in entries:
                t = entry["type"]
                type_counts[t] = type_counts.get(t, 0) + 1

            f.write("Event counts:\n")
            for t, count in sorted(type_counts.items()):
                f.write(f"  {t}: {count}\n")

            # List user commands
            f.write("\n" + "=" * 70 + "\n")
            f.write("User commands:\n")
            for entry in entries:
                if entry["type"] == "user_command":
                    f.write(f"  [Step {entry['step']}] {entry['command']}\n")

            # List errors
            errors = [e for e in entries if e["type"] == "error"]
            if errors:
                f.write("\n" + "=" * 70 + "\n")
                f.write("Errors:\n")
                for entry in errors:
                    f.write(f"  [Step {entry['step']}] {entry['error_msg']}\n")

        print(f"[Logger] Summary written to {summary_file}")
