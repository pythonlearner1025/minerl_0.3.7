#!/usr/bin/env python3
"""
VLLMAgentWrapper for minerl 0.3.7
Ports JarvisVLA's VLLM_AGENT to work with legacy minerl API for human-interactive gameplay.

Key differences from original:
- No minestudio dependencies (uses raw minerl 0.3.7)
- Realtime observation loop (polls at intervals, no step blocking)
- Direct minerl action dict output (no minestudio callbacks)
"""

import time
import json
import copy
import base64
import random
import numpy as np
from pathlib import Path
from typing import Literal, Dict, List, Union, OrderedDict
from io import BytesIO
from datetime import datetime
from PIL import Image
from collections import Counter
from openai import OpenAI


# ============================================================================
# ACTION MAPPING (adapted from jarvisvla/inference/action_mapping.py)
# ============================================================================

def map_control_token(num: int, place: int, tokenizer_type: str = "qwen2_vl", not_text: bool = False) -> str:
    """Map a number at a specific position to a control token."""
    if tokenizer_type == "qwen2_vl":
        special_tokens = [
            # Group 0: hotbar (10 tokens)
            [["<|reserved_special_token_180|>", 151837], ["<|reserved_special_token_181|>", 151838],
             ["<|reserved_special_token_182|>", 151839], ["<|reserved_special_token_183|>", 151840],
             ["<|reserved_special_token_184|>", 151841], ["<|reserved_special_token_185|>", 151842],
             ["<|reserved_special_token_186|>", 151843], ["<|reserved_special_token_187|>", 151844],
             ["<|reserved_special_token_188|>", 151845], ["<|reserved_special_token_189|>", 151846]],
            # Group 1: forward/back (3 tokens)
            [["<|reserved_special_token_190|>", 151847], ["<|reserved_special_token_191|>", 151848],
             ["<|reserved_special_token_192|>", 151849]],
            # Group 2: left/right (3 tokens)
            [["<|reserved_special_token_193|>", 151850], ["<|reserved_special_token_194|>", 151851],
             ["<|reserved_special_token_195|>", 151852]],
            # Group 3: sprint/sneak (3 tokens)
            [["<|reserved_special_token_196|>", 151853], ["<|reserved_special_token_197|>", 151854],
             ["<|reserved_special_token_198|>", 151855]],
            # Group 4: use (2 tokens)
            [["<|reserved_special_token_199|>", 151856], ["<|reserved_special_token_200|>", 151857]],
            # Group 5: drop (2 tokens)
            [["<|reserved_special_token_201|>", 151858], ["<|reserved_special_token_202|>", 151859]],
            # Group 6: attack (2 tokens)
            [["<|reserved_special_token_203|>", 151860], ["<|reserved_special_token_204|>", 151861]],
            # Group 7: jump (2 tokens)
            [["<|reserved_special_token_205|>", 151862], ["<|reserved_special_token_206|>", 151863]],
            # Group 8: camera (2 tokens)
            [["<|reserved_special_token_207|>", 151864], ["<|reserved_special_token_208|>", 151865]],
            # Group 9: inventory (2 tokens)
            [["<|reserved_special_token_176|>", 151833], ["<|reserved_special_token_177|>", 151834]],
            # Group 10: camera_x (21 tokens)
            [["<|reserved_special_token_209|>", 151866], ["<|reserved_special_token_210|>", 151867],
             ["<|reserved_special_token_211|>", 151868], ["<|reserved_special_token_212|>", 151869],
             ["<|reserved_special_token_213|>", 151870], ["<|reserved_special_token_214|>", 151871],
             ["<|reserved_special_token_215|>", 151872], ["<|reserved_special_token_216|>", 151873],
             ["<|reserved_special_token_217|>", 151874], ["<|reserved_special_token_218|>", 151875],
             ["<|reserved_special_token_219|>", 151876], ["<|reserved_special_token_220|>", 151877],
             ["<|reserved_special_token_221|>", 151878], ["<|reserved_special_token_222|>", 151879],
             ["<|reserved_special_token_223|>", 151880], ["<|reserved_special_token_224|>", 151881],
             ["<|reserved_special_token_225|>", 151882], ["<|reserved_special_token_226|>", 151883],
             ["<|reserved_special_token_227|>", 151884], ["<|reserved_special_token_228|>", 151885],
             ["<|reserved_special_token_229|>", 151886]],
            # Group 11: camera_y (21 tokens)
            [["<|reserved_special_token_230|>", 151887], ["<|reserved_special_token_231|>", 151888],
             ["<|reserved_special_token_232|>", 151889], ["<|reserved_special_token_233|>", 151890],
             ["<|reserved_special_token_234|>", 151891], ["<|reserved_special_token_235|>", 151892],
             ["<|reserved_special_token_236|>", 151893], ["<|reserved_special_token_237|>", 151894],
             ["<|reserved_special_token_238|>", 151895], ["<|reserved_special_token_239|>", 151896],
             ["<|reserved_special_token_240|>", 151897], ["<|reserved_special_token_241|>", 151898],
             ["<|reserved_special_token_242|>", 151899], ["<|reserved_special_token_243|>", 151900],
             ["<|reserved_special_token_244|>", 151901], ["<|reserved_special_token_245|>", 151902],
             ["<|reserved_special_token_246|>", 151903], ["<|reserved_special_token_247|>", 151904],
             ["<|reserved_special_token_248|>", 151905], ["<|reserved_special_token_249|>", 151906],
             ["<|reserved_special_token_250|>", 151907]],
        ]
        return special_tokens[place][num][not_text]
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported")


def tag_token(place: int, tokenizer_type: str = "qwen2_vl", return_type: int = 0):
    """Return start/end tag tokens."""
    assert place in {0, 1}
    if tokenizer_type == "qwen2_vl":
        special_tokens = [
            ('<|reserved_special_token_178|>', 151835),  # start tag
            ('<|reserved_special_token_179|>', 151836),  # end tag
        ]
        return special_tokens[place][return_type]
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported")


def remap_control_token(token: int, tokenizer_type: str = "qwen2_vl") -> tuple:
    """Map a control token back to (group_idx, num)."""
    if tokenizer_type == "qwen2_vl":
        re_tokens = {
            151837: [0, 0], 151838: [0, 1], 151839: [0, 2], 151840: [0, 3], 151841: [0, 4],
            151842: [0, 5], 151843: [0, 6], 151844: [0, 7], 151845: [0, 8], 151846: [0, 9],
            151847: [1, 0], 151848: [1, 1], 151849: [1, 2],
            151850: [2, 0], 151851: [2, 1], 151852: [2, 2],
            151853: [3, 0], 151854: [3, 1], 151855: [3, 2],
            151856: [4, 0], 151857: [4, 1],
            151858: [5, 0], 151859: [5, 1],
            151860: [6, 0], 151861: [6, 1],
            151862: [7, 0], 151863: [7, 1],
            151864: [8, 0], 151865: [8, 1],
            151833: [9, 0], 151834: [9, 1],
            151866: [10, 0], 151867: [10, 1], 151868: [10, 2], 151869: [10, 3], 151870: [10, 4],
            151871: [10, 5], 151872: [10, 6], 151873: [10, 7], 151874: [10, 8], 151875: [10, 9],
            151876: [10, 10], 151877: [10, 11], 151878: [10, 12], 151879: [10, 13], 151880: [10, 14],
            151881: [10, 15], 151882: [10, 16], 151883: [10, 17], 151884: [10, 18], 151885: [10, 19],
            151886: [10, 20],
            151887: [11, 0], 151888: [11, 1], 151889: [11, 2], 151890: [11, 3],
            151891: [11, 4], 151892: [11, 5], 151893: [11, 6], 151894: [11, 7], 151895: [11, 8],
            151896: [11, 9], 151897: [11, 10], 151898: [11, 11], 151899: [11, 12], 151900: [11, 13],
            151901: [11, 14], 151902: [11, 15], 151903: [11, 16], 151904: [11, 17], 151905: [11, 18],
            151906: [11, 19], 151907: [11, 20],
        }
        return re_tokens.get(token, (-1, -1))
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported")


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def encode_image_to_base64(image: Union[np.ndarray, Image.Image], format='JPEG') -> str:
    """Encode image to base64 string for OpenAI API."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be numpy array or PIL Image")

    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def resize_image_for_vllm(image: Union[np.ndarray, Image.Image],
                          target_size: tuple = (640, 360)) -> Image.Image:
    """
    Resize image to match training resolution.

    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target (width, height) - default (640, 360) matches JarvisVLA training

    Returns:
        Resized PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    # Resize to target resolution (JarvisVLA was trained on 640x360)
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)

    return image


# ============================================================================
# VLLM AGENT WRAPPER
# ============================================================================

INSTRUCTION_TEMPLATE = [
    "help me to craft a {}.",
    "craft a {}.",
    "Craft a {}.",
    "Could you craft a {} for me?",
    "I need you to craft a {}.",
]


class VLLMAgentWrapper:
    """
    Agent wrapper that connects to VLLM server via OpenAI API.
    Manages conversation history, prompt construction, and action decoding.
    """

    def __init__(self,
                 base_url: str,
                 api_key: str = "EMPTY",
                 checkpoint_path: str = None,
                 history_num: int = 0,
                 action_chunk_len: int = 1,
                 instruction_type: Literal['simple', 'recipe', 'normal'] = 'normal',
                 temperature: float = 0.7,
                 tokenizer_type: str = "qwen2_vl",
                 action_space_keys: list = None,
                 logger=None):

        self.base_url = base_url
        self.api_key = api_key
        self.tokenizer_type = tokenizer_type
        self.history_num = history_num
        self.action_chunk_len = action_chunk_len
        self.instruction_type = instruction_type
        self.temperature = temperature

        # Action space filter (if None, use all actions)
        self.action_space_keys = action_space_keys

        # Logger (optional)
        self.logger = logger

        # OpenAI client for VLLM
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        models = self.client.models.list()
        self.model = models.data[0].id
        print(f"Connected to VLLM server: {self.model}")

        # Load prompt library and recipes
        assets_path = Path(__file__).parent / "assets"
        self.prompt_library = self._load_json(assets_path / "instructions.json")
        self.recipe_fold = assets_path / "recipes"
        self.recipes = {}

        # Action tokenizer config
        self.bases = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21]
        self.act_beg_id = tag_token(0, tokenizer_type, return_type=1)
        self.act_end_id = tag_token(1, tokenizer_type, return_type=1)

        # Conversation history
        self.history = []
        self.actions = []  # Action chunk buffer

        print(f"Agent initialized: history_num={history_num}, "
              f"instruction_type={instruction_type}, temp={temperature}")

    def _load_json(self, path: Path) -> dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def reset(self):
        """Reset conversation history."""
        self.history = []
        self.actions = []

    def _rule_based_instruction(self, env_prompt: str) -> str:
        """Generate simple instruction from env_prompt."""
        item_name = env_prompt[11:].replace("_", " ")  # Remove "craft item "
        instruction_template = random.choice(INSTRUCTION_TEMPLATE)
        return instruction_template.format(item_name)

    def _create_basic_instruction(self, env_prompt: str) -> str:
        """Create basic instruction from prompt library or rule-based."""
        instructions = self.prompt_library.get(env_prompt, {}).get("instruct")
        if instructions:
            instruction = np.random.choice(instructions)
        else:
            instruction = self._rule_based_instruction(env_prompt)

        if instruction.strip()[-1] != '.':
            instruction += ".\n"
        instruction += "\n"
        return instruction

    def _create_thought(self, env_prompt: str) -> str:
        """Extract thought from prompt library."""
        thought = self.prompt_library.get(env_prompt, {}).get("thought")
        if not thought:
            thought = env_prompt.replace("item", str(1)).replace("_", " ").replace(":", " ")
        thought += '.\n'
        return thought

    def _get_recipe_item_name(self, ingredient: dict) -> str:
        """Extract item name from recipe ingredient."""
        item_name = ingredient.get("item")
        if not item_name:
            item_name = ingredient.get("tag")
        return item_name

    def _create_recipe_prompt_from_library(self, item_name: str) -> str:
        """Load and format recipe from JSON files."""
        if item_name in self.recipes:
            return self.recipes[item_name]

        recipe_path = self.recipe_fold / f"{item_name}.json"
        if not recipe_path.exists():
            self.recipes[item_name] = ""
            return ""

        recipe_file = self._load_json(recipe_path)
        recipe_type = recipe_file.get("type", None)

        prompt = ""
        if not recipe_type:
            self.recipes[item_name] = ""
            return ""

        elif recipe_type == "minecraft:crafting_shapeless":
            prompt += "\nYou will need the following ingredients:\n"
            ingredients = recipe_file.get("ingredients", [])
            ingredients_list = []
            for ingredient in ingredients:
                ingredient_name = self._get_recipe_item_name(ingredient)
                if not ingredient_name:
                    break
                ingredients_list.append(ingredient_name[10:].replace("_", " "))
            ingredients_dict = Counter(ingredients_list)
            for item, number in ingredients_dict.items():
                prompt += f"{number} {item}, "
            prompt += "\n"

        elif recipe_type == "minecraft:crafting_shaped":
            prompt += "\nArrange the materials in the crafting grid according to the following pattern:\n"
            patterns = recipe_file.get("pattern", [])
            if not patterns:
                return ""
            ingredients = recipe_file.get("key", {})
            ingredients_dict = {}
            for ingredient_mark, value in ingredients.items():
                ingredient_name = self._get_recipe_item_name(value)
                ingredients_dict[ingredient_mark] = ingredient_name[10:]
            prompt += "\n"
            for pattern_line in patterns:
                for pattern_mark in pattern_line:
                    if pattern_mark == " ":
                        ingredients_name = "air"
                    else:
                        ingredients_name = ingredients_dict.get(pattern_mark, "air")
                    prompt += f" {ingredients_name} |"
                if prompt[-1] == '|':
                    prompt = prompt[:-1]
                    prompt += "\n"
            prompt += "\n"
        else:
            self.recipes[item_name] = ""
            return ""

        result_num = recipe_file.get("result", {}).get("count", 1)
        prompt += f"and get {result_num} {item_name.replace('_', ' ')}.\n"
        self.recipes[item_name] = prompt
        return prompt

    def _create_recipe_prompt(self, env_prompt: str, method: str = "crafting_table") -> str:
        """Create recipe prompt from env_prompt."""
        prompt = ""
        if "recipe_book" in method:
            prompt += "\nUse the recipe book to craft.\n"
            return prompt

        recipe = self.prompt_library.get(env_prompt, {}).get("recipe")
        if recipe:
            prompt += "\nArrange the materials in the crafting grid according to the following pattern:\n"
            prompt += recipe[0]
        else:
            item_name = env_prompt.replace(" ", "_").split(":")[-1]
            prompt += self._create_recipe_prompt_from_library(item_name)
        return prompt

    def create_instruction(self, env_prompt: str, method: str = "inventory") -> str:
        """
        Create full instruction prompt based on instruction_type.

        Args:
            env_prompt: Task description (e.g., "craft item crafting_table")
            method: "inventory" or "crafting_table"

        Returns:
            Formatted instruction string
        """
        if self.instruction_type == 'recipe':
            prompt = self._create_basic_instruction(env_prompt)
            recipe_prompt = self._create_recipe_prompt(env_prompt, method=method)
            prompt += recipe_prompt
        elif self.instruction_type == 'simple':
            prompt = self._create_thought(env_prompt)
        elif self.instruction_type == 'normal':
            natural_text = env_prompt.replace("_", " ").replace(":", " ")
            prompt = random.choice(
                self.prompt_library.get(env_prompt, {"instruct": [natural_text]})["instruct"]
            )
        # direct prompt, no querying from prebuilt lib 
        elif self.instruction_type == 'freestyle':
            prompt = env_prompt
        else:
            raise ValueError(f"Unknown instruction_type: {self.instruction_type}")

        return prompt

    def _create_image_message(self, image: Union[np.ndarray, Image.Image]) -> dict:
        """Create OpenAI API image message from observation."""
        # Convert to PIL if needed (no resize - env already provides 640x360)
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype('uint8'))
        else:
            image_pil = image

        # Log preprocessed image that's actually sent to VLLM
        if self.logger:
            self.logger._write_log({
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "type": "image_preprocessing",
                "step": self.logger.step_count,
                "image_size": image_pil.size  # (width, height)
            })
            self.logger.log_preprocessed_image(image_pil)

        image_b64 = encode_image_to_base64(image_pil)

        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        }

    def _create_vllm_message(self,
                             role: Literal["user", "assistant"],
                             prompt: str,
                             image: Union[np.ndarray, Image.Image] = None) -> dict:
        """
        Create message for VLLM API.

        Args:
            role: "user" or "assistant"
            prompt: Text prompt
            image: Optional image (for user messages)

        Returns:
            OpenAI-compatible message dict
        """
        message = {"role": role, "content": []}

        # Add text
        message["content"].append({"type": "text", "text": prompt})

        # Add image if provided
        if image is not None:
            message["content"].append(self._create_image_message(image))

        return message

    def _token_to_group_actions(self, tokens: List[int]) -> List[List[int]]:
        """
        Convert token sequence to group action representations.

        Args:
            tokens: List of token IDs from VLLM output

        Returns:
            List of group actions (each is a list of 12 integers)
        """
        actions = []
        action_base = [0] * len(self.bases)
        camera_null = [self.bases[-1] // 2, self.bases[-2] // 2]
        action_base[-2:] = camera_null

        start_idx = 0
        while start_idx < len(tokens):
            try:
                first_index_n1 = tokens.index(self.act_beg_id, start_idx)
                first_index_n2 = tokens.index(self.act_end_id, first_index_n1 + 1)
            except ValueError:
                break

            control_tokens = tokens[first_index_n1 + 1:first_index_n2]
            action = copy.copy(action_base)

            for token in control_tokens:
                place, num = remap_control_token(token, tokenizer_type=self.tokenizer_type)
                if place != -1:
                    action[place] = num

            # If camera changed, set inventory flag
            if action[-2:] != camera_null:
                action[-4] = 1

            actions.append(copy.copy(action))
            start_idx = first_index_n2 + 1

        if len(actions) == 0:
            actions.append(action_base)

        return actions

    def _group_action_to_minerl_action(self, group_action: List[int]) -> dict:
        """
        Convert group action representation to minerl action dict.

        Args:
            group_action: List of 12 integers representing the action

        Returns:
            minerl-compatible action dict
        """
        # TODO(human): Implement the mapping from group_action to minerl action space
        # Group indices: 0=hotbar, 1=forward/back, 2=left/right, 3=sprint/sneak,
        #                4=use, 5=drop, 6=attack, 7=jump, 8=camera_button,
        #                9=inventory, 10=camera_x, 11=camera_y
        #
        # Minerl action space (example from minerl 0.3.7):
        # {
        #     'camera': [pitch_delta, yaw_delta],  # float32, range ~[-180, 180]
        #     'forward': 0/1,
        #     'back': 0/1,
        #     'left': 0/1,
        #     'right': 0/1,
        #     'jump': 0/1,
        #     'sprint': 0/1,
        #     'sneak': 0/1,
        #     'attack': 0/1,
        #     'use': 0/1,
        #     'drop': 0/1,
        #     'inventory': 0/1,
        #     ... other keys depending on env
        # }
        #
        # Camera quantization: groups 10 & 11 are quantized camera deltas (0-20 bins)
        # Need to map back to continuous values using mu-law inverse
        def inverse_mu_law(bin_val, n_bins=21, mu=20, maxval=10, binsize=1):
            """
            Inverse mu-law dequantization for camera movement.

            Args:
                bin_val: Quantized bin value (0-20)
                n_bins: Number of bins (21 for JarvisVLA)
                mu: Mu-law parameter (20 for JarvisVLA)
                maxval: Maximum camera value (10 degrees)
                binsize: Bin size (1 for JarvisVLA)

            Returns:
                Continuous camera angle delta
            """
            # Convert bin to value: bin * binsize - maxval
            xy = bin_val * binsize - maxval

            # Normalize
            xy_norm = xy / maxval

            # Mu-law decode: sign(xy) * (1/mu) * ((1 + mu)^|xy| - 1)
            if xy_norm == 0:
                result_norm = 0
            else:
                sign = 1 if xy_norm > 0 else -1
                result_norm = sign * (1.0 / mu) * (np.power(1 + mu, abs(xy_norm)) - 1)

            # Scale back
            result = result_norm * maxval
            return result

        # Build complete action dict from group_action
        full_action = {
            # Camera: [pitch, yaw] in degrees
            "camera": np.array([
                inverse_mu_law(group_action[11]),  # pitch (camera_y)
                inverse_mu_law(group_action[10])   # yaw (camera_x)
            ], dtype=np.float32),

            # Movement
            "forward": int(group_action[1] == 1),
            "back": int(group_action[1] == 2),
            "left": int(group_action[2] == 1),
            "right": int(group_action[2] == 2),

            # Sprint/Sneak
            "sprint": int(group_action[3] == 1),
            "sneak": int(group_action[3] == 2),

            # Actions
            "jump": int(group_action[7] == 1),
            "attack": int(group_action[6] == 1),
            "use": int(group_action[4] == 1),
            "drop": int(group_action[5] == 1),
            "inventory": int(group_action[9] == 1),
        }

        # Add hotbar slots
        for i in range(9):
            full_action[f"hotbar.{i+1}"] = int(group_action[0] == i)

        # Filter action dict if action_space_keys was provided
        if self.action_space_keys is not None:
            action = {k: v for k, v in full_action.items() if k in self.action_space_keys}
        else:
            action = full_action

        return action

    def _decode_tokens_to_actions(self, tokens: List[int]) -> List[dict]:
        """
        Decode VLLM output tokens to minerl action dicts.

        Args:
            tokens: List of token IDs from VLLM

        Returns:
            List of minerl action dicts
        """
        group_actions = self._token_to_group_actions(tokens)
        minerl_actions = [self._group_action_to_minerl_action(ga) for ga in group_actions]
        return minerl_actions

    def forward(self, observation: np.ndarray, task_instruction: str,
                method: str = "inventory", verbose: bool = False) -> dict:
        """
        Run inference on observation and return minerl action.

        Args:
            observation: RGB image observation from minerl (H x W x 3)
            task_instruction: Task description (e.g., "craft item crafting_table")
            method: "inventory" or "crafting_table"
            verbose: Print debug info

        Returns:
            minerl action dict
        """
        # Log forward call
        if self.logger:
            self.logger.log_forward_call(
                task=task_instruction,
                method=method,
                obs_shape=observation.shape,
                history_len=len(self.history),
                instruction_type=self.instruction_type
            )
            self.logger.log_observation(observation)
            self.logger.log_history_state(self.history)

        # If we have buffered actions, return next one
        if self.actions:
            action = self.actions.pop(0)
            if verbose:
                print(f"[Buffered action] {len(self.actions)} remaining")
            if self.logger:
                self.logger.log_action(action)
            return action

        # Build messages for VLLM
        messages = []

        # Add history if enabled
        if self.history_num and self.history:
            for hist_image, hist_action, hist_thought, hist_idx in self.history:
                if self.instruction_type == 'recipe':
                    prompt = f"\nthought: {hist_thought}\nobservation: "
                else:
                    prompt = "\nobservation: "

                # First message includes full instruction
                if hist_idx == 0:
                    instruction = self.create_instruction(task_instruction, method=method)
                    prompt = instruction + prompt

                messages.append(self._create_vllm_message("user", prompt, hist_image))
                messages.append(self._create_vllm_message("assistant", hist_action, None))

        # Create current observation message
        instruction = self.create_instruction(task_instruction, method=method)
        thought = self._create_thought(task_instruction) if self.instruction_type == "recipe" else ""

        if self.instruction_type == 'recipe':
            prompt = f"\nthought: {thought}\nobservation: "
        else:
            prompt = "\nobservation: "

        if not self.history_num:
            prompt = instruction + prompt

        messages.append(self._create_vllm_message("user", prompt, observation))

        # Log prompt
        if self.logger:
            self.logger.log_prompt(messages, prompt_text=instruction)

        # Call VLLM
        if verbose:
            print(f"[VLLM] Calling with {len(messages)} messages")

        vllm_start = time.time()
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
            top_p=0.99,
            extra_body={"skip_special_tokens": False, "top_k": -1}
        )

        outputs = chat_completion.choices[0].message.content
        vllm_end = time.time()
        vllm_time_ms = (vllm_end - vllm_start) * 1000

        if verbose:
            print(f"[VLLM] Response: {outputs[:100]}...")
            print(f"[VLLM] Time: {vllm_time_ms:.1f}ms")

        # Extract special token IDs using regex
        # Match pattern: <|reserved_special_token_NNN|> where NNN is 176-250
        import re
        pattern = r'<\|reserved_special_token_(\d+)\|>'
        matches = re.findall(pattern, outputs)

        token_ids = []
        for match in matches:
            token_num = int(match)
            token_id = 151657 + token_num  # Qwen2-VL special token base offset
            token_ids.append(token_id)

        if not token_ids:
            if verbose:
                print("[Warning] No action tokens found in response, using null action")
            if self.logger:
                self.logger.log_error("No action tokens in VLLM response", "token_parsing")
            return self._null_action()

        if verbose:
            print(f"[VLLM] Extracted {len(token_ids)} special tokens: {token_ids[:5]}...")

        # Log VLLM response
        if self.logger:
            self.logger.log_vllm_response(outputs, token_ids, vllm_time_ms)

        # Update history if enabled
        if self.history_num:
            new_hist_entry = (observation, outputs, thought,
                             self.history[-1][-1] + 1 if self.history else 0)
            if not self.history:
                # Initialize history with null actions
                null_action = self._null_action_token()
                self.history = [(observation, null_action, thought, 0)] * self.history_num
            else:
                self.history = self.history[1:] + [new_hist_entry]

        # Decode tokens to actions
        actions = self._decode_tokens_to_actions(token_ids)

        # Buffer actions if chunk length > 1
        if len(actions) > 0:
            self.actions = actions[:self.action_chunk_len]
            action = self.actions.pop(0)
            if self.logger:
                self.logger.log_action(action)
            return action
        else:
            # Return null action if no valid actions decoded
            if self.logger:
                self.logger.log_error("No valid actions decoded", "action_decoding")
            return self._null_action()

    def _null_action_token(self) -> str:
        """Get null action token string."""
        null_bin = self.bases[-1] // 2
        null_group = [0] * len(self.bases)
        null_group[-2:] = [null_bin, null_bin]

        # Convert to token string
        tokens = []
        for i, num in enumerate(null_group[-2:]):
            tokens.append(map_control_token(num, i + 10, self.tokenizer_type))

        return self.act_beg_token + "".join(tokens) + self.act_end_token

    def _null_action(self) -> dict:
        """Get null action dict (no movement)."""
        null_bin = self.bases[-1] // 2
        null_group = [0] * len(self.bases)
        null_group[-2:] = [null_bin, null_bin]
        return self._group_action_to_minerl_action(null_group)


# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    import gym
    import minerl

    print("=" * 70)
    print("VLLM AGENT WRAPPER TEST")
    print("=" * 70)

    # Initialize agent
    print("\n[1/4] Initializing VLLM agent...")
    agent = VLLMAgentWrapper(
        base_url="http://localhost:8000/v1",  # Update with your VLLM server URL
        checkpoint_path=None,  # Optional: path to model for tokenizer
        history_num=0,  # Number of history frames (0 = no history)
        action_chunk_len=1,  # Number of actions to predict per inference
        instruction_type='normal',  # 'normal', 'recipe', or 'simple'
        temperature=0.7
    )
    print("✓ Agent initialized")

    # Create minerl environment
    print("\n[2/4] Creating MineRL environment...")
    env = gym.make('MineRLTreechop-v0')
    print("✓ Environment created")

    # Reset environment
    print("\n[3/4] Resetting environment...")
    obs = env.reset()
    print(f"✓ Environment reset - Observation shape: {obs['pov'].shape}")

    # Test forward pass
    print("\n[4/4] Testing agent forward pass...")
    task = "craft item crafting_table"
    try:
        action = agent.forward(
            observation=obs['pov'],
            task_instruction=task,
            method="inventory",
            verbose=True
        )
        print(f"✓ Action generated: {action}")
        print(f"  - Camera: {action['camera']}")
        print(f"  - Forward: {action['forward']}")
        print(f"  - Jump: {action['jump']}")
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    env.close()
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nNote: This test requires a running VLLM server.")
    print("Start VLLM server with:")
    print("  vllm serve <model_path> --port 8000")
    print("=" * 70)
