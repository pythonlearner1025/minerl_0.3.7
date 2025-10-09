#!/usr/bin/env python3

import argparse
import time
import gym
import minerl
import numpy as np
from agent_wrapper import VLLMAgentWrapper

import threading
import queue
from agent_logger import AgentLogger

# Simple crafting detection wrapper
class CraftingDetector:
    def __init__(self, craft_item):
        self.craft_item = craft_item
        self.initial_count = 0
        self.goal_achieved = False

    def reset(self, obs):
        if 'inventory' in obs and self.craft_item in obs['inventory']:
            self.initial_count = obs['inventory'][self.craft_item]
        else:
            self.initial_count = 0
        self.goal_achieved = False

    def check(self, obs, reward, done, info):
        if 'inventory' in obs and self.craft_item in obs['inventory']:
            current_count = obs['inventory'][self.craft_item]
            if current_count > self.initial_count and current_count > 0:
                self.goal_achieved = True
                reward = 1.0
                done = True
                info['goal_achieved'] = True
                info[f'{self.craft_item}_count'] = int(current_count)
                print(f"✓ {self.craft_item} crafted! Count: {current_count}")
        return reward, done, info

input_queue = queue.Queue()

def get_user_input():
    while True:
        msg = input("User command: ")
        input_queue.put(msg)

# begin user input thread
input_thread = threading.Thread(target=get_user_input, daemon=1)
input_thread.start()

def run_realtime_agent(base_url: str,
                      task: str,
                      env_name: str = 'MineRLTreechop-v0',
                      max_steps: int = 200,
                      sample_interval: float = 0.5,
                      history_num: int = 0,
                      instruction_type: str = 'normal',
                      temperature: float = 0.7,
                      verbose: bool = True,
                      craft_item: str = None):

    # Initialize logger
    logger = AgentLogger(log_dir="logs")

    # Create environment first to get action space
    print(f"\n[1/4] Creating environment: {env_name}")
    env = gym.make(env_name)

    # Get action space keys
    action_space_keys = list(env.action_space.spaces.keys())
    print(f"✓ Environment action space: {action_space_keys}")

    # Initialize agent with action space
    print("\n[2/4] Initializing VLLM agent...")
    agent = VLLMAgentWrapper(
        base_url=base_url,
        history_num=history_num,
        action_chunk_len=1,
        instruction_type=instruction_type,
        temperature=temperature,
        action_space_keys=action_space_keys,  # Filter actions to match env
        logger=logger  # Add logger
    )
    print(f"✓ Agent initialized (temp={temperature}, history={history_num})")

    # Enable realtime mode
    print("\n[3/4] Enabling realtime mode...")
    env.make_interactive(port=6666, realtime=True)  # Uncomment for GUI mode
    print("✓ Realtime mode enabled")

    # Reset environment
    print("\n[4/4] Starting episode...")
    obs = env.reset()
    print(f"✓ Episode started - Task: {task}")

    # Initialize crafting detector if specified
    crafting_detector = None
    if craft_item:
        crafting_detector = CraftingDetector(craft_item)
        crafting_detector.reset(obs)
        print(f"✓ Crafting detection enabled for: {craft_item}")
        if 'inventory' in obs:
            print(f"✓ Initial {craft_item} count: {obs['inventory'].get(craft_item, 0)}")

    print("AGENT LOOP RUNNING")
    print(f"Sample interval: {sample_interval}s ({1/sample_interval:.1f} fps)")
    print(f"Max steps: {max_steps}")

    step_count = 0
    last_sample_time = time.time()
    episode_reward = 0.0
    first = True
    method = "inventory"
    try:
        while step_count < max_steps:
            current_time = time.time()

            # get user input
            try:
                user_msg = input_queue.get_nowait()
                logger.log_user_command(user_msg)

                # Ignore empty commands
                if not user_msg.strip():
                    print("(Empty command ignored)")
                elif user_msg.lower() == "reset":
                    print("Resetting env...")
                    obs = env.reset()
                    agent.reset()
                    step_count = 0
                    episode_reward = 0.0
                    logger.log_env_reset()
                else:
                    agent.reset()
                    task = user_msg
                    method = "freestyle"
                    print(f"New Task: {task}")
            except queue.Empty:
                pass

            # Get agent action
            if verbose:
                print(f"\n[Step {step_count}] Getting action from agent...")
            
            s = time.time()
            print(f'task: {task}') 
            action = agent.forward(
                observation=obs['pov'],
                task_instruction=task,
                method=method,
                verbose=verbose
            )
            e = time.time()
            if verbose:
                print(f"wall clock time: {(e-s)*1000:0.2f}") 
                print(f"  Action: forward={action['forward']}, "
                      f"jump={action['jump']}, attack={action['attack']}, "
                      f"camera={action['camera']}")

            obs, reward, done, info = env.step(action)

            # Check crafting detection if enabled
            if crafting_detector:
                reward, done, info = crafting_detector.check(obs, reward, done, info)

            episode_reward += reward
            if reward > 0 and verbose:
                print(f"  ✓ REWARD: {reward} (total: {episode_reward})")

            if done:
                print(f"\n✓ Episode completed at step {step_count}!")
                print(f"  Total reward: {episode_reward}")
                if crafting_detector and crafting_detector.goal_achieved:
                    print(f"  🎉 Goal achieved: {craft_item} crafted!")
                break

            step_count += 1
            logger.increment_step()
            last_sample_time = current_time

            if first:
                print("waiting for user to connect")
                #input()
                first = False

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    except Exception as e:
        logger.log_error(str(e), error_type=type(e).__name__)
        raise

    # Cleanup
    env.close()
    logger.generate_summary()
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Steps completed: {step_count}")
    print(f"Total reward: {episode_reward}")
    print(f"Task: {task}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Run JarvisVLA agent in realtime mode")

    # VLLM settings
    parser.add_argument('--base-url', type=str, default="http://localhost:3000/v1",
                       help='VLLM server URL (e.g., http://localhost:8000/v1)')

    # Task settings
    parser.add_argument('--task', type=str, default='Mine the oak log.',
                       help='Task instruction')
    parser.add_argument('--env', type=str, default='MineRLTreechop-v0',
                       help='MineRL environment name')
    parser.add_argument('--craft', type=str, default="oak_log",
                       help='Detect when this item is crafted and end episode (e.g., crafting_table)')

    # Agent settings
    parser.add_argument('--history-num', type=int, default=2,
                       help='Number of conversation history frames (0 = disabled)')
    parser.add_argument('--instruction-type', type=str, default='normal',
                       choices=['normal', 'recipe', 'simple'],
                       help='Instruction type')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')

    # Execution settings
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum number of agent steps')
    parser.add_argument('--sample-interval', type=float, default=0.5,
                       help='Seconds between observations (default: 0.5 = 2 fps)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print debug info')

    args = parser.parse_args()

    # Auto-switch to environment with inventory if craft detection is enabled
    if args.craft and args.env == 'MineRLTreechop-v0':
        print("Auto-switching to MineRLObtainTest-v0 (has inventory + crafting)")
        args.env = 'MineRLObtainTest-v0'

    run_realtime_agent(
        base_url=args.base_url,
        task=args.task,
        env_name=args.env,
        max_steps=args.max_steps,
        sample_interval=args.sample_interval,
        history_num=args.history_num,
        instruction_type=args.instruction_type,
        temperature=args.temperature,
        verbose=args.verbose,
        craft_item=args.craft
    )


if __name__ == "__main__":
    main()
