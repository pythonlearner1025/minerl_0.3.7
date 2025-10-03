#!/usr/bin/env python3
"""
MineRL Interactive Server Test with UUID Fix
Run this first to start the server on port 6666
"""
import gym
import minerl
import time

print("=" * 70)
print("MINERL INTERACTIVE SERVER (WITH UUID FIX)")
print("=" * 70)

print("\n[1/3] Creating MineRLTreechop-v0 environment...")
env = gym.make('MineRLTreechop-v0')
print("✓ Environment created")

print("\n[2/3] Making environment interactive on port 6666...")
env.make_interactive(port=6666, realtime=True)
print("✓ Interactive mode enabled")

print("\n[3/3] Resetting environment...")
obs = env.reset()
print("✓ Environment reset - Minecraft agent is running")

print("\n" + "=" * 70)
print("✓ SERVER READY - Port 6666 is OPEN!")
print("=" * 70)
print("\nNow in another terminal, run:")
print("  cd /tmp")
print("  source ~/minerl_venv/bin/activate")
print("  python3 test_minerl_client_final.py")
print("=" * 70)

print("\nServer running for 300 seconds (5 minutes)...")
print("Watch for '[MineRL UUID Fix]' messages in logs when client connects")
print("")

start = time.time()
step = 0
try:
    while time.time() - start < 300:
        obs, reward, done, info = env.step(env.action_space.noop())
        step += 1
        if step % 50 == 0:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] Step {step}")
        if done:
            obs = env.reset()
            print("  Episode ended, reset environment")
except KeyboardInterrupt:
    print("\n\nInterrupted by user!")

print(f"\n✓ Completed {step} steps")
env.close()
print("Server shut down")
