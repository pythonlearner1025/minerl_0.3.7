#!/usr/bin/env python3
"""
MineRL Interactive Client Test
Run this AFTER the server is ready on port 6666
"""
import sys
import time
import logging
import os
import tempfile
from minerl.env.malmo import InstanceManager, malmo_version
from minerl.env.core import MineRLEnv
from minerl.env import comms
import socket
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 6666
INTERACTOR_PORT = 31415

print("=" * 70)
print("MINERL INTERACTIVE CLIENT (WITH UUID FIX)")
print("=" * 70)
print(f"\nTarget server: 127.0.0.1:{SERVER_PORT}")
print(f"Interactor port: {INTERACTOR_PORT}")
print("=" * 70)

# Check for existing interactor
print("\n[1/3] Checking for existing interactor...")
try:
    InstanceManager.add_existing_instance(INTERACTOR_PORT)
    instance = InstanceManager.get_instance(-1)
    print(f"✓ Found existing interactor on port {INTERACTOR_PORT}")
except AssertionError:
    print("✗ No existing interactor found")
    print("\nERROR: You need to launch a Minecraft instance manually on port", INTERACTOR_PORT)
    print("Run this command in another terminal first:")
    print(f"  python -c \"import minerl; minerl.env.malmo.InstanceManager.allocate_pool([{INTERACTOR_PORT}])\"")
    sys.exit(1)

# Connect to server
print(f"\n[3/3] Connecting to server at 127.0.0.1:{SERVER_PORT}...")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
sock.settimeout(60)

try:
    sock.connect(('localhost', INTERACTOR_PORT))
    print("  - Socket connected to interactor")

    MineRLEnv._hello(sock)
    print("  - Sent hello message")

    comms.send_message(sock, f"<Interact>127.0.0.1:{SERVER_PORT}</Interact>".encode())
    print("  - Sent interact command")

    reply = comms.recv_message(sock)
    if reply is None:
        print("✗ No response from interactor - is the server running?")
        sock.close()
        sys.exit(1)

    ok, = struct.unpack('!I', reply)
    sock.close()

    if ok:
        print("✓ Connection successful!")
        print("\n" + "=" * 70)
        print("✓ MINECRAFT WINDOW SHOULD BE JOINING THE SERVER")
        print("=" * 70)
        print("\nCheck the Minecraft window - you should see the agent's world!")
        print("\nLook for these messages in ~/.malmo/logs/*.log:")
        print("  - '[MineRL UUID Fix] Generated UUID for Player...'")
        print("  - 'GameProfile@...id=<VALID_UUID>,name=Player...' (NOT <null>!)")
        print("  - 'Player... joined the game'")
        print("\nPress Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print("✗ Connection failed - server returned error")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error connecting: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
