# JarvisVLA in MineRL 0.3.7 

The project started out forking an older version of [MineRL](https://github.com/minerllabs/minerl) to get human interaction with the agent working, which was broken in the original JarvisVLA's MineStudio implementation. I then noticed JarvisVLA failing in simple tasks like mining oak logs in MineRL 0.3.7's minecraft env, while succeeding in the original JarvisVLA repo's minecraft env.        

## Installing and testing original JarvisVLA  

The embedded JarvisVLA-oak is a fork of [JarvisVLA](https://github.com/CraftJarvis/JarvisVLA) that runs on my machine (Ubuntu 22.04.5 LTS, RTX 3090 Ti with CUDA 13) without conda. Follow the install instructions in the README and run "./run_oak_log_10.sh" to test JarvisVLA in the oak log gathering task for 10 iterations.    

## Installing and testing JarvisVLA in older minerl

```bash
./full_install.sh
```
```bash
python -m venv vllm_env
source vllm_env/bin/activate
pip install -r requirements_vllm.txt
```

## Quickstart

Activate agent venv
```bash
source agent_env/bin/activate
```

In another terminal run the VLLM server (hosted on port 3000 default):

```bash
source vllm_env/bin/activate
./vllm.sh
```

Run the agent on your task (in terminal with agent_env active):

```bash
python agent.py --task <your task prompt> --craft <item id>
```

Where task is the text prompt to be sent to the VLA and craft is the programmatic name of a minecraft id that triggers environment completion when detected in the inventory.

For example to prompt the agent to get oak logs: 

```bash
python agent.py --task "harvest oak logs from the tree" --craft oak_log
```

This will launch minecraft environment and the agent will take actions until max_steps is reached or oak_log is obtained. 

## Interaction

To interact with the agent during inference, open another terminal while agent is still taking actions in environment and do:
```bash
python -m minerl.interactor 6666
```

This will automatically spawn a new minecraft client and connect you to agent's server.

## Notes

JarvisVLA is very brittle. I tried several prompts of the variation "hit tree and get logs," and across all of them success rate is low (<10%). In the success cases, JarvisVLA spawned withe tree trunk in the center of the screen, within or one or two blocks away from hitting range. 

The two main failure modes were tree detection and attack spamming. In the first case, it would begin hitting objects that are not trees (dirt blocks or leaf blocks). In the second case, the agent spent all steps attacking and doing nothing else.         

I suspect the cause of the failures is running the agent in a different minecraft version (1.12.1) and graphic settings relative to its training data. 

So I tested with JarvisVLA official repo:
```bash
./home/minjune/JarvisVLA-oak/run_oak_log_10.sh
```

Results videos in /home/minjune/JarvisVLA-oak/logs/{0-9}

Qualitatively the agent performs MUCH better! 

To exactly replicate environment from JarvisVLA-oak, I copied over its options.txt file while removing newer featuers. 

The primary changes to replicate minestudio was changing MineRL's old FOV (130 Quake Pro mode) to 70 (Minestudio's default), setting gamma (brightness) to 2.0, and doubling particle quality. 

But still seeing large differences in behavior of the agent - making me suspect something else was wrong. Specifically, the agent was getting stuck in loops where it would output just the attack token over and over again. 

Example log: 
```bash
[Step 728] Getting action from agent...
task: Mine the oak log
[VLLM] Calling with 5 messages
[VLLM] Response: <|reserved_special_token_178|><|reserved_special_token_204|><|reserved_special_token_219|><|reserved...
[VLLM] Time: 410.6ms
[VLLM] Extracted 5 special tokens: [151835, 151861, 151876, 151897, 151836]...
wall clock time: 458.30
  Action: forward=0, jump=0, attack=1, camera=[0. 0.]
```

I rechecked special token -> minerl action mapping to confirm its correct + message formatting being sent to VLLM. Concluding that it is indeed out of distribution brittleness. 

## Conclusion

This was my first venture into getting a "VLA" model to work, and it immediately ran into brittleness issues with small environment changes. While I can't be certain exactly what the cause is (would require ablations in working MineStudio env to replicate behavior of MineRL, frankly don't have time to do this), I got a taste of how subtle shifts in environment distribution can break models that are very strong on paper.    

## Modifications from MineRL

1. **spaces.py** - Removed `self.shape = ()`
2. **core.py** - Fixed `collections.Mapping` → `collections.abc.Mapping`
3. **observables.py** - Fixed `np.int` → `int`
4. **MalmoEnvServer.java** - Added UUID generation (50+ lines)
5. **build.gradle** - Configured for local MixinGradle
