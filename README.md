# Ensemble-based Offline Reinforcement Learning with Adaptive Behavior Cloning

Author's PyTorch implementation of EABC. EABC is an ensemble based offline RL algorithm built upon TD3+BC, with adjustable behavior cloning term. It achieves more stable and better performance on D4RL MuJoCo benchmarks. The implementation is built on top of official repository of [TD3+BC](https://github.com/sfujim/TD3_BC.git).

## Usage
The code is implemented with Python 3.9, [PyTorch 2.1.1](https://pytorch.org/), [MuJoCo 3.0.1](http://www.mujoco.org/) ([mujoco-py 2.1](https://github.com/openai/mujoco-py)), [OpenAI gym 0.23.1](https://github.com/openai/gym), and the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Packages `gym` and `numpy`, are also needed (any version should work). 

Results can be reproduced by running:
```
python main.py
```
