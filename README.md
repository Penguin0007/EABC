# Ensemble-based Offline Reinforcement Learning with Adaptive Behavior Cloning

Official PyTorch implementation of EABC from authors. EABC is an ensemble based offline actor critic algorithm built upon TD3+BC, with adjustable behavior cloning term. It achieves more stable and better performance. The implementation is based on official repository of [TD3+BC](https://github.com/sfujim/TD3_BC.git).

## Usage
The code is implemented with Python 3.9, [PyTorch 2.1.1](https://pytorch.org/), [MuJoCo 2.1](http://www.mujoco.org/) (and [mujoco-py 2.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym), with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Mujoco license is also required in order to run the D4RL experiments. Packages `gym` and `numpy`, are also needed (any version should work). 

Results can be reproduced by running on terminal:
```
./run_experiments.sh
```
