import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from agent import DDPGAgent

from config import *
from params import args

# https://github.com/ShawnshanksGui/DATE_project/tree/master/DRLTE/drlte
# https://github.com/blackredscarf/pytorch-DDPG
# https://zhuanlan.zhihu.com/p/65931777
# https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py

from trainer import Trainer
from tester import Tester

env = gym.make('Pendulum-v0')
env.reset()
# env.render()

# install env to the running params

configs = {
    'args': args,
    'env': env,
    'gamma': 0.99,
    'actor_lr': 0.001,
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 10000,
    'batch_size': 32,
}

agent = DDPGAgent(**configs)

if args.RUNNING_TYPE == "train":
    trainer = Trainer(agent, env, configs)
    trainer.train()
elif args.RUNNING_TYPE == "retrain":
    episode, step = agent.load_checkpoint(
        args.CHECKPOINT_DIR, args.CHECKPOINT_START_EPISODE)
    trainer = Trainer(agent, env, configs)
    trainer.train(episode, step)
elif args.RUNNING_TYPE == "test":
    tester = Tester(agent, env, './running_log/model')
    tester.test(True)
else:
    print("unknown running type: ", args.RUNNING_TYPE)
env.close()
