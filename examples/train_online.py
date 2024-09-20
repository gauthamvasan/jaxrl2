#! /usr/bin/env python
import gymnasium as gym
# import gym
import tqdm, os, pickle
import numpy as np

from absl import app, flags
from ml_collections import config_flags

import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e7), "Number of training steps.")
flags.DEFINE_integer("replay_buffer_size", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Log wandb.")
flags.DEFINE_string("results_dir", "./results", "Save returns.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.replay_buffer_size
    )
    replay_buffer.seed(FLAGS.seed)

    ###### Added by Gautham
    os.makedirs(FLAGS.results_dir, exist_ok=True)
    pkl_fpath = os.path.join(FLAGS.results_dir, "./{}_{}_{}_seed-{}.pkl".format(
        FLAGS.env_name, "sac", "default", FLAGS.seed))
    ###### 

    observation, _, = env.reset()
    done = False
    rets, ep_steps = [], []
    ret, ep_step = 0, 0
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, truncated, info = env.step(action)
        ret += reward; ep_step += 1

        if not done or truncated:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            {
                "observations": observation,
                "actions": action,
                "rewards": reward,
                "masks": mask,
                "dones": done,
                "next_observations": next_observation,
            }
        )
        observation = next_observation

        if done or truncated:
            observation, _, = env.reset()
            done = False
            rets.append(ret)
            ep_steps.append(ep_step)
            ret, ep_step = 0, 0
            # for k, v in info["episode"].items():
            #     decode = {"r": "return", "l": "length", "t": "time"}
            #     wandb.log({f"training/{decode[k]}": v}, step=i)
                

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            # if i % FLAGS.log_interval == 0:
            #     for k, v in update_info.items():
            #         wandb.log({f"training/{k}": v}, step=i)

        if i % 10000 == 0:
            save_pkl(rets, ep_steps, FLAGS, pkl_fpath)

        # if i % FLAGS.eval_interval == 0:
        #     eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
        #     for k, v in eval_info.items():
        #         wandb.log({f"evaluation/{k}": v}, step=i)

    print(f"Seed {FLAGS.seed} ended with mean return {np.mean(rets)} in {sum(ep_steps)} steps.")


def save_pkl(rets, ep_steps, args, pkl_fpath):
    # Save hyper-parameters and config info
    # hyperparams_dict = vars(args)
    pkl_data = {
        # 'args': hyperparams_dict
    }

    ### Saving data
    data = np.zeros((2, len(rets)))
    data[0] = np.array(ep_steps)
    data[1] = np.array(rets)
    pkl_data[args.seed] = {'returns': data, 'N': sum(ep_steps), 'R': np.mean(rets)}

    # Partial save. This should make it easier to resume failed experiments.
    with open(pkl_fpath, "wb") as handle:
        pickle.dump(pkl_data, handle)

if __name__ == "__main__":
    app.run(main)
