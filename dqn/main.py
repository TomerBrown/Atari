import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
from argparse import ArgumentParser

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='unnamed', help='Run name (appears in output file)')
    parser.add_argument('--action_selection', type=str, default='epsilon greedy', help='epsilon greedy / softmax')
    parser.add_argument('--beta_func', type=str, default='log', help='loglog / log / root')
    parser.add_argument('--beta_pow', type=float, default=0.07, help='Power for root beta func')
    parser.add_argument('--optim', type=str, default=None, help='None for RMSProp, or "Adam" for adam')
    return parser.parse_args()

def main(env, num_timesteps, args):
    
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )
    if args.optim == 'Adam':
        optimizer_spec = OptimizerSpec(
            constructor=optim.Adam,
            kwargs=dict(lr=1e-5),
        )

    exploration_schedule = LinearSchedule(1000000, 0.1)
    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        selection_method=args.action_selection, 
        run_name=args.run_name,
        beta_func=args.beta_func,
        beta_pow=args.beta_pow,
    )

if __name__ == '__main__':
    args = get_args()
    print(args)
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed, args.run_name)

    main(env, task.max_timesteps, args)
