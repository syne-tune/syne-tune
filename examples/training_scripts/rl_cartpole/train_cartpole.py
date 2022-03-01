"""
Adapts the introductionary example of rllib that trains a Cartpole with PPO.
https://docs.ray.io/en/master/rllib/index.html
The input arguments learning-rate and gamma discount factor can be tuned for maximizing the episode mean reward.
"""
from argparse import ArgumentParser
from syne_tune import Reporter
from ray.rllib.agents.ppo import PPOTrainer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_training_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    args, _ = parser.parse_known_args()

    # Configure the algorithm.
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "CartPole-v0",
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "tf",
        "gamma": args.gamma,
        "lr": args.lr,
    }

    trainer = PPOTrainer(config=config)

    reporter = Reporter()
    # Run it for n max_training_steps iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    # Episode reward mean is reported each time.
    for i in range(args.max_training_steps):
        results = trainer.train()
        reporter(
            training_iter=i+1,
            episode_reward_mean=results['episode_reward_mean'],
        )