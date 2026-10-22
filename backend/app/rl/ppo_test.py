import os

import gymnasium as gym
from stable_baselines3 import PPO


ENV_ID = "InvertedPendulum-v5"
MODEL_PATH = "ppo_inverted_pendulum_test"


def train():
    print("=" * 60)
    print("MUJOCO + PPO TRAINING TEST")
    print("=" * 60)

    print(f"Environment: {ENV_ID}")

    env = gym.make(ENV_ID)

    print("Environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu",
    )

    print("\nStarting SHORT PPO training...")
    print("This is only a validation run, not final training.")

    model.learn(total_timesteps=5000)

    print("\nTraining completed.")

    model.save(MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}.zip")

    env.close()


def evaluate():
    print("\n" + "=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)

    model = PPO.load(MODEL_PATH)

    print("Model loaded successfully.")

    env = gym.make(
        ENV_ID,
        render_mode="human",
    )

    print("Evaluation environment created.")

    total_reward = 0.0
    episode_rewards = []

    num_episodes = 3

    for episode in range(num_episodes):
        observation, info = env.reset()

        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _states = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            episode_reward += float(reward)

        episode_rewards.append(episode_reward)
        total_reward += episode_reward

        print(
            f"Episode {episode + 1}: "
            f"Reward = {episode_reward:.2f}"
        )

    average_reward = total_reward / num_episodes

    print("\nEvaluation completed.")
    print(f"Average reward: {average_reward:.2f}")

    env.close()

    print("\nMuJoCo evaluation window closed.")


def main():
    train()

    if os.path.exists(MODEL_PATH + ".zip"):
        evaluate()
    else:
        print("ERROR: Model file was not created.")


if __name__ == "__main__":
    main()