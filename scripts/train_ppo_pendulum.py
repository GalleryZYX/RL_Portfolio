"""Train a Stable Baselines3 PPO agent on Pendulum-v1.

This script is a small reproducibility smoke test for the course project:
it verifies that PyTorch, CUDA, Gymnasium, Stable Baselines3, model saving,
evaluation, and TensorBoard logging all work before we connect SB3 to the
custom portfolio environment.
"""

from __future__ import annotations
# 用于命令行参数和路径管理
import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy # 用来训练后评估模型平均奖励
from stable_baselines3.common.monitor import Monitor # 用来包装环境，记录 episode 长度、episode reward 等日志


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on Gymnasium Pendulum-v1 with Stable Baselines3."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20_000,
        help="Number of PPO training timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes after training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible smoke tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device used by SB3. Use cuda:0 to target GPU 0.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory for saved SB3 models.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for TensorBoard logs.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    """Fall back to CPU with a clear message if CUDA is unavailable."""
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        return "cpu"
    return requested_device


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # 创建输出目录
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # 创建 Gymnasium 环境
    env = Monitor(gym.make("Pendulum-v1"))

    print(f"Using device: {device}")
    if device.startswith("cuda"):
        device_index = torch.device(device).index or 0
        print(f"CUDA device name: {torch.cuda.get_device_name(device_index)}")

    # 创建 PPO 模型
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed, # 固定随机种子
        device=device,
        tensorboard_log=str(args.log_dir), # 告诉 SB3 把 TensorBoard 日志写到 runs/
    )

    # 开始训练
    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="ppo_pendulum",
        progress_bar=True,
    )

    model_path = args.models_dir / "ppo_pendulum"
    model.save(model_path)
    print(f"Saved model to: {model_path}.zip")

    # 验证：模型保存以后，能不能重新加载并继续用于推理/评估
    # 后面接自己的交易环境时，训练好的模型也会用类似方式加载：
    # model = PPO.load("models/xxx", env=env)
    loaded_model = PPO.load(model_path, env=env, device=device)

    # 评估模型
    mean_reward, std_reward = evaluate_policy(
        loaded_model,
        env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True, # 评估时使用确定性动作，不再随机采样动作
    )

    # 输出平均奖励、标准差和 TensorBoard 日志路径
    print(f"Mean reward over {args.eval_episodes} episodes: {mean_reward:.2f}")
    print(f"Reward std: {std_reward:.2f}")
    print(f"TensorBoard logs: {args.log_dir.resolve()}")

    # 释放环境资源
    env.close()


if __name__ == "__main__":
    main()
