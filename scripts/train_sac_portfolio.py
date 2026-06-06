"""使用 Stable Baselines3 SAC 在 PortfolioEnv 上训练多资产配置智能体。

SAC (Soft Actor-Critic) 是一种 off-policy 最大熵深度强化学习算法，
天然适合连续动作空间。本脚本连接真实 A 股长面板数据，训练一个
51 只股票的多资产动态权重分配智能体。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # scripts/ directory
from envs.portfolio_env import PortfolioEnv
from validation_callback import ValidationCallback, split_dataframe_by_date


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on PortfolioEnv with real A-share stock data."
    )
    # 数据
    parser.add_argument("--train-data", type=Path,
                        default=Path("data/rl_train_data_2023_2025.csv"),
                        help="Path to training CSV (long-panel format).")
    parser.add_argument("--test-data", type=Path,
                        default=Path("data/rl_test_data_2026_now.csv"),
                        help="Path to test CSV for out-of-sample evaluation.")
    # 环境
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0,
                        help="Initial portfolio value.")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Per-side transaction cost rate.")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling window for z-score normalization.")
    # SAC 超参数
    parser.add_argument("--total-timesteps", type=int, default=300_000,
                        help="Total SAC training timesteps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate for all networks.")
    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="Replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size for gradient steps.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient for target networks.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (higher = more exploration).")
    parser.add_argument("--learning-starts", type=int, default=5000,
                        help="Collect this many steps before starting gradient updates.")
    parser.add_argument("--train-freq", type=int, default=1,
                        help="Update the model every N environment steps.")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient steps per training call.")
    parser.add_argument("--net-arch", type=str, default="128,64",
                        help="Comma-separated hidden layer sizes for actor/critic.")
    # 训练流程
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device (cuda:0 / cpu).")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate every N timesteps during training.")
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="Number of evaluation episodes.")
    parser.add_argument("--val-start", type=str, default="2025-01-01",
                        help="Start date for validation split (YYYY-MM-DD).")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop if validation doesn't improve for N consecutive evals.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Directory for saved models and normalizers.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"),
                        help="Directory for TensorBoard logs.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        return "cpu"
    return requested


def make_env(df, initial_cash: float = 1_000_000.0,
             commission: float = 0.001, window: int = 20):
    """返回一个可调用对象，用于 DummyVecEnv。"""
    def _init():
        env = PortfolioEnv(
            df=df,
            initial_cash=initial_cash,
            commission=commission,
            window=window,
        )
        env = Monitor(env)
        return env
    return _init


def compute_metrics(portfolio_values: np.ndarray) -> dict:
    """从 portfolio 净值序列计算常用金融指标。"""
    pv = np.asarray(portfolio_values, dtype=np.float64)
    returns = np.diff(pv) / (pv[:-1] + 1e-8)

    total_return = (pv[-1] / pv[0]) - 1.0
    # 年化夏普 (假设每日数据，年化因子 252)
    ann_factor = np.sqrt(252)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * ann_factor)

    # 最大回撤
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    max_drawdown = float(np.min(drawdown))

    win_rate = float(np.mean(returns > 0))

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "mean_daily_return": float(np.mean(returns)),
        "final_value": float(pv[-1]),
    }


def evaluate_portfolio(model, vec_norm, env_caller, n_episodes: int = 3):
    """用确定性策略跑完整 episode，收集净值曲线并计算金融指标。"""
    # 评估期间冻结 VecNormalize 统计量，避免被评估数据污染
    if vec_norm is not None:
        saved_training = vec_norm.training
        vec_norm.training = False

    all_metrics = []
    for ep in range(n_episodes):
        env = env_caller()
        obs, _ = env.reset()
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs)
        done = False
        portfolio_values = [env.unwrapped.initial_cash]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            portfolio_values.append(info["portfolio_value"])
            if not done and vec_norm is not None:
                obs = vec_norm.normalize_obs(obs)

        metrics = compute_metrics(np.array(portfolio_values))
        metrics["episode"] = ep
        all_metrics.append(metrics)
        env.close()

    # 恢复 VecNormalize 训练状态
    if vec_norm is not None:
        vec_norm.training = saved_training

    # 汇总多 episode 均值
    avg = {}
    for key in all_metrics[0]:
        if key == "episode":
            continue
        avg[key] = float(np.mean([m[key] for m in all_metrics]))
    return avg, all_metrics


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # 解析网络结构
    net_arch = [int(x.strip()) for x in args.net_arch.split(",")]

    # 输出目录
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载数据 ---------------------------------------------------------
    print(f"Loading training data: {args.train_data}")
    df_full = pd.read_csv(args.train_data)
    n_stocks = df_full["ts_code"].nunique()
    n_days = df_full["trade_date"].nunique()
    print(f"  {n_stocks} stocks, {n_days} trading days")

    # ---- 时序切分训练/验证集 -----------------------------------------------
    df_train, df_val = split_dataframe_by_date(df_full, val_start=args.val_start)
    n_train_days = df_train["trade_date"].nunique()
    n_val_days = df_val["trade_date"].nunique()
    print(f"  Train: {n_train_days} days (before {args.val_start})")
    print(f"  Val:   {n_val_days} days (from {args.val_start})")

    # ---- 创建环境 ---------------------------------------------------------
    # 训练环境
    train_env_fn = make_env(df_train,
                            initial_cash=args.initial_cash,
                            commission=args.commission,
                            window=args.window)
    train_vec_env = DummyVecEnv([train_env_fn])
    train_vec_norm = VecNormalize(
        train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0,
        training=True,
    )

    # 验证环境（独立实例，用于 ValidationCallback 早停判断）
    val_env_fn = make_env(df_val,
                           initial_cash=args.initial_cash,
                           commission=args.commission,
                           window=args.window)
    val_vec_env = DummyVecEnv([val_env_fn])
    val_vec_norm = VecNormalize(
        val_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0,
        training=True,
    )

    print(f"Observation space: {train_vec_env.observation_space}")
    print(f"Action space:      {train_vec_env.action_space}")
    print(f"Entropy coef:      {args.ent_coef}")

    # ---- 创建 SAC 模型 ----------------------------------------------------
    model = SAC(
        policy="MlpPolicy",
        env=train_vec_norm,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_kwargs={
            "net_arch": net_arch,
            "optimizer_kwargs": {"weight_decay": 1e-4},
        },
        seed=args.seed,
        device=device,
        tensorboard_log=str(args.log_dir),
        verbose=1,
    )

    print(f"Policy net_arch: {net_arch}")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        idx = torch.device(device).index or 0
        print(f"GPU: {torch.cuda.get_device_name(idx)}")

    # ---- 训练回调（验证集 + 早停）-----------------------------------------
    callback = ValidationCallback(
        val_vec_norm,
        best_model_save_path=str(args.models_dir / "sac_best"),
        log_path=str(args.log_dir / "val"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        patience=args.early_stopping_patience,
        min_delta=0.001,
    )

    # ---- 训练 -------------------------------------------------------------
    print(f"\nStarting training for {args.total_timesteps:,} timesteps ...")
    print(f"Early stopping patience: {args.early_stopping_patience} evaluations")
    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="sac_portfolio",
        callback=callback,
        progress_bar=True,
    )

    # ---- 保存模型 ---------------------------------------------------------
    model_path = args.models_dir / "sac_portfolio"
    norm_path = args.models_dir / "sac_portfolio_vecnorm.pkl"
    model.save(model_path)
    train_vec_norm.save(str(norm_path))
    print(f"Saved model:  {model_path}.zip")
    print(f"Saved norm:   {norm_path}")

    # ---- 样本内评估 -------------------------------------------------------
    print("\n=== In-sample evaluation (training data) ===")
    fresh_env_fn = make_env(df_train,
                            initial_cash=args.initial_cash,
                            commission=args.commission,
                            window=args.window)
    avg_metrics, _ = evaluate_portfolio(model, train_vec_norm, fresh_env_fn, n_episodes=3)
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and abs(v) < 100 else f"  {k}: {v:,.2f}")

    # ---- 样本外评估 -------------------------------------------------------
    if args.test_data.exists():
        print(f"\n=== Out-of-sample evaluation (test data) ===")
        df_test = pd.read_csv(args.test_data)
        n_days_test = df_test["trade_date"].nunique()
        print(f"  {df_test['ts_code'].nunique()} stocks, {n_days_test} trading days")
        test_env_fn = make_env(df_test,
                                initial_cash=args.initial_cash,
                                commission=args.commission,
                                window=args.window)
        N_TEST_EPISODES = 5
        test_avg, test_all = evaluate_portfolio(model, train_vec_norm, test_env_fn,
                                                 n_episodes=N_TEST_EPISODES)
        for k, v in test_avg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) and abs(v) < 100 else f"  {k}: {v:,.2f}")
        # 报告 episode 间标准差
        returns_across = [m["total_return"] for m in test_all]
        print(f"  return_std_({N_TEST_EPISODES}ep): {np.std(returns_across):.4f}")

        # ---- 子期间分析（Q1 vs Q2）----------------------------------------
        df_test_dt = df_test.copy()
        df_test_dt["trade_date"] = pd.to_datetime(df_test_dt["trade_date"])
        q1_cutoff = pd.to_datetime("2026-04-01")
        for period_name, period_df in [
            ("2026 Q1 (Jan-Mar)", df_test_dt[df_test_dt["trade_date"] < q1_cutoff]),
            ("2026 Q2 (Apr-May)", df_test_dt[df_test_dt["trade_date"] >= q1_cutoff]),
        ]:
            if period_df["trade_date"].nunique() == 0:
                continue
            period_fn = make_env(period_df, initial_cash=args.initial_cash,
                                 commission=args.commission, window=args.window)
            period_avg, _ = evaluate_portfolio(model, train_vec_norm, period_fn, n_episodes=1)
            print(f"  [{period_name}] return={period_avg['total_return']:.4f}  "
                  f"sharpe={period_avg['sharpe_ratio']:.4f}  max_dd={period_avg['max_drawdown']:.4f}")
    else:
        print(f"\nTest data not found at {args.test_data}, skipping out-of-sample eval.")

    print(f"\nTensorBoard logs: {args.log_dir.resolve()}")
    train_vec_env.close()
    val_vec_env.close()


if __name__ == "__main__":
    main()
