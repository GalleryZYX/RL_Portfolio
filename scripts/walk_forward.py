"""Walk-Forward 滚动窗口回测框架。

在多个时间窗口上独立训练+评估，汇总统计 OOS 表现，
提供比单次测试更可信的模型评估。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.portfolio_env import PortfolioEnv
from scripts.train_sac_portfolio import main as train_main, parse_args as train_parse_args
from scripts.validation_callback import split_dataframe_by_date
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def make_env_for_eval(df, initial_cash, commission, window):
    """创建评估用环境。"""
    def _init():
        env = PortfolioEnv(
            df=df,
            initial_cash=initial_cash,
            commission=commission,
            window=window,
        )
        return Monitor(env)
    return _init


def compute_metrics(portfolio_values: np.ndarray) -> dict:
    """计算常用金融指标。"""
    pv = np.asarray(portfolio_values, dtype=np.float64)
    returns = np.diff(pv) / (pv[:-1] + 1e-8)
    total_return = (pv[-1] / pv[0]) - 1.0
    ann_factor = np.sqrt(252)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * ann_factor)
    peak = np.maximum.accumulate(pv)
    max_drawdown = float(np.min((pv - peak) / peak))
    win_rate = float(np.mean(returns > 0))
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "final_value": float(pv[-1]),
    }


def run_window(
    df_full: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    args: argparse.Namespace,
    window_id: int,
) -> dict:
    """在一个时间窗口上：训练 + 验证早停 + 测试评估。

    训练期间用 80/20 时序切分做训练/验证早停。
    """

    print(f"\n{'='*60}")
    print(f"Window {window_id}: Train {train_start}~{train_end}, Test {test_start}~{test_end}")
    print(f"{'='*60}")

    df_full["trade_date"] = pd.to_datetime(df_full["trade_date"])

    # 训练期数据
    train_mask = (df_full["trade_date"] >= train_start) & (df_full["trade_date"] <= train_end)
    df_train_full = df_full[train_mask]

    # 训练期内按时序切分 train/val（80/20）
    dates_sorted = sorted(df_train_full["trade_date"].unique())
    split_idx = int(len(dates_sorted) * 0.8)
    val_start_date = dates_sorted[split_idx].strftime("%Y-%m-%d")

    df_train, df_val = split_dataframe_by_date(df_train_full, val_start=val_start_date)

    # 确保 train/val 使用相同的股票池（有些股票可能在早期尚未上市）
    train_stocks = set(df_train["ts_code"].unique())
    df_val = df_val[df_val["ts_code"].isin(train_stocks)]

    n_train_days = df_train["trade_date"].nunique()
    n_val_days = df_val["trade_date"].nunique()
    n_train_stocks = len(train_stocks)
    print(f"  Train: {n_train_days} days ({n_train_stocks} stocks), Val: {n_val_days} days")

    # 测试期数据（只用训练集中存在的股票）
    test_mask = (df_full["trade_date"] >= test_start) & (df_full["trade_date"] <= test_end)
    df_test = df_full[test_mask]
    df_test = df_test[df_test["ts_code"].isin(train_stocks)]
    n_test_days = df_test["trade_date"].nunique()
    print(f"  Test:  {n_test_days} days")

    # ---- 训练环境 ----
    train_env_fn = make_env_for_eval(
        df_train,
        initial_cash=args.initial_cash,
        commission=args.commission,
        window=args.window,
    )
    train_vec_env = DummyVecEnv([train_env_fn])
    train_vec_norm = VecNormalize(
        train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True,
    )

    # ---- 验证环境 ----
    val_env_fn = make_env_for_eval(
        df_val,
        initial_cash=args.initial_cash,
        commission=args.commission,
        window=args.window,
    )
    val_vec_env = DummyVecEnv([val_env_fn])
    val_vec_norm = VecNormalize(
        val_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True,
    )

    print(f"  Obs: {train_vec_env.observation_space.shape}")

    # ---- SAC 模型 ----
    net_arch = [int(x.strip()) for x in args.net_arch.split(",")]
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
        seed=args.seed + window_id,
        device=args.device,
        verbose=0,
    )

    # ---- 训练（使用 ValidationCallback 早停）----
    from scripts.validation_callback import ValidationCallback

    callback = ValidationCallback(
        val_vec_norm,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        patience=args.early_stopping_patience,
        verbose=0,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    best_val = callback.best_val_reward
    best_step = callback.best_step
    stopped_early = callback._stopped_early

    # ---- 测试评估（无惩罚）----
    test_env_fn = make_env_for_eval(
        df_test,
        initial_cash=args.initial_cash,
        commission=args.commission,
        window=args.window,
    )
    test_env = test_env_fn()
    obs, _ = test_env.reset()
    obs = train_vec_norm.normalize_obs(obs)
    done = False
    pv = [args.initial_cash]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        pv.append(info["portfolio_value"])
        if not done:
            obs = train_vec_norm.normalize_obs(obs)

    metrics = compute_metrics(np.array(pv))
    metrics["window"] = window_id
    metrics["train_days"] = n_train_days
    metrics["test_days"] = n_test_days
    metrics["best_val_reward"] = best_val
    metrics["best_step"] = best_step
    metrics["stopped_early"] = stopped_early

    test_env.close()
    train_vec_env.close()
    val_vec_env.close()

    print(f"  Result: return={metrics['total_return']:.4f}, "
          f"sharpe={metrics['sharpe_ratio']:.4f}, "
          f"max_dd={metrics['max_drawdown']:.4f}")

    return metrics


def run_walk_forward(
    csv_path: str,
    windows: list[tuple[str, str, str, str]],
    base_args: argparse.Namespace,
) -> pd.DataFrame:
    """在多个窗口上运行 Walk-Forward 回测。

    Parameters
    ----------
    csv_path : str
        完整历史数据 CSV 路径。
    windows : list of (train_start, train_end, test_start, test_end)
        各窗口的时间范围。
    base_args : argparse.Namespace
        训练参数。

    Returns
    -------
    pd.DataFrame
        每个窗口的评估指标。
    """
    df_full = pd.read_csv(csv_path)
    all_results = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        result = run_window(df_full, tr_s, tr_e, te_s, te_e, base_args, window_id=i + 1)
        all_results.append(result)

    # 汇总
    results_df = pd.DataFrame(all_results)
    print(f"\n{'='*60}")
    print("Walk-Forward 汇总结果")
    print(f"{'='*60}")
    print(results_df[["window", "train_days", "test_days",
                       "total_return", "sharpe_ratio", "max_drawdown"]].to_string(index=False))
    print(f"\n均值: return={results_df['total_return'].mean():.4f} "
          f"± {results_df['total_return'].std():.4f}, "
          f"sharpe={results_df['sharpe_ratio'].mean():.4f} "
          f"± {results_df['sharpe_ratio'].std():.4f}")
    print(f"胜率: {(results_df['total_return'] > 0).mean():.0%} 窗口正收益")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 滚动窗口回测")
    parser.add_argument("--data", type=str, default="data/rl_full_data_2020_2026.csv",
                        help="完整历史数据 CSV")
    parser.add_argument("--window-years", type=int, default=2,
                        help="每窗口训练年数")
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 构建训练参数
    class TrainArgs:
        pass

    train_args = TrainArgs()
    train_args.initial_cash = 1_000_000.0
    train_args.commission = 0.001
    train_args.window = 20
    train_args.total_timesteps = args.total_timesteps
    train_args.learning_rate = 3e-4
    train_args.buffer_size = 100_000
    train_args.batch_size = 128
    train_args.gamma = 0.99
    train_args.tau = 0.005
    train_args.ent_coef = 0.01
    train_args.learning_starts = 5000
    train_args.train_freq = 1
    train_args.gradient_steps = 1
    train_args.net_arch = "128,64"
    train_args.seed = 42
    train_args.device = args.device
    train_args.eval_freq = 10_000
    train_args.eval_episodes = 3
    train_args.early_stopping_patience = 10

    # 定义时间窗口（年度滚动）
    df = pd.read_csv(args.data)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    all_dates = sorted(df["trade_date"].unique())
    min_year = all_dates[0].year
    max_year = all_dates[-1].year

    windows = []
    wy = args.window_years
    for test_year in range(min_year + wy, max_year + 1):
        train_start = f"{test_year - wy}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"
        windows.append((train_start, train_end, test_start, test_end))

    print(f"数据范围: {min_year}~{max_year}, 窗口年数: {wy}")
    print(f"窗口数: {len(windows)}")
    for i, w in enumerate(windows):
        print(f"  W{i+1}: Train {w[0]}~{w[1]} → Test {w[2]}~{w[3]}")

    run_walk_forward(args.data, windows, train_args)


if __name__ == "__main__":
    main()
