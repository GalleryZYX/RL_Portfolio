"""自定义验证回调：时序切分验证集 + 早停机制。

与 EvalCallback 不同，此回调评估在验证集上进行，
当验证奖励连续 N 次不提升时触发早停，并自动保存验证表现最佳的模型。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class ValidationCallback(EvalCallback):
    """在独立验证集上评估，支持早停和最佳模型保存。

    Parameters
    ----------
    eval_env : VecEnv
        验证环境（应与训练环境使用独立的 VecNormalize）。
    best_model_save_path : str or Path, optional
        验证最佳模型的保存目录。
    log_path : str or Path, optional
        评估日志保存目录。
    eval_freq : int
        每多少步评估一次。
    n_eval_episodes : int
        每次评估的 episode 数。
    deterministic : bool
        是否使用确定性策略。
    patience : int
        允许验证奖励不提升的最大评估次数，超过后触发早停。
    min_delta : float
        判定为"提升"的最小奖励增量。
    verbose : int
        日志详细程度。
    """

    def __init__(
        self,
        eval_env: VecEnv,
        best_model_save_path: str | Path | None = None,
        log_path: str | Path | None = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 3,
        deterministic: bool = True,
        patience: int = 10,
        min_delta: float = 0.001,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=str(best_model_save_path) if best_model_save_path else None,
            log_path=str(log_path) if log_path else None,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.best_val_reward = -np.inf
        self.evaluations_without_improvement = 0
        self.best_step = 0
        self._stopped_early = False

    def _on_step(self) -> bool:
        """每步检查是否需要评估，并在评估后判断早停条件。"""
        result = super()._on_step()
        # EvalCallback 的 _on_step 会在 eval 后更新 self.best_mean_reward
        # 我们基于验证集上的 last_mean_reward 来判断

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # last_mean_reward 是最近一次 eval 的平均 reward
            current_reward = self.last_mean_reward

            if current_reward > self.best_val_reward + self.min_delta:
                self.best_val_reward = current_reward
                self.best_step = self.n_calls
                self.evaluations_without_improvement = 0
                if self.verbose > 0:
                    print(f"  [Val] New best: {current_reward:.4f} @ step {self.n_calls}")
            else:
                self.evaluations_without_improvement += 1
                if self.verbose > 0:
                    print(
                        f"  [Val] No improvement: {current_reward:.4f} "
                        f"(best: {self.best_val_reward:.4f}, "
                        f"patience: {self.evaluations_without_improvement}/{self.patience})"
                    )

            # 早停判断
            if self.evaluations_without_improvement >= self.patience:
                print(
                    f"\n  Early stopping triggered at step {self.n_calls}: "
                    f"no improvement for {self.patience} consecutive evaluations.\n"
                    f"  Best validation reward: {self.best_val_reward:.4f} @ step {self.best_step}"
                )
                self._stopped_early = True
                return False  # 返回 False 通知 SB3 停止训练

        return result


def split_dataframe_by_date(
    df: "pd.DataFrame",
    val_start: str = "2025-01-01",
    date_col: str = "trade_date",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """按时序切分 DataFrame。

    Parameters
    ----------
    df : pd.DataFrame
        完整的长面板数据。
    val_start : str
        验证集起始日期（含）。此前为训练集，此后为验证集。
    date_col : str
        日期列名。

    Returns
    -------
    train_df, val_df : tuple of pd.DataFrame
    """
    import pandas as pd

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    val_start_dt = pd.to_datetime(val_start)

    train_df = df[df[date_col] < val_start_dt]
    val_df = df[df[date_col] >= val_start_dt]

    return train_df, val_df
