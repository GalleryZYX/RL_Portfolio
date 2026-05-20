import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    最小可运行版本：多资产仓位分配环境
    状态：收盘价归一化 + 当前持仓权重
    动作：连续向量，经 Softmax 归一化为持仓权重
    奖励：每日组合对数收益率（已扣除 0.1% 交易成本）
    """
    metadata = {"render_modes": []}

    def __init__(self, df, initial_cash=1_000_000.0, commission=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_stocks = len([c for c in df.columns if c.startswith("close")])
        self.initial_cash = initial_cash
        self.commission = commission

        # 状态维度：每只股票的归一化收盘价 + 当前持仓比例
        obs_dim = self.n_stocks + self.n_stocks
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 动作空间：连续向量，后续在 step 里用 softmax 转成权重
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )

        self.current_step = None
        self.current_weights = None
        self.portfolio_value = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.zeros(self.n_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        """构建状态向量：归一化收盘价 + 当前持仓权重"""
        close_cols = [c for c in self.df.columns if c.startswith("close")]
        prices = self.df.loc[self.current_step, close_cols].values.astype(np.float32)

        # 用过去 20 个交易日均值做归一化（避免绝对价格差异干扰）
        start = max(0, self.current_step - 19)
        mean_price = (
            self.df.loc[start : self.current_step, close_cols]
            .mean()
            .values.astype(np.float32)
        )
        norm_prices = prices / (mean_price + 1e-8)

        obs = np.concatenate([norm_prices, self.current_weights])
        return obs

    def step(self, action):
        # action -> softmax 权重
        exp_a = np.exp(action - np.max(action))
        target_weights = exp_a / (np.sum(exp_a) + 1e-8)

        close_cols = [c for c in self.df.columns if c.startswith("close")]
        prices_t = self.df.loc[self.current_step, close_cols].values.astype(np.float32)
        next_step = self.current_step + 1
        prices_t1 = self.df.loc[next_step, close_cols].values.astype(np.float32)

        # 组合收益 = 权重 · 各股票次日收益率
        returns = prices_t1 / (prices_t + 1e-8) - 1.0
        portfolio_return = float(np.dot(target_weights, returns))

        # 交易成本：按权重变化绝对值之和 × 佣金率
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        cost = turnover * self.commission

        # 总收益
        gross_return = portfolio_return - cost
        self.portfolio_value *= 1.0 + gross_return

        # 奖励：对数收益率（更稳定，避免数值爆炸）
        reward = np.log(1.0 + gross_return + 1e-8)

        # 更新状态
        self.current_weights = target_weights.astype(np.float32)
        self.current_step = next_step

        terminated = bool(self.current_step >= len(self.df) - 1)
        truncated = False

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        info = {
            "portfolio_value": float(self.portfolio_value),
            "gross_return": gross_return,
            "cost": cost,
        }
        return obs, float(reward), terminated, truncated, info
