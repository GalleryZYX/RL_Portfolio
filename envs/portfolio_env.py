import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """Multi-asset portfolio allocation environment.

    Consumes long-panel stock data and produces a reinforcement learning
    environment where the agent allocates capital across ``N`` stocks.

    Parameters
    ----------
    df : pd.DataFrame
        Long-panel data. Required columns: ``ts_code``, ``trade_date``,
        a price column (default ``close``), and the columns listed in
        *feature_cols*.
    feature_cols : list[str] or None
        Feature columns used in the observation. When ``None``, a sensible
        default set of scale-invariant features is used.
    price_col : str
        Column used for daily return computation (default ``"close"``).
    initial_cash : float
        Starting portfolio value.
    commission : float
        Per-side transaction cost rate (e.g. 0.001 = 0.1 %).
    window : int
        Rolling window size for online z-score normalisation of features.
        Must be >= 2.
    """

    # ------------------------------------------------------------------
    # Scale-invariant defaults — these avoid feeding raw non-stationary
    # prices into the network and stay in reasonable numeric ranges.
    _DEFAULT_FEATURES = [
        # 量价与技术
        "pct_chg",
        "momentum_5",
        "volatility_10",
        "macd_dif",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "natr_14",
        "gap",
        "intraday_amp",
        "close_loc",
        # 价格隐含情感代理（Phase B v2）
        "overnight_gap_abs",
        "intraday_momentum",
    ]

    def __init__(
        self,
        df,
        feature_cols=None,
        price_col="close",
        initial_cash=1_000_000.0,
        commission=0.001,
        window=20,
        drawdown_penalty=0.0,
        concentration_penalty=0.0,
    ):
        super().__init__()

        self.price_col = price_col
        self.initial_cash = initial_cash
        self.commission = commission
        self.window = max(window, 2)
        self.drawdown_penalty = drawdown_penalty
        self.concentration_penalty = concentration_penalty

        # ---- resolve feature columns ---------------------------------
        if feature_cols is None:
            feature_cols = list(self._DEFAULT_FEATURES)
        # Never leak the price column into the state features.
        if self.price_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != self.price_col]
        self.feature_cols = list(feature_cols)

        # ---- build internal data structures --------------------------
        self._prepare_data(df)

        # ---- Gymnasium spaces ----------------------------------------
        obs_dim = self.n_stocks * self.n_features + self.n_stocks
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_stocks,), dtype=np.float32
        )

        # ---- episode state -------------------------------------------
        self.current_step = None
        self.current_weights = None
        self.portfolio_value = None

    # ==================================================================
    # Data preparation
    # ==================================================================

    def _prepare_data(self, df):
        """Pivot long-panel *df* into dense arrays and forward-fill gaps."""
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # ---- 从现有量价列衍生尺度不变特征 --------------------------------
        eps = 1e-8
        if "gap" in self.feature_cols and "open" in df.columns and "pre_close" in df.columns:
            df["gap"] = (df["open"] - df["pre_close"]) / (df["pre_close"] + eps)
        if "intraday_amp" in self.feature_cols and "high" in df.columns and "low" in df.columns:
            df["intraday_amp"] = (df["high"] - df["low"]) / (df["pre_close"] + eps)
        if "close_loc" in self.feature_cols and "close" in df.columns:
            df["close_loc"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + eps)
        if "overnight_gap_abs" in self.feature_cols and "open" in df.columns and "pre_close" in df.columns:
            df["overnight_gap_abs"] = (df["open"] - df["pre_close"]).abs() / (df["pre_close"] + eps)
        if "intraday_momentum" in self.feature_cols and "open" in df.columns and "close" in df.columns:
            df["intraday_momentum"] = (df["close"] - df["open"]) / (df["open"] + eps)

        self.dates = sorted(df["trade_date"].unique())
        self.stocks = sorted(df["ts_code"].unique())
        self.n_stocks = len(self.stocks)
        self.n_features = len(self.feature_cols)

        n_days = len(self.dates)

        # 3-D feature tensor  &  2-D price matrix
        self._features = np.zeros(
            (n_days, self.n_stocks, self.n_features), dtype=np.float32
        )
        self._prices = np.zeros((n_days, self.n_stocks), dtype=np.float32)

        stock_idx = {s: i for i, s in enumerate(self.stocks)}

        for day_idx, date in enumerate(self.dates):
            day = df[df["trade_date"] == date]
            for _, row in day.iterrows():
                si = stock_idx.get(row["ts_code"])
                if si is None:
                    continue
                self._features[day_idx, si] = row[self.feature_cols].values.astype(
                    np.float32
                )
                self._prices[day_idx, si] = float(row[self.price_col])

        # Forward-fill days where a stock was suspended (price == 0)
        for s in range(self.n_stocks):
            for d in range(1, n_days):
                if self._prices[d, s] == 0.0:
                    self._prices[d, s] = self._prices[d - 1, s]
                    self._features[d, s] = self._features[d - 1, s]

        self._n_days = n_days

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.zeros(self.n_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.portfolio_peak = self.initial_cash
        self.portfolio_returns = []  # 用于跟踪近期波动
        return self._get_obs(), {}

    def step(self, action):
        # --- softmax → portfolio weights -------------------------------
        action = np.asarray(action, dtype=np.float32)
        exp_a = np.exp(action - np.max(action))
        target_weights = exp_a / (np.sum(exp_a) + 1e-8)

        # --- daily return ----------------------------------------------
        prices_t = self._prices[self.current_step]
        next_step = self.current_step + 1
        prices_t1 = self._prices[next_step]

        returns = prices_t1 / (prices_t + 1e-8) - 1.0
        portfolio_return = float(np.dot(target_weights, returns))

        # --- transaction cost ------------------------------------------
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        cost = turnover * self.commission

        # --- reward ----------------------------------------------------
        gross_return = portfolio_return - cost
        self.portfolio_value *= 1.0 + gross_return
        base_reward = np.log(1.0 + gross_return + 1e-8)

        # ---- risk penalties (Phase C3) ---------------------------------
        penalty = 0.0

        # 回撤惩罚：组合净值低于历史峰值时惩罚
        if self.drawdown_penalty > 0:
            self.portfolio_peak = max(self.portfolio_peak, self.portfolio_value)
            drawdown = (self.portfolio_peak - self.portfolio_value) / self.portfolio_peak
            penalty += self.drawdown_penalty * drawdown

        # 集中度惩罚：权重越集中（HHI 越高）惩罚越大
        if self.concentration_penalty > 0:
            hhi = float(np.sum(target_weights ** 2))  # 1/N ~ 1
            penalty += self.concentration_penalty * hhi

        reward = base_reward - penalty

        # --- advance state ---------------------------------------------
        self.current_weights = target_weights
        self.current_step = next_step

        terminated = self.current_step >= self._n_days - 1
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

    # ==================================================================
    # Observation helpers
    # ==================================================================

    def _get_obs(self):
        """Return the observation for the current step.

        The observation is the concatenation of:

        * Per-stock features, z-score normalised over a trailing window
          (no look-ahead).
        * Current portfolio weights.
        """
        step = self.current_step
        raw = self._features[step]  # [N_stocks, N_features]

        # Trailing-window statistics
        start = max(0, step - self.window + 1)
        window = self._features[start : step + 1]  # [W, N_stocks, N_features]
        mean = window.mean(axis=0)
        std = window.std(axis=0)

        # Guard against division by zero for constant features.
        std = np.where(std < 1e-8, 1.0, std)
        normed = (raw - mean) / std

        # Clip extreme outliers to keep the observation well-behaved.
        normed = np.clip(normed, -10.0, 10.0)

        obs = np.concatenate([normed.ravel(), self.current_weights])
        return obs.astype(np.float32)
