import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from envs.portfolio_env_test import PortfolioEnv


def generate_fake_data(n_days=500, n_stocks=5, seed=42):
    """生成模拟日线数据：随机游走"""
    rng = np.random.RandomState(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.randn(n_days, n_stocks) * 0.02, axis=0))
    cols = [f"close_{i}" for i in range(n_stocks)]
    df = pd.DataFrame(prices, columns=cols)
    return df


if __name__ == "__main__":
    df = generate_fake_data(n_days=500, n_stocks=5)
    env = PortfolioEnv(df)

    obs, info = env.reset(seed=42)
    done = False
    step_count = 0

    while not done:
        # 随机动作：从动作空间里随机采样
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    print(f"总步数: {step_count}")
    print(f"最终组合净值: {info['portfolio_value']:,.2f}")
    print(f"初始资金: {env.initial_cash:,.2f}")
    print("环境运行正常，无报错。")
