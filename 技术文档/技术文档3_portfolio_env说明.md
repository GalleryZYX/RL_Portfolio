# PortfolioEnv 详细说明
**更新日期：** 2026-05-20  
**编写人：** 胡锦宏  
**内容：** PortfolioEnv 详细说明
## 概述

`PortfolioEnv` 是一个基于 Gymnasium 标准接口的**多资产投资组合配置强化学习环境**。智能体在每个交易日观察市场特征，决定资金在 N 只股票间的分配权重，目标是最大化对数收益率的累计和。

环境消费 Tushare 导出的**长面板数据（Long Panel Data）**，内部将其转换为稠密的 3D 特征张量和 2D 价格矩阵。

---

## 数据格式要求

输入的 `df`（`pd.DataFrame`）必须包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `ts_code` | str | 股票代码，如 `600519.SH` |
| `trade_date` | str/int | 交易日期，如 `20230104`（环境中会转为 datetime） |
| `close`（或自定义 `price_col`） | float | 用于计算日收益率的收盘价 |
| 各特征列 | float | 构成状态空间的因子，详见下方特征列表 |

数据行之间日期允许有间隔（非交易日会被自动跳过，时间索引只取唯一日期），但 `trade_date` 必须整体升序。

---

## 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | `pd.DataFrame` | 必填 | 长面板股票数据 |
| `feature_cols` | `list[str]` 或 `None` | `None` | 特征列名。`None` 时使用下方默认特征集 |
| `price_col` | `str` | `"close"` | 价格列名，用于计算日收益率 |
| `initial_cash` | `float` | `1,000,000.0` | 初始组合净值 |
| `commission` | `float` | `0.001` | 单边交易成本率（0.1%） |
| `window` | `int` | `20` | 滚动窗口大小，用于特征 z-score 归一化（最小为 2） |

### 默认特征因子（`feature_cols=None`）

| 因子 | 说明 | 类别 |
|------|------|------|
| `pct_chg` | 当日涨跌幅 (%) | 量价 |
| `momentum_5` | 5 日收益率 | 动量 |
| `volatility_10` | 10 日波动率（收盘价收益率滚动标准差） | 风险 |
| `macd_dif` | MACD 差离值（12 日 EMA - 26 日 EMA） | 趋势 |
| `turnover_rate` | 换手率 (%) | 流动性 |
| `volume_ratio` | 量比 | 流动性 |
| `pe_ttm` | 滚动市盈率 | 估值 |
| `pb` | 市净率 | 估值 |

这些特征均为**尺度不变（scale-invariant）**指标（比率、百分比、去趋势值），避免了绝对价格输入网络造成的数值不稳定。

---

## Gymnasium 空间

### 观测空间（Observation Space）

```
Box(low=-inf, high=inf, shape=(N_stocks × N_features + N_stocks,), float32)
```

观测向量由两部分拼接而成：

1. **特征段**（`N_stocks × N_features` 维）：每只股票各特征的 z-score 归一化值，扁平排列。归一化使用**当前步之前最多 `window` 个交易日**的均值和标准差（无未来信息泄露）。
2. **权重段**（`N_stocks` 维）：当前各股票的持仓权重，和为 1。

以默认配置（51 只股票 × 8 特征）为例，观测维度 = 51×8 + 51 = **459 维**。

z-score 归一化后，特征值被裁剪到 `[-10, +10]` 区间，防止极端值破坏训练稳定性。

### 动作空间（Action Space）

```
Box(low=-1.0, high=1.0, shape=(N_stocks,), float32)
```

连续动作向量，每个元素在 `[-1, 1]` 之间。在 `step()` 中通过 **softmax** 转换为和为 1 的持仓权重：

```
target_weights = softmax(action)
```

---

## 奖励函数

每一步的奖励公式：

```
daily_return     = Σ(target_weights_i × (price_{t+1,i} / price_{t,i} - 1))
turnover         = Σ|target_weights_i - current_weights_i|
cost             = turnover × commission
gross_return     = daily_return - cost
reward           = log(1 + gross_return + ε)
```

设计要点：
- **对数收益率**：比直接收益率更稳定，避免数值爆炸；且具有可加性（累积奖励 = 对数总收益）。
- **交易成本**：与权重变化量成正比，惩罚频繁换仓。
- ε = 1e-8 防止 `log(0)`。

---

## 数据预处理（`_prepare_data` 内部逻辑）

1. **长面板 → 稠密张量**：按 `trade_date` 和 `ts_code` 交叉索引，将长面板转为 `(n_days, n_stocks, n_features)` 的特征张量和 `(n_days, n_stocks)` 的价格矩阵。
2. **前向填充停牌**：若某股票在某日的价格为 0（停牌），则用前一日的数据填充价格和特征，保持矩阵完整性。
3. **日期排序**：`trade_date` 强制升序，确保 MDP 时间箭头正确。

---

## MDP 生命周期

```
reset() → (obs, {})
    ↓
step(action) → (next_obs, reward, terminated, truncated, info)
    ↓
... (重复直到 terminated=True)
```

- **`reset(seed, options)`**：回到第 0 个交易日，清空持仓，返回初始观测。
- **`step(action)`**：执行一次交易日的组合调整，返回新的观测和奖励。当到达最后一天时 `terminated=True`。
- **`info`** 字典包含 `portfolio_value`（当前净值）、`gross_return`（扣除成本后收益率）、`cost`（交易成本）。

---

## 使用示例

```python
import pandas as pd
from envs.portfolio_env import PortfolioEnv

# 加载真实数据
df = pd.read_csv("data/rl_train_data_2023_2025.csv")

# 创建环境
env = PortfolioEnv(
    df,
    feature_cols=None,      # 使用默认 8 因子
    price_col="close",
    initial_cash=1_000_000.0,
    commission=0.001,
    window=20,
)

# 运行一个随机策略 episode
obs, info = env.reset(seed=42)
done = False
while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"最终净值: {info['portfolio_value']:,.2f}")
```

### 接入 Stable Baselines3

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    df = pd.read_csv("data/rl_train_data_2023_2025.csv")
    return PortfolioEnv(df)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./runs/")
model.learn(total_timesteps=500_000)
```

---

## 与原始版本的区别

| 方面 | 原始版本（已提交） | 当前版本（本地修改） |
|------|-------------------|---------------------|
| 数据格式 | 宽表 `close_0`~`close_4` | 长面板 `ts_code` + `trade_date` |
| 股票数 | 固定 5 只假数据 | 任意只真实股票（当前 51 只） |
| 状态特征 | 收盘价简单归一化 | 8 个量化因子 + z-score 滚动归一化 |
| 停牌处理 | 无 | 前向填充 |
| 代码结构 | 平铺 | 分段注释，`_prepare_data` / API / helpers |
