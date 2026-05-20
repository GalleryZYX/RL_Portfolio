# 数据文件格式规范

## 目录文件清单

```
data/
├── rl_train_data_2023_2025.csv   # 训练集（样本内）
├── rl_test_data_2026_now.csv     # 测试集（样本外）
├── data_collection.py            # 数据采集脚本（Tushare）
├── test_individual_stock.py      # 单股行情测试脚本（AkShare）
├── selected_stocks.txt           # 股票池代码列表
└── README.md                     # 数据处理流程说明文档
```

---

## CSV 数据文件（`rl_train_data_2023_2025.csv` / `rl_test_data_2026_now.csv`）

两个 CSV 文件采用完全相同的列结构，按时间轴切分：

| 文件 | 行数 | 股票数 | 交易日数 | 日期范围 |
|------|------|--------|----------|----------|
| `rl_train_data_2023_2025.csv` | 36,532 | 51 | 717 | 2023-01-17 ~ 2025-12-31 |
| `rl_test_data_2026_now.csv` | 4,376 | 51 | 86 | 2026-01-05 ~ 2026-05-18 |

数据格式为**长面板（Long Panel）**，每行代表单只股票在单个交易日的全部字段切片。

### 列定义（共 21 列）

#### 标识列（2 列）

| 列名 | 类型 | 说明 |
|------|------|------|
| `ts_code` | str | 股票代码，格式 `XXXXXX.SH`（沪）或 `XXXXXX.SZ`（深） |
| `trade_date` | str | 交易日，格式 `YYYYMMDD`，全局升序排列 |

#### 基础量价（9 列，源自 `pro.daily`）

| 列名 | 类型 | 说明 |
|------|------|------|
| `open` | float64 | 开盘价 |
| `high` | float64 | 最高价 |
| `low` | float64 | 最低价 |
| `close` | float64 | 收盘价（也是 `PortfolioEnv` 默认的 `price_col`） |
| `pre_close` | float64 | 前收盘价 |
| `change` | float64 | 涨跌额（`close - pre_close`） |
| `pct_chg` | float64 | 涨跌幅（%） |
| `vol` | float64 | 成交量（手） |
| `amount` | float64 | 成交额（千元） |

#### 估值与流动性（6 列，源自 `pro.daily_basic`）

| 列名 | 类型 | 说明 |
|------|------|------|
| `pe_ttm` | float64 | 滚动市盈率，亏损企业填 0 |
| `pb` | float64 | 市净率 |
| `ps_ttm` | float64 | 滚动市销率 |
| `total_mv` | float64 | 总市值（万元） |
| `circ_mv` | float64 | 流通市值（万元） |
| `turnover_rate` | float64 | 换手率（%） |
| `volume_ratio` | float64 | 量比 |

#### 衍生技术因子（3 列，本地计算）

| 列名 | 类型 | 计算方式 | 说明 |
|------|------|----------|------|
| `momentum_5` | float64 | `close.pct_change(5)` | 5 日动量 |
| `volatility_10` | float64 | `close.pct_change().rolling(10).std()` | 10 日波动率 |
| `macd_dif` | float64 | `EMA(close,12) - EMA(close,26)` | MACD 差离值 |

---

## 股票池（`selected_stocks.txt`）

共 51 只股票，覆盖 7 个板块：

| 板块 | 数量 | 代表标的 |
|------|------|----------|
| 科技与半导体 | 12 | 科大讯飞、海康威视、中芯国际、寒武纪等 |
| 新能源与汽车 | 9 | 宁德时代、比亚迪、隆基绿能等 |
| 消费与医药 | 10 | 贵州茅台、五粮液、恒瑞医药、迈瑞医疗等 |
| 大金融与权重 | 5 | 招商银行、中国平安、中信证券等 |
| 周期、化工与资源 | 6 | 万华化学、紫金矿业、中国石化等 |
| 基建与制造 | 5 | 三一重工、中国中车、万科 A 等 |
| 农林牧渔及其他科创 | 4 | 温氏股份、智飞生物、联影医疗等 |

---

## 数据预处理规则

1. **时序对齐**：每只股票按 `trade_date` 升序排列，禁止时间倒序。
2. **估值缺失填补**：`pe_ttm`、`pb`、`ps_ttm` 中的 NaN 填为 0（对应亏损/无意义估值）。
3. **衍生因子截断**：`momentum_5`、`volatility_10`、`macd_dif` 计算产生的前 N 行 NaN 通过 `dropna` 删除，不影响后续训练。
4. **停牌处理**：由 `PortfolioEnv._prepare_data()` 在加载后执行前向填充。

---

## 数据采集脚本（`data_collection.py`）

- 数据源：**Tushare Pro API**（需 token）
- 调用接口：`pro.daily` + `pro.daily_basic`
- 衍生因子在本地用 pandas 计算（避免未来函数）
- 输出：按时间轴切分为训练集和测试集两个 CSV

## 补充脚本（`test_individual_stock.py`）

- 数据源：**AkShare**
- 用途：手动抽查单只股票某时间段的行情，便于验证数据正确性
- 当前示例：查询 `601208`（东材科技）的日线数据
