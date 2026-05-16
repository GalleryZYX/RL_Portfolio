## 数据文件格式规范

### 原始数据
- 文件名：`stock_data.csv`
- 位置：`data/stock_data.csv`
- 必须包含列：
  - `trade_date`：交易日，格式 YYYYMMDD，升序排列
  - `close_0` ~ `close_4`：5 只股票收盘价，列名严格以 `close_` 为前缀

### 特征数据
- 文件名：`features.csv`
- 位置：`data/features.csv`
- 在原始数据基础上增加技术指标列，命名建议：`ma5_0`, `rsi_0` 等