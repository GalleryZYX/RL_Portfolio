# Task Plan: SAC 投资组合模型改进

## Goal
在初次 SAC 训练完成的基础上，逐步改进模型以解决严重过拟合问题（训练 178x vs 测试 -0.29%），并纳入新闻/舆情特征，最终达到样本外正收益、夏普 > 0.5。

## Current Phase
Phase A v2 — 折中参数调优（抗欠拟合）

## Phases

### Phase 0: SAC 初次训练（已完成）
- [x] `train_sac_portfolio.py` 编写与修复（训练/评估环境隔离、VecNormalize 冻结）
- [x] 100K 步训练完成
- [x] 训练集评估：累计对数收益 5.185（≈178 倍）
- [x] 测试集评估：-0.29%，夏普 0.05，最大回撤 -12.81%
- [x] 输出文件：模型、日志、技术文档4、训练总结
- **Status:** complete

### Phase A: 抗过拟合（已完成）
- [x] A1: 缩小网络 `[256,256]` → `[64,32]`
- [x] A2: 固定熵系数 `auto` → `0.05`
- [x] A3: 添加 L2 权重衰减 `weight_decay=1e-4`
- [x] A4: 梯度裁剪（`max_grad_norm` 被 SB3 SAC 拒绝，已移除）
- [x] A5: 时序验证集 + 早停回调（训练 2023-2024，验证 2025）
- [x] A6: 特征精简（去掉 vol/amount，新增 gap/intraday_amp/close_loc）
- [x] A7: 样本外评估改进（5 episode + Q1/Q2 子期间拆分）
- [x] A8: 超参数适配：lr `1e-3`，batch `128`，steps `300K`，learning_starts `5000`
- **Status:** complete
- **结果:** 测试集 +0.66%，夏普 0.20，过拟合消除。但 [64,32] 网络欠拟合（训练收益 -2.37%）

### Phase A v2: 折中参数调优
- [ ] 网络 `[64,32]` → `[128,64]`（恢复适度容量）
- [ ] 熵系数 `0.05` → `0.01`（允许收敛）
- [ ] 学习率 `1e-3` → `3e-4`（更稳定）
- [ ] 保持 steps 300K、验证早停、特征集不变
- [ ] 修复早停 callback 中 `model.stop_training` 未被检查的 bug
- **Status:** pending

### Phase B: 新闻/舆情特征
- [ ] B1: 测试数据源 API（AKShare 热门排名、Tushare 新闻 + SnowNLP）
- [ ] B2: 编写 `data/sentiment_pipeline.py`
- [ ] B3: 集成到 `data/data_collection.py`，重新导出训练/测试 CSV
- [ ] B4: 更新 `PortfolioEnv._DEFAULT_FEATURES`
- [ ] B5: 重新训练并评估
- **Status:** pending

### Phase C: 方法论升级
- [ ] C1: 自定义 Dropout 策略网络
- [ ] C2: Walk-Forward 滚动窗口回测框架
- [ ] C3: 风险感知奖励函数（回撤惩罚、集中度惩罚）
- [ ] C4: Optuna 超参数自动搜索
- [ ] C5: Agent 集成（5 种子取均值）
- **Status:** pending

## Key Questions
1. ~~缩小网络 + 正则化能否将 OOS 从 -0.29% 提升到正收益？~~ **→ 已答：是，+0.66%，但欠拟合**
2. [64,32]→[128,64] + ent_coef 0.05→0.01 能否在防止过拟合的同时恢复学习能力？
3. 新闻情感特征是否比纯量价特征提供额外信息？
4. 哪个风险惩罚项（回撤 vs 集中度）对泛化最有帮助？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 选 SAC 而非 PPO | Off-policy 样本效率高，最大熵天然鼓励持仓多样化 |
| [256,256] → [64,32] | Phase A: 参数从 70 万降至 4 万，成功消除过拟合但导致欠拟合 |
| [64,32] → [128,64]（Phase A v2） | 折中容量，~10 万参数，平衡欠拟合与过拟合 |
| 固定 ent_coef=0.05→0.01 | 0.05 过高导致训练无法收敛，0.01 适度降低探索 |
| 时序验证集切分 | 金融数据必须按时间切分，避免未来信息泄露 |
| 去掉 vol/amount | 绝对量纲噪声大，z-score 无法消除跨股票量级差异 |
| 新增 gap/intraday_amp/close_loc | 从已有未用价格列衍生，全为尺度不变比率 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| SAC 100K 训练严重过拟合（178x vs -0.29%） | 1 | Phase A 消除过拟合成功 |
| Phase A 欠拟合（训练 -2.37%） | 1 | Phase A v2: 增大网络、降低熵系数 |
| `max_grad_norm` 不被 SB3 SAC 接受 | 1 | 移除该参数 |
| 早停 callback 打印停止但训练未实际停止 | 1 | `model.stop_training` 在 SAC off-policy 循环未生效，待修 |
| `requirements.txt` push 时 merge conflict | 1 | `git checkout --ours` 保留本地版本后重新 push |
| 远程有新提交导致 push 被拒 | 1 | stash → pull --rebase → stash pop → push |

## Notes
- Phase A 成功消除过拟合（核心目标），但 [64,32] 矫枉过正
- Phase A v2 用 [128,64] + ent_coef 0.01 折中，下一步执行
- 早停 bug 不影响模型质量（最佳模型 @80K 已保存），但浪费了 120K 步的计算
- Phase B 可以在 v2 达到合理 baseline 后启动
- 每个 Phase 结束更新此文件状态
