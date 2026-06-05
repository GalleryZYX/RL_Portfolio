# Findings & Decisions

## Requirements
- SAC 算法训练 51 只 A 股投资组合配置智能体
- 使用长面板 CSV 数据（Tushare），11 个预计算特征因子
- 经 softmax 输出持仓权重，最大化对数收益率
- 解决训练过拟合（训练 178x vs 测试 -0.29%）
- 教师建议：纳入社会新闻及舆论影响

## Research Findings
- **SAC off-policy 样本效率**：100K 步（140 episodes）即可收敛，远少于 PPO 所需的 1M 步
- **ent_coef=auto 的陷阱**：α 从 0.775 自动降至 0.00008，策略坍缩为确定性，丧失泛化能力
- **ent_coef=0.05 过高**：固定 0.05 导致训练 episode 奖励卡在 -0.32，验证奖励持平 0.24，无法收敛
- **[256,256] 网络严重过参数化**：约 70 万参数 vs 714 个交易日，参数/数据比过高
- **[64,32] 网络过小**：4 万参数处理 663 维输入的信息瓶颈太紧，训练集上都是负收益（-2.37%）
- **绝对量纲特征噪声大**：vol 和 amount 跨股票差异达 2-3 个数量级，z-score 在 20 日窗口内不足以消除
- **尺度不变衍生特征有效**：gap/intraday_amp/close_loc 从已有列计算，无需新数据即可生成
- **VecNormalize 训练/评估必须隔离**：共用会导致评估 episode 污染训练 running stats
- **SB3 SAC 不支持 max_grad_norm**：该参数在 2.8.0 版本不存在，需通过 policy_kwargs 或自定义实现
- **早停 bub**：`model.stop_training = True` 在 SAC off-policy `learn()` 循环中未及时检查，训练持续至 total_timesteps
- **Q2 2026 显著优于 Q1**：+7.39% vs -8.02%，说明市场环境对策略表现影响巨大
- **SB3 SAC 不支持原生 dropout**：需继承 SACPolicy 自定义网络
- **Tushare pro.news() + SnowNLP**：可行的中文金融文本情感分析链路
- **AKShare stock_hot_rank_em()**：东方财富每日热门股票排名，无需 NLP 即可获得关注度代理变量
- **CSV 尚有未用列**：open, high, low, pre_close, change, ps_ttm, total_mv, circ_mv 可用于衍生新特征

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Phase A 优先于 B/C | 抗过拟合是当前瓶颈，解决后再加新特征才有意义 |
| [64,32] 网络（Phase A） | 成功消除过拟合，但导致欠拟合（训练 -2.37%） |
| [128,64] 网络（Phase A v2） | 折中容量，~10 万参数，平衡欠拟合与过拟合 |
| 固定 ent_coef=0.05→0.01 | 0.05 过高导致无法收敛，0.01 折中 |
| 时序切分验证集（非随机） | 金融数据的时间箭头不可逆，随机切分 = 未来信息泄露 |
| 新增 gap/intraday_amp/close_loc | 从已有未用价格列衍生，无需新数据，全为尺度不变比率 |
| SnowNLP 作为 NLP 首选 | 轻量、无需 GPU、中文友好；若效果差则降级为 FinBERT 或纯结构特征 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| SAC 训练 178x 收益在测试集无效（过拟合） | Phase A 成功消除 |
| Phase A [64,32] 欠拟合（训练 -2.37%） | Phase A v2: 增大网络 + 降熵系数 |
| ent_coef=auto 导致 α → 0 | Phase A2: 改为固定 0.05 |
| ent_coef=0.05 导致无法收敛 | Phase A v2: 降为 0.01 |
| max_grad_norm 不被 SB3 SAC 2.8.0 支持 | 移除该参数 |
| 早停 callback 未实际停止训练（bug） | model.stop_training 在 off-policy 循环未生效 |
| EvalCallback 与训练共用 VecNormalize | 已修复：创建独立 eval_vec_norm |
| make_env 内过早 env.reset() | 已修复：移除 _init 中的 reset |
| Git push 冲突（requirements.txt） | 保留本地版本，stash → rebase → pop |
| VecNormalize.load() 需 venv 参数 | 先创建 DummyVecEnv 再 load |

## Resources
- SAC 训练脚本：`scripts/train_sac_portfolio.py`
- 验证回调：`scripts/validation_callback.py`（Phase A 新建）
- 环境代码：`envs/portfolio_env.py`
- 数据管线：`data/data_collection.py`
- 训练数据：`data/rl_train_data_2023_2025.csv`（36,532 行，714 天）
- 测试数据：`data/rl_test_data_2026_now.csv`（4,376 行，87 天）
- 技术文档：`技术文档/技术文档4_SAC算法与训练脚本/技术文档4.md`
- Phase 0 总结：`训练日志/SAC_06-04/训练总结_SAC_06-04.md`
- Phase A 总结：`训练日志/SAC_06-05_PhaseA/训练总结_SAC_PhaseA.md`
- SB3 SAC docs: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- AKShare docs: https://akshare.akfamily.xyz/
- SnowNLP: https://github.com/isnowfy/snownlp

## Visual/Browser Findings
- **Phase 0 TensorBoard**：eval/mean_reward 从 -0.003 单调升至 5.185，ent_coef 从 0.775 降至 0.000083
- **Phase A TensorBoard**：eval/mean_reward 全程持平 0.24，ent_coef 固定 0.05，rollout/ep_rew_mean 维持在 -0.32
- **Phase A 验证曲线**：10K 步即达最佳 0.2424，之后 290K 步完全无提升，说明模型容量不足
- **Phase A Q1 vs Q2**：Q1 -8.02%（模型无法应对），Q2 +7.39% 夏普 3.20（模型找到有效信号），市场环境分化显著
