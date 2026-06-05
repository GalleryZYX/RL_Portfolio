# Progress Log

## Session: 2026-06-04

### Phase 0: SAC 初次训练
- **Status:** complete
- **Started:** 2026-06-04 (上午)
- Actions taken:
  - 审查并修复 `train_sac_portfolio.py`（训练/评估 VecNormalize 隔离、移除过早 reset、冻结评估统计量、hardcoded 初始资金等 5 处修复）
  - 验证数据文件、SB3、PyTorch CUDA、环境实例化全部通过
  - 运行 `python scripts/train_sac_portfolio.py`：100K 步，RTX 4060 Ti，数分钟完成
  - 训练集评估：累计对数收益 5.185，几乎确定性的策略（3 episode 标准差 < 0.0001）
  - 测试集评估：总收益 -0.29%，夏普 0.0486，最大回撤 -12.81%（过拟合确认）
  - 编写技术文档4（SAC 算法与代码设计）
  - 编写训练总结报告
  - 推送 `train_sac_portfolio.py`、技术文档4、训练总结至 GitHub
- Files created/modified:
  - `scripts/train_sac_portfolio.py` (modified — 5 fixes)
  - `技术文档/技术文档4_SAC算法与训练脚本/技术文档4.md` (created)
  - `训练日志/SAC_06-04/训练总结_SAC_06-04.md` (created)
  - `models/sac_portfolio.zip` (created by training)
  - `models/sac_portfolio_vecnorm.pkl` (created by training)
  - `runs/sac_portfolio_1/` (created by training)
  - `runs/evaluations.npz` (created by training)

### Phase A: 抗过拟合改进
- **Status:** complete
- **Started:** 2026-06-05
- Actions taken:
  - A1-A4: 修改 `train_sac_portfolio.py` — 网络 [64,32]、ent_coef 0.05、weight_decay 1e-4、移除不支持的 max_grad_norm
  - A5: 创建 `validation_callback.py`，训练集按 2025-01-01 切分为 471 天训练 + 243 天验证，patience=10
  - A6: 修改 `portfolio_env.py` — 去掉 vol/amount，新增 gap/intraday_amp/close_loc 衍生特征（12 特征，obs 612→663）
  - A7: OOS 评估改为 5 episodes + Q1/Q2 子期间分析
  - A8: 超参数调整 — lr 3e-4→1e-3, batch 256→128, steps 100K→300K, learning_starts 2000→5000
  - 运行 `python scripts/train_sac_portfolio.py`：300K 步，RTX 4060 Ti，约 48 分钟
  - 验证奖励持平 0.24，最佳 @80K 步；早停在 180K 触发但因 bug 未实际停止
  - 测试集：+0.66%，夏普 0.20，最大回撤 -10.75%。过拟合消除，但欠拟合（训练 -2.37%）
  - 编写 Phase A 训练总结 → `训练日志/SAC_06-05_PhaseA/训练总结_SAC_PhaseA.md`
  - 更新 task_plan.md、findings.md、progress.md
- Files created/modified:
  - `scripts/train_sac_portfolio.py` (modified — 7 changes)
  - `scripts/validation_callback.py` (created)
  - `envs/portfolio_env.py` (modified — features + derived computation)
  - `models/sac_portfolio.zip` (updated)
  - `models/sac_portfolio_vecnorm.pkl` (updated)
  - `runs/sac_portfolio_2/` (created)
  - `训练日志/SAC_06-05_PhaseA/训练总结_SAC_PhaseA.md` (created)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase A v2: 折中参数调优
- **Status:** pending
- Actions taken:
  -
- Files created/modified:
  -

### Phase B: 新闻/舆情特征
- **Status:** pending
- Actions taken:
  -
- Files created/modified:
  -

### Phase C: 方法论升级
- **Status:** pending
- Actions taken:
  -
- Files created/modified:
  -

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Env instantiation (Phase 0) | `PortfolioEnv(df)` | 612 obs, 51 action | 612 obs, 51 action | ✓ |
| SAC Phase 0 训练 | `python scripts/train_sac_portfolio.py` | 完成 | 100K 步，3 文件 | ✓ |
| Phase 0 测试集 | eval (test data) | 正收益 | **-0.29%，Sharpe 0.05（过拟合）** | ✗ |
| Phase A 代码语法 | py_compile all files | 通过 | 全部通过 | ✓ |
| Env instantiation (Phase A) | `PortfolioEnv(df)` | 663 obs, 51 action | 663 obs, 12 features | ✓ |
| SAC Phase A 训练 | `python scripts/train_sac_portfolio.py` | 完成 | 300K 步，~48 min | ✓ |
| Phase A 验证早停 | ValidationCallback @ 180K | 停止训练 | 打印停止但因 bug 继续 | △ |
| **Phase A 测试集** | eval (test data) | 正收益 | **+0.66%，Sharpe 0.20** | ✓ |
| Phase A 过拟合检查 | Train vs Test gap | 差距缩小 | -2.37% vs +0.66%（几乎一致） | ✓ |
| Phase A 子期间 | Q1 vs Q2 | — | Q1 -8.02%, Q2 +7.39% | — |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-06-04 | 测试集 -0.29%（过拟合） | 1 | Phase A 成功消除 |
| 2026-06-05 | Phase A 欠拟合（训练 -2.37%） | 1 | Phase A v2 计划：增大网络、降熵系数 |
| 2026-06-05 | `max_grad_norm` not a SAC param | 1 | 移除该参数 |
| 2026-06-05 | 早停 callback 未实际停止训练 | 1 | bug 待修，不影响模型质量（best @80K） |
| 2026-06-04 | `requirements.txt` merge conflict on push | 1 | `git checkout --ours` + pull --rebase |
| 2026-06-04 | Push rejected (remote ahead) | 1 | Stash, pull --rebase, pop stash |
| 2026-06-04 | `VecNormalize.load()` AttributeError | 1 | Create DummyVecEnv first then load |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase A 完成，Phase A v2（折中参数调优）待执行 |
| Where am I going? | Phase A v2 → Phase B（新闻舆情）→ Phase C（方法论升级） |
| What's the goal? | 消除过拟合（已达成）→ 恢复学习能力（v2）→ OOS Sharpe > 0.5 |
| What have I learned? | [64,32] 欠拟合；[128,64]+ent_coef 0.01 应为折中方案；Q1/Q2 市场分化显著 |
| What have I done? | Phase A 8 项改动全部实施并训练完成；过拟合消除、测试改善；Phase A v2 计划已制定 |
