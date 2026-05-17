# RL_Portfolio

强化学习课程大作业：基于深度强化学习的多资产动态仓位分配策略。

## 环境配置

见'技术文档1'。

推荐使用 Python 3.10，并在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中不固定 CUDA 版 PyTorch，避免没有 NVIDIA GPU 的组员安装过慢或遇到 CUDA 兼容问题。`stable-baselines3` 会自动安装可用的 PyTorch 依赖。

如果需要在 NVIDIA GPU 上训练，并且本机驱动支持 CUDA 12.8，可先手动安装 CUDA 版 PyTorch，再安装项目依赖：

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

如果只需要 CPU 训练，直接运行 `pip install -r requirements.txt` 即可。

## Stable Baselines3 PPO 示例

项目新增了一个官方环境复现脚本：

```text
scripts/train_ppo_pendulum.py
```

该脚本用于跑通 Stable Baselines3 的 PPO 训练流程，环境为 Gymnasium `Pendulum-v1`。它会完成：

- 创建 `Pendulum-v1` 环境
- 使用 PPO + `MlpPolicy` 训练智能体
- 保存模型到 `models/ppo_pendulum.zip`
- 重新加载模型并评估平均奖励
- 写入 TensorBoard 日志到 `runs/`

CPU 运行：

```bash
python scripts/train_ppo_pendulum.py --device cpu
```

指定GPU (例如GPU0) 运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo_pendulum.py --device cuda:0
```

可选参数示例：

```bash
python scripts/train_ppo_pendulum.py --device cpu --total-timesteps 50000 --eval-episodes 10
```

预期输出会包含类似内容：

```text
Saved model to: models/ppo_pendulum.zip
Mean reward over 5 episodes: -1165.97
Reward std: 181.66
TensorBoard logs: /path/to/RL_Portfolio/runs
```

注意：`Pendulum-v1` 的奖励通常为负数，越接近 0 表示表现越好。本脚本当前主要用于确认 SB3 训练、保存、加载、评估和日志链路正常，不追求最优成绩。

查看 TensorBoard：

```bash
tensorboard --logdir runs
```

随后打开终端显示的本地网页地址，通常为 `http://localhost:6006`。

## 数据获取

由于 Tushare API 存在请求频率限制，数据文件不纳入 Git 版本控制。

请从以下链接下载 `stock_data.csv`，放置于项目根目录的 `data/` 文件夹下：
- 

数据说明：包含 2019–2024 年 n 只沪深300成分股日线收盘价。
