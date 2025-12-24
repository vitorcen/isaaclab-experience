# Isaac Lab 演示与推理

Isaac Lab 提供丰富的预训练策略和环境，可以直接运行观察机器人行为，无需训练。

> **📦 预训练检查点下载**：运行 `./download_checkpoints.sh` 可批量下载可用的检查点。并非所有任务都有公开的预训练模型。

---

## 运行推理的两种方式

### 方式一：使用官方预训练模型（推荐）

最简单的方式，自动从 NVIDIA Nucleus 下载预训练检查点。

> **注意**：如果你使用的是 pip 安装方式（推荐），请直接使用 Python 运行脚本，而不是使用 `isaaclab.sh`。

```bash
conda activate isaaclab

# 如果你使用的是 pip 安装方式 (确保已激活 conda 环境)
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --use_pretrained_checkpoint

# 如果你使用的是源码安装方式
cd IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --use_pretrained_checkpoint
```

### 方式二：使用本地训练的模型

如果你已经训练过模型，使用本地检查点：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint /path/to/checkpoint.pt

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint /path/to/checkpoint.pt
```

---

## 可用的演示环境

### 🤖 经典控制

#### 1. Cartpole - 倒立摆平衡

**任务**：`Isaac-Cartpole-Direct-v0`

**描述**：控制小车移动以平衡杆子直立。

**运行**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 64

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 64
```

**变体**：

- `Isaac-Cartpole-RGB-Camera-Direct-v0`：使用 RGB 相机观察
- `Isaac-Cartpole-Depth-Camera-Direct-v0`：使用深度相机观察

---

#### 2. Cart Double Pendulum - 双摆车

**任务**：`Isaac-Cart-Double-Pendulum-Direct-v0`

**描述**：更具挑战性的双摆平衡任务。

**运行**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cart-Double-Pendulum-Direct-v0 \
    --use_pretrained_checkpoint

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cart-Double-Pendulum-Direct-v0 \
    --use_pretrained_checkpoint
```

---

### 🦎 四足机器人运动

#### 3. ANYmal-C - 四足机器人行走

**任务**：`Isaac-Velocity-Rough-Anymal-C-Direct-v0` ✅

**描述**：四足机器人在崎岖地形上的速度跟踪。

> ✅ **检查点可用**：此任务有预训练检查点

**运行**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 32

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 32
```

**变体**：

- ✅ `Isaac-Velocity-Flat-Anymal-C-Direct-v0`：平坦地形（有检查点）
- ✅ `Isaac-Velocity-Flat-Anymal-D-v0`：ANYmal-D 平坦地形（有检查点）

---

#### 4. Ant - 蚂蚁机器人

**任务**：`Isaac-Ant-Direct-v0` ✅

**描述**：四足蚂蚁机器人学习行走。

**运行**：

```bash
conda activate isaaclab

#  pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ant-Direct-v0 \
    --use_pretrained_checkpoint

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ant-Direct-v0 \
    --use_pretrained_checkpoint
```

---

#### 5. Unitree 系列 - 商业四足机器人

**可用任务** ✅：

- `Isaac-Velocity-Flat-Unitree-A1-v0`
- `Isaac-Velocity-Flat-Unitree-Go1-v0`
- `Isaac-Velocity-Flat-Unitree-Go2-v0`

**描述**：Unitree 商业四足机器人在平坦地形的速度跟踪。

**运行示例（Go2）**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --use_pretrained_checkpoint

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --use_pretrained_checkpoint
```

---

#### 6. Unitree 人形机器人 - G1 & H1

**可用任务** ✅：

- `Isaac-Velocity-Flat-G1-v0` - G1 平地行走
- `Isaac-Velocity-Rough-G1-v0` - G1 崎岖地形行走
- `Isaac-Velocity-Flat-H1-v0` - H1 平地行走

**描述**：Unitree G1 和 H1 是商业人形机器人，支持速度跟踪和复杂地形导航。

**运行示例（G1 平地）**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16
```

**运行示例（G1 崎岖地形）**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-G1-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-G1-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16
```

---

### 🚶 人形机器人

#### 7. Humanoid - 人形机器人行走

**任务**：`Isaac-Humanoid-Direct-v0`

**描述**：人形机器人学习行走和平衡。

**运行**：

```bash
conda activate isaaclab

#  pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16
```

---

#### 8. Humanoid AMP - 基于运动先验的人形机器人

**任务**：`Isaac-Humanoid-AMP-Run-Direct-v0`

**描述**：使用 Adversarial Motion Priors 学习自然运动。

**运行**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-AMP-Run-Direct-v0 \
    --use_pretrained_checkpoint

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-AMP-Run-Direct-v0 \
    --use_pretrained_checkpoint
```

---

### ✋ 灵巧手操作

#### 9. Shadow Hand - 物体翻转

**任务**：`Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0`

**描述**：Shadow Hand 灵巧手翻转立方体。

**运行**：

```bash
conda activate isaaclab

#  pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 8

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 8
```

---

#### 10. Allegro Hand - Allegro 灵巧手

**任务**：`Isaac-Allegro-Hand-Direct-v0`

**描述**：Allegro Hand 的物体操作。

**运行**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Allegro-Hand-Direct-v0 \
    --use_pretrained_checkpoint

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Allegro-Hand-Direct-v0 \
    --use_pretrained_checkpoint
```

---

### 🦾 机械臂操作

#### 11. Franka Cabinet - 开柜子

**任务**：`Isaac-Franka-Cabinet-Direct-v0`

**描述**：Franka 机械臂学习打开抽屉。

> ⚠️ **注意**：此任务目前没有公开的预训练检查点，需要自行训练。

**运行**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Franka-Cabinet-Direct-v0 \
    --use_pretrained_checkpoint

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Franka-Cabinet-Direct-v0 \
    --use_pretrained_checkpoint
```

---

### 🚁 飞行器

#### 12. Quadcopter - 四旋翼

**任务**：`Isaac-Quadcopter-Direct-v0`

**描述**：四旋翼飞行器的悬停和位置控制。

**运行**：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --use_pretrained_checkpoint \
    --num_envs 16
```

---

## 高级选项

### 录制视频

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --use_pretrained_checkpoint \
    --video \
    --video_length 200

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --use_pretrained_checkpoint \
    --video \
    --video_length 200
```

视频保存在：`logs/rsl_rl/<task>/videos/play/`

---

### 实时模式

以真实时间速度运行（如果硬件允许）：

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-Direct-v0 \
    --use_pretrained_checkpoint \
    --real-time

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-Direct-v0 \
    --use_pretrained_checkpoint \
    --real-time
```

---

### 调整环境数量

```bash
--num_envs 64   # 同时运行 64 个环境（默认根据任务配置）
```

---

### 指定设备

```bash
--device cuda:0   # 使用 GPU 0
--device cpu      # 使用 CPU（不推荐）
```

---

## 支持的 RL 框架

Isaac Lab 支持多种强化学习框架的推理：

### RSL-RL（推荐）

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task <task_name> \
    --use_pretrained_checkpoint

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task <task_name> \
    --use_pretrained_checkpoint
```

### RL-Games

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/rl_games/play.py \
    --task <task_name>

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task <task_name>
```

### SKRL

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/skrl/play.py \
    --task <task_name>

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task <task_name>
```

### Stable-Baselines3

```bash
conda activate isaaclab

# pip 安装方式
python scripts/reinforcement_learning/sb3/play.py \
    --task <task_name>

# 源码安装方式
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
    --task <task_name>
```

---

## 环境配置

### Direct 环境 vs Manager-Based 环境

- **Direct 环境**：单文件实现，性能优先，适合快速迭代

  - 任务名格式：`Isaac-<Name>-Direct-v0`
  - 示例：`Isaac-Cartpole-Direct-v0`
- **Manager-Based 环境**：模块化设计，易于扩展

  - 任务名格式：`Isaac-<Name>-v0`
  - 示例：`Isaac-Lift-Cube-Franka-v0`

---

## 完整任务列表

### Direct 环境（高性能）

| 任务名                                      | 描述              | 难度     |
| ------------------------------------------- | ----------------- | -------- |
| `Isaac-Cartpole-Direct-v0`                | 倒立摆平衡        | ⭐       |
| `Isaac-Cart-Double-Pendulum-Direct-v0`    | 双摆平衡          | ⭐⭐     |
| `Isaac-Ant-Direct-v0`                     | 蚂蚁机器人行走    | ⭐⭐     |
| `Isaac-Humanoid-Direct-v0`                | 人形机器人行走    | ⭐⭐⭐   |
| `Isaac-Humanoid-AMP-Run-Direct-v0`        | AMP 人形运动      | ⭐⭐⭐⭐ |
| `Isaac-Velocity-Rough-Anymal-C-Direct-v0` | 四足崎岖地形      | ⭐⭐⭐   |
| `Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0` | Shadow Hand 翻转  | ⭐⭐⭐⭐ |
| `Isaac-Allegro-Hand-Direct-v0`            | Allegro Hand 操作 | ⭐⭐⭐⭐ |
| `Isaac-Franka-Cabinet-Direct-v0`          | 开柜子            | ⭐⭐⭐   |
| `Isaac-Quadcopter-Direct-v0`              | 四旋翼飞行        | ⭐⭐⭐   |

---

## 常见问题

### 预训练检查点下载失败

**现象**：提示 `A pre-trained checkpoint is currently unavailable for this task.`

**原因**：该任务的预训练模型尚未发布或网络连接问题。

**解决方案**：

1. 运行项目根目录下的 `./download_checkpoints.sh` 查看哪些检查点可用
2. 检查任务名是否正确（区分大小写）
3. 使用其他有预训练模型的任务（如 `Isaac-Cartpole-Direct-v0`, `Isaac-Humanoid-Direct-v0`, `Isaac-Ant-Direct-v0`）
4. 自己训练该任务的模型

**已知可用的检查点**（截至 2025-12-24）：

**Direct Workflow 任务**（高性能）：

- ✅ `Isaac-Cartpole-Direct-v0`
- ✅ `Isaac-Humanoid-Direct-v0`
- ✅ `Isaac-Ant-Direct-v0`
- ✅ `Isaac-Quadcopter-Direct-v0`
- ✅ `Isaac-Velocity-Flat-Anymal-C-Direct-v0`
- ✅ `Isaac-Velocity-Rough-Anymal-C-Direct-v0`
- ✅ `Isaac-Repose-Cube-Allegro-Direct-v0`
- ❌ `Isaac-Franka-Cabinet-Direct-v0` (无预训练检查点)

**Manager-Based 任务**（模块化）：

- ✅ `Isaac-Reach-Franka-v0`
- ✅ `Isaac-Cartpole-v0`, `Isaac-Humanoid-v0`, `Isaac-Ant-v0`
- ✅ `Isaac-Velocity-Flat-Anymal-C-v0`, `Isaac-Velocity-Rough-Anymal-C-v0`
- ✅ `Isaac-Velocity-Flat-Anymal-D-v0`
- ✅ Unitree 四足系列: `A1-v0`, `Go1-v0`, `Go2-v0`
- ✅ Unitree 人形系列: `G1-v0` (Flat/Rough), `H1-v0` (Flat)

---

### 运行时内存不足

**解决方案**：减少环境数量

```bash
--num_envs 8   # 从 64 减少到 8
```

---

### GPU 显存不足

**解决方案**：

1. 减少环境数量：`--num_envs 4`
2. 禁用视频录制：移除 `--video`
3. 使用更简单的任务（如 Cartpole）

---

### 渲染窗口不显示

**解决方案**：

1. 确保运行在有显示器的环境（非 SSH）
2. 使用 `--video` 录制视频而不是实时查看
3. 使用 `--headless` 参数无头运行并录制视频

---

## 快速参考

| 需求         | 命令 (pip 安装方式使用 `python`)                                                                                   |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| 最简单的演示 | `python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-Direct-v0 --use_pretrained_checkpoint` |
| 录制视频     | 添加 `--video --video_length 200`                                                                                  |
| 实时速度运行 | 添加 `--real-time`                                                                                                 |
| 调整环境数量 | 添加 `--num_envs 32`                                                                                               |
| 查看可用任务 | 查看[官方文档 - 环境列表](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)                  |

---

## 推荐入门路径

1. **入门**：`Isaac-Cartpole-Direct-v0` - 简单快速 ✅
2. **四足运动**：`Isaac-Ant-Direct-v0` - 观察运动学习 ✅
3. **人形机器人**：`Isaac-Humanoid-Direct-v0` - 复杂平衡 ✅
4. **机械臂操作**：`Isaac-Reach-Franka-v0` - 机械臂到达任务 ✅
5. **高级挑战**：`Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0` - 灵巧手操作

> ✅ 表示有预训练检查点可用

---

**Happy Robot Watching! 🎥🤖**

> **提示**：在使用 `python` 运行脚本时，请确保已进入 `IsaacLab` 目录并激活了正确的 conda 环境 (`isaaclab`)。
