# Ball Rolling Tactile 任务结构说明

本文档详细解释 `ball_rolling_tactile` 任务各个部分的作用。

## 目录结构

```
ball_rolling_tactile/
├── __init__.py                    # 环境注册文件
├── ball_rolling_tactile_rgb.py    # 主要环境实现（RGB触觉图像）
├── ball_rolling_depth.py          # 深度图版本
├── ball_rolling_taxim_fots.py    # Taxim+FOTS仿真版本
├── ball_rolling_tactile_rgb_uipc.py  # UIPC软体仿真版本
├── agents/                        # 强化学习算法配置
│   ├── __init__.py
│   ├── skrl_ppo_cfg.yaml          # PPO配置（仅本体感觉）
│   ├── skrl_ppo_tactile_rgb_cfg.yaml  # PPO配置（触觉RGB）
│   ├── skrl_ppo_camera_cfg.yaml   # PPO配置（相机）
│   └── skrl_sac_cfg.yaml          # SAC配置
└── docs/                          # 文档
```

## 各部分详细说明

### 1. `__init__.py` - 环境注册

**作用**: 将环境注册到 Gymnasium 注册表中，使其可以通过 `gym.make()` 创建。

**关键内容**:
- 导入环境类和配置类
- 使用 `gym.register()` 注册环境
- 指定 `entry_point`（环境类的路径）
- 指定配置文件路径（`skrl_cfg_entry_point` 等）

**示例**:
```python
gym.register(
    id="TacEx-Ball-Rolling-Tactile-RGB-v0",
    entry_point=f"{__name__}.ball_rolling_tactile_rgb:BallRollingTactileRGBEnv",
    kwargs={
        "env_cfg_entry_point": BallRollingTactileRGBCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_tactile_rgb_cfg.yaml",
    },
)
```

### 2. `ball_rolling_tactile_rgb.py` - 环境实现

这是核心文件，包含环境的完整实现。

#### 2.1 配置类 (`BallRollingTactileRGBCfg`)

**作用**: 定义环境的所有可配置参数。

**主要部分**:
- **`viewer`**: 3D 视图配置（相机位置、视角等）
- **`sim`**: 仿真参数（时间步长、物理引擎设置等）
- **`scene`**: 场景设置（并行环境数量、环境间距等）
- **`robot`**: 机器人配置（Franka Panda + GelSight）
- **`gsmini`**: 触觉传感器配置
- **`object`**: 物体配置（球）
- **`observation_space`**: 观测空间定义
- **`action_space`**: 动作空间定义
- **`reward_cfg`**: 奖励函数配置

#### 2.2 环境类 (`BallRollingTactileRGBEnv`)

继承自 `DirectRLEnv`，实现以下核心方法：

##### `__init__()` - 初始化
- 调用父类初始化
- 设置 IK 控制器
- 初始化缓冲区（动作、状态等）
- 创建可视化器（如果启用）

##### `_setup_scene()` - 场景设置
- 创建机器人 (`Articulation`)
- 创建物体 (`RigidObject`)
- 克隆环境（创建多个并行环境）
- 创建传感器（`FrameTransformer`, `GelSightSensor`）
- 创建地面和灯光

##### `_pre_physics_step()` - 物理步进前处理
- 接收并处理动作
- 将动作转换为 IK 命令
- 设置控制器目标

##### `_apply_action()` - 应用动作
- 计算当前末端执行器位姿
- 使用 IK 控制器计算关节目标位置
- 设置机器人关节目标

##### `_get_observations()` - 获取观测
- **本体感觉观测**: 末端执行器位置、姿态、目标位置、物体位置等
- **视觉观测**: 触觉传感器 RGB 图像
- 返回字典格式的观测

##### `_get_rewards()` - 计算奖励
- 根据任务目标计算各种奖励项
- 例如：到达物体奖励、目标跟踪奖励、成功奖励等
- 返回总奖励

##### `_get_dones()` - 判断终止条件
- 检查是否成功完成任务
- 检查是否超时
- 检查是否越界
- 返回 `(terminated, truncated)` 元组

##### `_reset_idx()` - 重置环境
- 重置机器人状态
- 重置物体位置
- 随机化目标位置
- 重置内部缓冲区

### 3. `agents/` - 算法配置

#### 3.1 `skrl_ppo_tactile_rgb_cfg.yaml`

**作用**: 定义 PPO 算法的超参数和网络结构。

**主要部分**:
- **`models`**: 定义策略网络和价值网络
  - `policy`: 策略网络（GaussianMixin，输出动作分布）
  - `value`: 价值网络（DeterministicMixin，输出状态价值）
  - 网络结构：CNN 提取视觉特征 + MLP 处理本体感觉
- **`memory`**: 经验回放缓冲区配置
- **`agent`**: PPO 算法超参数
  - `rollouts`: 每次收集的步数
  - `learning_epochs`: 每次更新的轮数
  - `learning_rate`: 学习率
  - `discount_factor`: 折扣因子
- **`trainer`**: 训练器配置
  - `timesteps`: 总训练步数

### 4. 数据流

```
训练脚本
  ↓
gym.make("TacEx-Ball-Rolling-Tactile-RGB-v0")
  ↓
环境初始化 (__init__)
  ↓
场景设置 (_setup_scene)
  ↓
训练循环:
  ├─ 获取动作 (agent.act)
  ├─ 环境步进 (env.step)
  │  ├─ _pre_physics_step (处理动作)
  │  ├─ _apply_action (应用动作)
  │  ├─ 物理仿真步进
  │  ├─ _get_observations (获取观测)
  │  ├─ _get_rewards (计算奖励)
  │  └─ _get_dones (判断终止)
  └─ 更新策略 (agent.update)
```

### 5. 关键设计模式

1. **配置类模式**: 使用 `@configclass` 装饰器，所有参数集中管理
2. **继承模式**: 继承 `DirectRLEnv`，复用基础功能
3. **传感器模式**: 通过 `scene.sensors` 注册传感器，自动管理
4. **并行环境**: 通过 `clone_environments()` 创建多个并行环境
5. **IK 控制**: 使用 `DifferentialIKController` 进行任务空间控制

### 6. 与 UIPC 版本的区别

- **不使用 UIPC**: 物体是刚体 (`RigidObject`)，不支持软体变形
- **性能更高**: 刚体仿真比软体仿真快得多
- **功能限制**: 无法模拟软体接触和变形

### 7. 扩展建议

要创建新任务，可以：
1. 复制 `ball_rolling_tactile_rgb.py` 作为模板
2. 修改配置类中的参数
3. 修改 `_get_rewards()` 实现任务特定的奖励
4. 修改 `_get_dones()` 实现任务特定的终止条件
5. 在 `__init__.py` 中注册新环境

