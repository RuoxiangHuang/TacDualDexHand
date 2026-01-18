# 触觉数据采集指南

本指南说明如何使用 `collect_tactile_shape_data.py` 脚本收集触觉数据用于训练分类器。

## 快速开始

### 1. 自动模式（推荐）

自动模式会自动遍历所有形状并收集数据，无需手动干预：

```bash
cd /Users/ruoxianghuang/Desktop/TacDualDexHand
python scripts/demos/shape_touch/collect_tactile_shape_data.py \
    --auto_mode \
    --samples_per_shape 5 \
    --sample_interval 5 \
    --output_dir ./data/tactile_shapes \
    --num_envs 1
```

**参数说明：**
- `--auto_mode`: 启用自动数据采集模式（必需）
- `--samples_per_shape`: 每个形状收集的样本数量（默认：100）
- `--sample_interval`: 采样间隔步数（默认：5，即每5步采集一次）
- `--output_dir`: 数据保存目录（默认：`./data/tactile_shapes`）
- `--num_envs`: 并行环境数量（默认：1）

### 2. 手动模式

手动模式允许你通过 GUI 控制机器人，手动选择何时采集数据：

```bash
python scripts/demos/shape_touch/collect_tactile_shape_data.py \
    --samples_per_shape 100 \
    --sample_interval 5 \
    --output_dir ./data/tactile_shapes \
    --num_envs 1
```

**操作说明：**
- 在 Isaac Sim GUI 中移动绿色的 Goal 标记来控制机器人
- 在终端中按 `s` 保存当前样本
- 按 `n` 切换到下一个形状
- 按 `q` 退出并保存数据

### 3. 增加数据多样性

使用 `--randomize_pose` 参数可以随机化末端执行器姿态，增加数据多样性：

```bash
python  scripts/demos/shape_touch/collect_tactile_shape_data.py \
    --auto_mode \
    --randomize_pose \
    --samples_per_shape 200 \
    --sample_interval 5 \
    --output_dir ./data/tactile_shapes
```

## 完整参数列表

```bash
python  scripts/demos/shape_touch/collect_tactile_shape_data.py \
    --num_envs 1 \                    # 并行环境数量
    --output_dir ./data/tactile_shapes \  # 输出目录
    --samples_per_shape 100 \         # 每个形状的样本数
    --sample_interval 5 \              # 采样间隔（步数）
    --auto_mode \                      # 启用自动模式
    --randomize_pose \                 # 随机化姿态增加多样性
    --debug_vis                        # 在GUI中显示触觉图像
```

## 输出格式

数据会以 Zarr 格式保存，包含以下内容：

- **tactile_images**: 触觉RGB图像数组，形状为 `(N, 240, 320, 3)`，原始分辨率
- **labels**: 形状类别标签数组，形状为 `(N,)`
- **shape_names**: 形状名称列表

文件命名格式：`tactile_shape_data_YYYYMMDD_HHMMSS.zarr`

## 数据收集建议

1. **样本数量**：
   - 最少：每个形状 50-100 个样本
   - 推荐：每个形状 200-500 个样本
   - 高质量：每个形状 1000+ 个样本

2. **采样间隔**：
   - 快速采集：`--sample_interval 1`（每步采集）
   - 平衡：`--sample_interval 5`（默认）
   - 减少冗余：`--sample_interval 10`

3. **数据多样性**：
   - 使用 `--randomize_pose` 增加姿态多样性
   - 多次运行收集不同接触状态的数据
   - 在不同形状位置和角度下采集

## 示例：完整的数据收集流程

```bash
# 1. 创建输出目录
mkdir -p ./data/tactile_shapes

# 2. 收集基础数据集（每个形状100个样本）
isaaclab -p scripts/demos/shape_touch/collect_tactile_shape_data.py \
    --auto_mode \
    --samples_per_shape 100 \
    --sample_interval 5 \
    --output_dir ./data/tactile_shapes

# 3. 收集增强数据集（每个形状200个样本，带姿态随机化）
isaaclab -p scripts/demos/shape_touch/collect_tactile_shape_data.py \
    --auto_mode \
    --randomize_pose \
    --samples_per_shape 200 \
    --sample_interval 5 \
    --output_dir ./data/tactile_shapes
```

## 验证数据

数据收集完成后，可以使用以下 Python 代码验证数据：

```python
import zarr
import numpy as np

# 加载数据
data_path = "./data/tactile_shapes/tactile_shape_data_YYYYMMDD_HHMMSS.zarr"
root = zarr.open(data_path, mode='r')

# 查看数据信息
print(f"样本数量: {root.attrs['num_samples']}")
print(f"形状数量: {root.attrs['num_shapes']}")
print(f"每个形状的样本数: {root.attrs['samples_per_shape']}")
print(f"图像形状: {root.attrs['image_shape']}")

# 查看数据
tactile_images = root['tactile_images'][:]
labels = root['labels'][:]
shape_names = root['shape_names'][:].tolist()  # Convert to list

print(f"\n触觉图像形状: {tactile_images.shape}")
print(f"标签形状: {labels.shape}")
print(f"形状名称: {shape_names}")
print(f"标签分布: {np.bincount(labels)}")
```

## 故障排除

1. **Isaac Sim 无法启动**：
   - 确保已正确安装 Isaac Sim 4.5
   - 检查 GPU 驱动和 CUDA 版本

2. **数据保存失败**：
   - 检查输出目录权限
   - 确保有足够的磁盘空间

3. **采集速度慢**：
   - 减少 `--num_envs` 到 1
   - 增加 `--sample_interval` 值

4. **内存不足**：
   - 减少 `--samples_per_shape`
   - 分批收集数据

## 下一步

数据收集完成后，可以：
1. 使用收集的数据训练触觉分类器
2. 使用 `run_shape_touch_with_classifier.py` 测试分类器性能
3. 将数据用于 Diffusion Policy 的条件输入
