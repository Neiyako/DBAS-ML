# 数据引擎（CloudWater ML Platform v2.0）

> 目标：让 TB 级 ERA5 NetCDF 在**单机内存有限**的前提下可读、可缓存、可校验、可追溯。

---

## 1. 数据流与分级缓存

```
原始文件 (data/raw, 只读)
   │  ① 元数据缓存（变量/维度/单位/契约，不读数值）
   ▼
Metadata Cache  (.cache/cloudwater_ml/metadata)
   │  ② 预处理缓存（SampleBundle: X/Y/meta/units）
   ▼
Preprocessing Cache  (.cache/cloudwater_ml/preprocessing)
   │  ③ 索引缓存（分批读取的区间/变量清单）
   ▼
Index Cache  (.cache/cloudwater_ml/raw)
   │
   ▼
内存缓存（进程内，默认上限 256 MiB，LRU 淘汰）
   │
   ▼
GPU Batch（pinned memory + non_blocking，仅 CUDA）
```

实现：`cloudwater_ml/data/cache.py` 的 `DataCache`。

### 缓存的三条硬约束

1. **永不写入 `data/raw`**：`DataCache` 构造时校验根目录，任何试图把缓存写进原始数据目录的
   配置都会直接报错；写路径还要求「必须在缓存根目录之内」。
2. **可失效**：key 由 **文件版本（size + mtime，可选 sha256）+ 处理规格** 生成。
   文件一变 → key 变 → 旧缓存自动 miss（不需要手工清理）。
3. **可清理 / 可重建 / 可追溯**：

```bash
cloudwater cache info                  # 命中/未命中/各级条目/占用
cloudwater cache clear                 # 全部清理（下次自动重建）
cloudwater cache clear --level metadata
```

### 配置

```yaml
data:
  cache:
    enabled: true          # false = 完全关闭（所有读写变 no-op）
    root: .cache/cloudwater_ml   # 默认在 data/ 之外
    hash_files: false      # true = 用内容哈希做版本（TB 级文件慎开）
```

> 缓存**只加速，不改变语义**：清理缓存后重跑，结果必须一致（cache 内容来自同一份原始数据）。

---

## 2. NetCDF 按需读取

```yaml
data:
  adapter: netcdf
  path: /data/era5/era5_2015.nc
  features: [t, q, rh, u, v, w]
  targets: [clwc, ciwc]
  netcdf:
    time_range: ["2015-01-01", "2015-12-31"]
    latitude_range: [20.0, 50.0]
    longitude_range: [90.0, 130.0]
    level_range: [500.0, 900.0]
    chunks: { time: 16 }
    engine: auto           # netcdf4 | h5netcdf | scipy
```

能力：

| 能力 | 说明 |
|---|---|
| 按时间/空间/气压层/变量筛选 | 位置索引切片，**不读未选中的值** |
| 分块读取 | 有 dask 时真正惰性；无 dask 时**显式告警**（不静默忽略） |
| 体积守卫 | 读取前估算内存，超限**先拦截**并给出**真正有效**的降级建议 |
| `max_samples` 降采样 | 在体积估算**之前**生效（v1.0 此处是死代码，已修） |
| 分批读取 `iter_batches` | 每批样本数受 `batch_samples` 约束；跨批 `sample_id` 唯一；每批带 units / source_file |
| ERA5 别名 | `lev` / `plev` / `pressure_level` 都能识别 |
| 单层 + 气压层混用 | 支持广播对齐；无法对齐时报**明确错误** |

> **不默认整载入**、**不默认转 Zarr**、**不复制原始数据**。

---

## 3. 数据质量系统

```bash
cloudwater data validate --config configs/data.yaml --out outputs/quality.md
cloudwater inspect --config configs/data.yaml
```

检查项（`data/validation.py`）：

| 类别 | 检查 |
|---|---|
| 数值 | NaN / Inf / 异常值（z-score）/ 变量范围（含物理量默认边界） |
| 坐标 | 经度范围与 0-360 处理、纬度范围、网格间隔 |
| 时间 | 单调性、重复时刻、缺测时段、时间连续性 |
| 气压层 | 层顺序、层单位、层范围 |
| 单位 | 变量单位是否齐全、是否与口径族相符 |
| 缺测分布 | 按层 / 按纬度带 / 按季节统计缺失比例 |
| 来源 | real / simulated / unknown 标签与判定依据 |

**报告优先原则**：默认 `quality.report_only: true` → 只出报告，**不改数据**。
确需清洗时必须显式配置，且每个动作都会被记录进实验 `notes`。

---

## 4. 溯源（Provenance）

每条样本携带：

| 字段 | 含义 |
|---|---|
| `sample_id` | 稳定标识（跨批次唯一） |
| `source` | `real` / `simulated` / `unknown` |
| `source_origin` | 来源标签的判定依据（config / dataset_contract / column / unresolved） |
| `dataset` | 数据集标识 |
| `time` / `latitude` / `longitude` / `level` | 时空坐标 |
| `generation_method` / `generation_config` | 仿真数据的生成方式与参数 |

**关键纪律**：来源标签解析不出来时是 `unknown`，**绝不默认成 simulated**。
（v1.0 的坑：真实 ERA5 无契约 → `unknown` → 被混合采样按 real/simulated 筛掉 → 训练集变空。）

`SampleBundle.subset()` 会**透传** `units` / `source_file` / `source_origin`，
因此划分后的 train/val/test 仍可溯源。

---

## 5. 归一化（Scaler）

- **统计量只在训练集拟合**，验证/测试只做 `transform`；
- 拟合证据写入 `preprocessors.json`（样本数、特征名、拟合时间、schema 版本、训练数组指纹），
  可用 `Preprocessors.verify_evidence()` 校验「确实只用了训练集」；
- **NaN/Inf 策略**：fit 时检测，默认**拒绝**并列出「哪一列、多少个」；
  可配置 `nan_policy: impute`（用训练集中位数填补，统计量仍只来自训练集）；
- 支持 `standard` / `minmax` / `none`，可保存、可恢复、可**逆变换回物理单位**。

---

## 6. 真实 / 仿真数据融合

```python
from cloudwater_ml.data.sources import MixedSampler

sampler = MixedSampler(train_bundle, seed=42)
sub, info = sampler.sample(real_fraction=0.75, strategy="ratio")
```

| 策略 | 用途 |
|---|---|
| `ratio` | 按真实数据比例（A~E 配比实验） |
| `source` | 按数据源指定数量 |
| `weighted` | 按来源权重分配名额 |
| `stratified` | 按年份/季节/区域分层后按比例抽样 |

- 配比**只作用于训练集**，测试集永不参与；
- 请求配比无法满足时，**如实记录实际配比**并写 notes（不伪造标签）；
- 来源标签完全不可解析时，**保持全部样本**（不筛空）；
- `keep_scenes_intact=True` 时按 `scene_id` 整组纳入，防止同场景近邻跨集泄漏。

配合 `scene_ids(bundle)` 可在划分前把同场景样本绑定，避免「同一原始场景的仿真变体」
同时出现在训练集与测试集。

---

## 7. 已知限制

- **训练侧尚未接流式**：`Trainer.fit` 收的是 numpy 矩阵；数据层的 `iter_batches`
  已可用，但训练循环的 `IterableDataset` 接入是下一步（当前 TB 级数据需先降采样到
  内存可容纳的规模）。
- 无 dask 时 `chunks` 不生效（会告警），实际读取走 netCDF4 的索引切片。
- 缓存不跨机器迁移（可重建，只影响速度）。
