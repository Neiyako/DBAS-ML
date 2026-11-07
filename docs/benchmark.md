# 性能基准指南（CloudWater ML Platform v2.0）

> 目标：把「慢在哪」变成**可测量**的结论，而不是猜测。
> 所有数字必须来自真实训练；跑不了的项记空，不填「看起来合理」的值。

---

## 1. 快速开始

```bash
# 单次基准（当前设备）
cloudwater gpu benchmark --samples 8192 --batch-size 256

# 写报告
cloudwater gpu benchmark --samples 16384 --epochs 3 --out outputs/benchmark.md

# Python API
from cloudwater_ml.benchmark import benchmark_training, run_benchmark_suite, recommend_config

res = benchmark_training(device="auto", precision="auto", batch_size=256, n_samples=8192)
print(res.samples_per_second, res.bottleneck, res.peak_vram_bytes)
```

**CPU 环境可跑**：基准不要求 CUDA，CPU 上给出基线吞吐与瓶颈归因。

---

## 2. 测什么

| 指标 | 口径 | 来源 |
|---|---|---|
| `samples_per_second` | 样本数 × epoch 数 ÷ 墙钟时间 | 实测 |
| `epoch_seconds` | 每 epoch 平均耗时 | 实测 |
| `data_load_seconds` | **从 DataLoader 取下一个 batch 的等待时间**（含 worker 排队） | 实测（`hardware/profiling.py`） |
| `transfer_seconds` | 把 batch 搬到目标设备的时间 | 实测 |
| `compute_seconds` | 前向 + 反向 + 优化器步进 | 实测（墙钟减去上面两项） |
| `peak_vram_bytes` | 显存峰值 | CUDA `max_memory_reserved`；MPS 用 driver 统计；CPU 为 `None` |
| `peak_host_rss_bytes` | 进程内存峰值 | `resource.getrusage` |
| `gpu_utilization` | `nvidia-smi` 采样 | **无 nvidia-smi 时为 `None`，不伪造** |
| `bottleneck` | `data_loading` / `compute` / `balanced` | 由上面三段耗时推导 |

---

## 3. 怎么读结果

| 现象 | 含义 | 动作 |
|---|---|---|
| `data_load_seconds` 占比 > 50% | **数据供给瓶颈** | 提高 `dataloader.num_workers`、`prefetch_factor`；确认是否每 epoch 重复解析同一 NetCDF（应命中缓存） |
| `compute_seconds` 占比 > 70% | **算力瓶颈** | 开 AMP（`precision: auto`）、`channels_last`（卷积模型）、`torch.compile`（需实测） |
| `transfer_seconds` 占比高 | 主机→设备搬运瓶颈 | CUDA 上开 `pin_memory: true` + `non_blocking`（框架在 CUDA 下自动开） |
| `samples_per_second` 随 batch 增大而下降 | 显存压力/缓存失效 | 降 batch 或开梯度累积 |
| `peak_vram_bytes` 接近 `free_vram` | 显存吃紧 | 降 batch / 开 `grad_checkpointing` / 降精度 |

> `bottleneck` 是**粗判**（阈值分档），用于快速定位；精确归因请看三段耗时的绝对值。

---

## 4. 矩阵基准

```python
from cloudwater_ml.benchmark import run_benchmark_suite

payload = run_benchmark_suite(
    devices=("auto",),
    precisions=("fp32", "bf16"),
    batch_sizes=(64, 128, 256, 512),
    n_samples=16384,
    epochs=2,
)
```

产出（可直接写进论文的实验记录）：

- `device_info`：运行环境与硬件快照；
- `results`：每个用例的完整指标；
- `recommendations`：按**实测硬件**推导的配置建议。

---

## 5. 配置建议的推导方式

`benchmark/runner.py:recommend_config`：

1. 读 `detect_hardware()` 的**运行时**结果（不写死型号/显存）；
2. CUDA：可用显存 × 0.6 作为预算，按特征数估算每样本激活占用，给出起步 `batch_size`
   （上限 1024，下限 32）；
3. 估算 batch 偏小（< 64）时建议 `gradient_accumulation_steps: 4`；
4. 有 bf16 能力 → `precision: bf16`；否则 `fp16`（CUDA 自动带 GradScaler）；
5. MPS → `fp16` 且 `pin_memory: false`；CPU → `fp32` + 少量 worker；
6. `torch.compile` 默认建议关闭（收益需实测，编译本身有成本）。

> 「RTX 5060」在不同批次上显存并不相同，**建议永远以本机 `cloudwater gpu info` 的实测值为准**。

---

## 6. 与实验记录的关系

训练实验本身也会记录同样的三段耗时：

```json
"runtime_plan": {"profiling": {"stages": {
   "data_load": {"total_seconds": 1.23, "count": 64},
   "transfer":  {"total_seconds": 0.11, "count": 64},
   "compute":   {"total_seconds": 9.80, "count": 64}}}}
```

因此**不需要**为每个实验单独跑基准——实验自己的记录已足够定位瓶颈；
基准模块用于「换硬件 / 换精度 / 换 batch 前先摸底」。

---

## 7. 性能测试与性能基准的区别

| | 性能测试（pytest `-m perf`） | 性能基准（benchmark 模块） |
|---|---|---|
| 目的 | **回归防护**（别把链路改慢 10 倍） | **摸底与选型**（该用多大 batch、什么精度） |
| 断言 | 吞吐下限 / 比例关系 | 无断言，输出数据供人判断 |
| 环境 | CI 机器即可，样本量小 | 目标硬件上跑，样本量接近真实 |

> 性能测试的阈值必须**按机器标定**后写入；未标定的绝对阈值在不同机器上会误报，
> 因此本仓库当前只提供基准模块，不提供写死的绝对吞吐断言。

---

## 8. 诚实性约定

- 基准**不**为了让数字好看而缩短 epoch 或减少样本后仍标称大样本量；
- 失败用例（异常）不会被计入结果表，但会在日志里留一条 warning；
- GPU 利用率取不到就是 `None`；
- 报告里的「建议」是**基于实测硬件的启发式**，不是保证；
  真正选型前请用目标数据规模复跑一次基准。
