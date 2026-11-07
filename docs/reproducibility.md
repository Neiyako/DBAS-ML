# 可复现性指南（CloudWater ML Platform v2.0）

> 可复现 ≠ 「跑出一样的数」。可复现 = **能说清当时发生了什么，并且能重新得到统计上一致的结果**。

---

## 1. 复现的四要素

| 要素 | 落盘位置 | 采集函数 |
|---|---|---|
| 代码版本 | `metrics.json → provenance.git` | `experiments/provenance.py:git_info` |
| 数据版本 | `metrics.json → provenance.data_fingerprint` | `experiments/provenance.py:data_fingerprint` |
| 配置 | `config.yaml`（展开）+ `config.source.yaml`（原文） | `utils/config.py:load_config` |
| 随机性与硬件 | `state.json` / `runtime_plan` / `provenance.environment` | `utils/reproducibility.py`、`hardware/` |

---

## 2. 代码版本

```json
"git": {"available": true, "commit": "…", "short_commit": "abc1234",
        "branch": "main", "dirty": true, "dirty_files": 3}
```

- 非 git 仓库或 git 不可用时：`{"available": false, "reason": "…"}`（**不编造 commit**）；
- `dirty: true` 表示工作区有未提交改动 —— 此时的实验**不可严格复现**，
  因为「当时的代码」并不存在于任何 commit 里（这是纪律问题，不是工具问题）。

> **建议**：正式实验前先提交代码，让 `dirty: false`。

---

## 3. 数据版本

```json
"data_fingerprint": {
  "available": true,
  "path": "/abs/path/era5_2015.nc",
  "size_bytes": 123456789,
  "mtime": 1699999999.0,
  "sha256": "…", "hashed_bytes": 16777216, "partial": true
}
```

- 大文件只哈希**首尾各 8 MiB**（`partial: true`），避免为指纹读完 TB 级文件；
- `path + size + mtime + sha256(片段)` 足以判断「是不是同一份数据」；
- 指纹会写进 `metrics.json`，重新评估时可核对。

---

## 4. 随机性

### 4.1 种子设置

```python
from cloudwater_ml.utils import set_seed
set_seed(42, deterministic=True)
```

统一设置 `random` / `numpy` / `torch`（/ `torch.cuda`），并设置 `PYTHONHASHSEED`。

### 4.2 续训的随机流连续性

检查点保存 `rng_state`（python / numpy / torch / cuda），续训时恢复：

```python
from cloudwater_ml.utils import capture_rng_state, restore_rng_state
```

因此「训练 5 epoch」与「训练 2 epoch + 续训 3 epoch」的随机流是连续的
（受 §6 的 GPU 非确定性限制）。

### 4.3 DataLoader 的顺序

- shuffle 顺序由**主进程 RNG** 决定，与 `num_workers` 无关；
- 但**每个 worker 内部的随机增强**（若模型/数据集用了 worker 级随机）需要额外
  `worker_init_fn` 固定 —— 当前仓库的数据层没有随机增强，故不受影响；
  若后续引入随机增强，必须同步补 `worker_init_fn`（否则顺序会漂）。

---

## 5. 确定性模式的边界（必须如实说明）

```yaml
training:
  deterministic: true
```

会调用 `torch.use_deterministic_algorithms(True, warn_only=True)`。

**仍然不确定的部分**（平台不掩盖）：

- CUDA 上部分算子（原子加 `index_put_`、某些卷积反向、scatter 类）本身不确定；
- cuDNN 的卷积算法选择可能随驱动/库版本变化；
- `warn_only=True` 意味着**不会报错**，只会在日志里告警；
- 多进程 DataLoader 的浮点归约顺序可能带来末位差异。

结论：**平台承诺「同种子 + 同数据 + 同配置 + 同环境 ⇒ 统计上一致」，不承诺位级一致。**
论文中若需要位级一致，应固定驱动/CUDA/PyTorch 版本并关闭多进程增强随机。

---

## 6. 依赖与环境

`metrics.json → environment` 记录：Python、平台、各依赖版本、PyTorch、CUDA 构建、cuDNN、
GPU 型号/算力/显存、MPS 可用性。

```bash
cloudwater doctor --json > env.json    # 存档环境快照
```

> 版本冲突排查提示：`experiments/runner.py:_environment` 用**发行名**取版本
> （`scikit-learn` 而不是 `sklearn`）——早期版本用错名字导致环境记录谎报「未安装」，已修正。

---

## 7. 归一化统计量的可复现

- 统计量**只能在训练集上拟合**：`fit_scalers` 记录拟合证据
  （样本数、特征名、拟合时间、schema 版本、**训练数组内容指纹**）；
- `Preprocessors.verify_evidence()` 可校验「确实只用了训练集」；
- 验证集与测试集只做 `transform`，永不参与拟合；
- scaler 状态随实验落盘（`preprocessors.json`），可精确还原指标口径；
- **逆变换**用于把预测还原到物理单位，指标一律在物理单位下计算。

---

## 8. 复现一个历史实验的操作步骤

```bash
# 1) 看原实验的溯源信息
cloudwater report --experiment experiment_001

# 2) 取出配置与 git commit
cat outputs/experiment_001/config.yaml
python -c "import json;print(json.load(open('outputs/experiment_001/metrics.json'))['provenance']['git'])"

# 3) 确认代码版本一致（必要时切分支/回退 commit）

# 4) 确认数据指纹一致
python -c "import json;print(json.load(open('outputs/experiment_001/metrics.json'))['provenance']['data_fingerprint'])"

# 5) 重跑
cloudwater train --config outputs/experiment_001/config.yaml --name rerun_001

# 6) 对比
cloudwater compare --experiments experiment_001 rerun_001
```

---

## 9. 复现失败时的排查顺序

| 现象 | 先查 |
|---|---|
| 指标明显不同 | 数据指纹 → 配置快照 → 代码 commit → 依赖版本 |
| 指标接近但不完全相同 | GPU 非确定性（§5）、worker 数、编译开关、精度档位 |
| 划分样本数不同 | 数据顺序/样本量变化（指纹）、划分参数、`seed` |
| 训练早期就分叉 | `set_seed` 是否在建模前调用、DataLoader `shuffle` 顺序、dropout 流 |
| 续训结果与不续训不一致 | 检查点是否含 `rng_state` / `history` / `early_stopping_state`（v2.0 已含） |
