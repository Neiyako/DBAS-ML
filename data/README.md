# data/ — 数据目录

| 子目录 | 用途 |
|---|---|
| `raw/` | 原始数据（ERA5 / 真实观测）。**全程只读**，本框架不自动修改真实数据 |
| `processed/` | 清洗/裁切后的数据产物（任何自动清洗都必须由配置显式开启并被记录） |
| `synthetic/` | 合成测试数据，仅用于验证工程链路 |

## 合成测试数据

- 文件：`synthetic/cloudwater_synth.npz`，由固定种子生成（可复现）：

```bash
cloudwater synth --out data/synthetic/cloudwater_synth.npz --n 5000 --seed 42 --force
# 或：python scripts/… / python -c "from cloudwater_ml.data import ensure_synthetic_dataset; ensure_synthetic_dataset(force=True)"
```

- 特征：`t, q, rh, u, v, w`；目标：`clwc, ciwc`；时间跨 2015–2019（支持 time/year 划分测试）。
- **该数据的 `source` 恒为 `simulated`，`dataset=synthetic_test_data`。它是软件测试构造式，不是真实大气物理模拟，也绝不会被标记为真实观测。**

## npz 键名约定（ArrayAdapter）

- 特征 `feature__<变量名>`，目标 `target__<变量名>`，元数据 `meta__<列名>`；
- 也接受裸变量名键；csv/parquet 则直接按列名选择。

## 接入真实数据

见根 README「如何接入真实 ERA5 数据」：netcdf 适配器按需分块读取，禁止默认整载入内存。
