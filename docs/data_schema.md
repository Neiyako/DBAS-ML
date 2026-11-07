# 数据结构与数据契约 · cloudwater-ml

> 出处标注：`路径:行号`。本文的每个键名/列名都能在代码里找到定义。

## 1. 统一载体 `SampleBundle`

定义在 `src/cloudwater_ml/data/schema.py:63`：

| 字段 | 类型/形状 | 含义 | 出处 |
|---|---|---|---|
| `X` | `float64 [n_samples, n_features]` | 输入特征矩阵，1 维输入会被自动 reshape 成列 | `schema.py:75,:86-87` |
| `Y` | `float64 [n_samples, n_targets]` | 目标矩阵；**单目标也是二维，列数 1** | `schema.py:69,:88-89` |
| `feature_names` | `list[str]`，长度 = `X.shape[1]` | 特征列名，顺序与列一致 | `schema.py:92-95` 校验不符即抛错 |
| `target_names` | `list[str]`，长度 = `Y.shape[1]` | 目标列名 | `schema.py:96-99` |
| `meta` | `DataFrame`，行数 = 样本数 | 追溯元数据，行与 X/Y 严格对齐 | `schema.py:108-111` 行数不符抛错 |
| `units` | `dict[var, str]` | 变量单位，来自数据源元数据，供质量检查 | `schema.py:80`；NetCDF 填充 `netcdf_adapter.py:241,:252` |
| `source_file` | `str` | 样本来自哪个数据文件（追溯） | `schema.py:81`；`loader.py:52-53` 兜底 |

内置操作：

- `subset(indices)`：按索引取子集，**保持 X/Y/meta 三者对齐**（划分与配比的基础操作）——
  `schema.py:129-138`；
- `source_labels` / `source_counts()`：来源标签读取，缺 `source` 列时按 `unknown`
  显式处理而不是补值——`schema.py:140-148`；
- `describe()`：给 `cloudwater inspect` 用的结构摘要（不读数值含义）——`schema.py:150`；
- `validate_bundle_dims()`：X/Y 必须二维 + 浮点 + 非空——`schema.py:165-178`。

## 2. 表格形状与时空预留形状

| 用途 | 形状 | 现状 | 出处 |
|---|---|---|---|
| 表格输入 `X` | `[n_samples, n_features]` | 已实现（全链路默认） | `schema.py:68` |
| 表格目标 `Y` | `[n_samples, n_targets]` | 已实现 | `schema.py:69` |
| 时空输入 `X_spatial` | `[batch, channels, height, width]` | **预留**（CNN/U-Net 第二阶段） | `spatial_shape_guard(shape, "spatial")`，`schema.py:181-191` |
| 时序时空 `X_temporal` | `[batch, time, channels, height, width]` | **预留**（ConvLSTM 第二阶段） | `spatial_shape_guard(shape, "temporal")` 同上 |

`spatial_shape_guard` 目前只做维数校验（4 维 / 5 维），已由
`tests/test_data.py:80` 覆盖；它被文档与第二阶段模型共用（`schema.py:182` 注释）。

## 3. meta 列与追溯约定

| 列 | 含义 | 谁产生 | 出处 |
|---|---|---|---|
| `sample_id` | 样本唯一 ID；缺列时自动插入 0..n-1 | `SampleBundle.__post_init__` 自动补 | `schema.py:112-114` |
| `time` | 时间戳（`DatetimeIndex`） | 各适配器 | NetCDF：`netcdf_adapter.py:167-169`；npz/csv：`array_adapter.py:110-114` |
| `year` | 年份（给 year 划分与泛化分组） | NetCDF 由 time 推导 `netcdf_adapter.py:170`；合成数据直接写 `synthetic.py:100` | |
| `latitude` / `longitude` | 经纬度（region 划分、空间图） | 坐标网格展平 `netcdf_adapter.py:161-166`；合成数据 `synthetic.py:82-83` | |
| `level` | 垂直层次（坐标别名 `lev`/`plev` 统一为 level） | `_ALIASES`，`netcdf_adapter.py:30-36` | |
| `source` | `real` / `simulated` / `unknown` | 见 §3.1 | |
| `source_origin` | `config` / `dataset_contract` / `unresolved`：标签从哪来 | `netcdf_adapter.py:181` |
| `dataset` | 数据集名（npz/csv 缺省时取文件名 stem） | `array_adapter.py:121-122`；NetCDF `netcdf_adapter.py:182` |
| `source_file`（Bundle 字段） | 数据文件路径 | `netcdf_adapter.py:253` / `loader.py:52` |
| `units`（Bundle 字段） | 每变量单位 | `netcdf_adapter.py:241` |

### 3.1 来源标签的三级解析（防止仿真冒充真实）

NetCDF（`netcdf_adapter.py:178-182`）：

```
data.source_label（配置显式覆盖，origin=config）
  > 数据集契约 source.kind（atmosgen:contract JSON 或 source_kind 属性，origin=dataset_contract）
  > "unknown"（origin=unresolved）
```

- 契约解析实现：`netcdf_adapter.py:70-84`（只接受 `real`/`simulated` 两个值，
  JSON 解析失败静默降级到属性查询）；
- npz/csv/parquet：`data.source_label` 覆盖 > 已有 `source` 列 > `unknown`
  （`array_adapter.py:116-120`）；
- 合成数据 `source` **恒为 `simulated`**，`dataset=synthetic_test_data`
  （`synthetic.py:98-99`），并有专门测试
  `tests/test_data.py:55 test_synthetic_source_always_simulated`；
- meta 里没有 `source` 列时 `SampleBundle.source_labels` 返回 `unknown` 而不是猜
  （`schema.py:140-145`）。

## 4. npz / csv / parquet 键列约定

实现：`src/cloudwater_ml/data/adapters/array_adapter.py`。

### 4.1 npz

- 特征键 `feature__<变量名>`，目标键 `target__<变量名>`，元数据键 `meta__<列名>`
  （`_frame_from_npz`，:64-89；写入方 `synthetic.py:110-119`）；
- **也接受裸变量名键**：查找顺序是 `feature__name` → `target__name` → `name`
  （:69-71），meta 同理 `meta__col` → `col`（:78）；
- 找不到变量直接报错并列出全部可用键（:73-76），**不猜测、不填充**；
- 元数据是 object 数组时报错并提示用 `cloudwater synth` 重新生成
  （安全默认 `allow_pickle=False`，:82-87）。

### 4.2 csv / parquet

- 先读表头检查缺失列（csv `nrows=0`，:37-42；parquet :52-56），缺失即报错并列出实际列；
- 只按 `data.features + data.targets + data.meta_columns` 选列读入（csv `usecols`，
  :43；parquet 先读后选 :57-58），避免整表载入；
- parquet 需要 `pyarrow`，缺失时报 `pip install 'cloudwater-ml[parquet]'`（:48-51）；
- `time` 列解析失败时**保留原样**交给质量检查报（:110-114）。

抽样（npz/csv/parquet 共用，:92-101）：`sampling.stride` 逐行抽样 +
`sampling.max_samples` 等距上限。

## 5. NetCDF 维度对齐规则与 v1 限制

流程：`_open_dataset` → `_apply_subsets`（范围筛选+stride）→ `_common_dims` →
`_flatten`（`netcdf_adapter.py:202-215`）。

### 5.1 v1 限制：所选变量必须共享同一维度集

`_common_dims`（`netcdf_adapter.py:139-149`）取第一个变量的维度序作为基准，
逐个比较其余变量的维度**集合**；只要有一个变量的维度集不同，就抛：

```
所选变量维度不一致，无法对齐成 [samples, features]（v1 限制：各变量须共享同一维度集，
见 docs/data_schema.md）: {...}
```

含义：v1 不做跨变量的维度求和/插值对齐（例如某变量是 3 维 `[time, lat, lon]`、
另一变量是 4 维 `[time, lev, lat, lon]` 时无法直接组成矩阵）。可选解法：
用 `level_range` 收窄到单层、在数据源侧裁出同维度变量，或改用
`iter_batches` 分批处理后再自行对齐。对应测试：
`tests/test_data.py:344 test_netcdf_disjoint_variables_dims_rejected`。

### 5.2 展平规则

- 变量按基准维度序 `transpose(*dims)` 后 `.ravel()`，逐列 `np.column_stack`
  （`netcdf_adapter.py:152-159`）→ 一个「网格点 = 一行样本」；
- meta 由坐标 `meshgrid(indexing="ij")` 同步展平（:161-166），保证与 X/Y 逐行对齐；
- `sample_id` 形如 `<文件名stem>-00000000`（:177）。

### 5.3 坐标别名

`lat→latitude`、`lon→longitude`、`lev/plev→level`、`valid_time→time`
（`netcdf_adapter.py:30-36`）；配了 `*_range` 却没有对应坐标时报错并列出实际坐标
（:117-119）。

## 6. 质量检查六项与「默认只读报告」

入口 `run_quality_checks(bundle, data_cfg)`（`validation.py:111`），
默认检查集（`validation.py:115-116`）= `configs/data.yaml:26` 的列表：

| # | 检查项 | 做什么 | 输出要点 | 出处 |
|---|---|---|---|---|
| 1 | `missing` | 逐列 NaN/Inf 计数、缺失比例、min/max/mean/std；meta.time 的 NaT 比例 | `per_column`、`total_missing_fraction`；有缺失即 warning 且**默认不插补** | `validation.py:120-135` |
| 2 | `nan_inf` | X/Y 的 NaN、Inf 总数 | `passed = 全为 0` | `validation.py:137-145` |
| 3 | `extreme` | 目标列 `\|z\|>4` 离群点统计（**仅报告不裁剪**），常数列标注不适用 | `z_threshold=4.0`（:18）、`total_extreme` | `validation.py:40-61,:147-148` |
| 4 | `range` | 与 `data.quality.ranges` 配置区间比对越界数 | `violations`、`passed`；未配置 ranges 时全部 skipped | `validation.py:64-86,:150-153` |
| 5 | `dimensions` | X/Y 维数、dtype、空样本、meta 行数 | `problems` 列表 | `validation.py:155-164` + `schema.py:165` |
| 6 | `units` | 数据源单位 vs `KNOWN_VARIABLES` 登记单位 | `mismatches` / `units_unknown`；**只提示不自动换算** | `validation.py:89-108,:166-173` |

**默认只读**：`quality.report_only` 默认 `true`（`configs/data.yaml:25`，
`validation.py:183`），此时 `apply_cleaning` 直接拒绝执行
（`validation.py:230-231`：`RuntimeError("quality.report_only=true（默认）：拒绝清洗…")`，
测试 `tests/test_data.py:257`）。显式 `report_only: false` + `cleaning: {dropna: true, clip: true}`
才清洗，且每个动作写进 `cleaning.applied` 可追踪（`validation.py:237-257`），
清洗后 runner 会**重新跑一遍检查并复核**（`runner.py:172-177`）。

报告格式：`format_quality_report` 输出 Markdown（`validation.py:193`），
CLI 用 `--out` 落盘（`cli.py:100-103`），实验里写到
`outputs/experiment_XXX/logs/quality_report.md`（`runner.py:178-179`）。

## 7. NetCDF 内存守卫参数表

铁律见 `netcdf_adapter.py:3-11`：**禁止默认整载入内存**。

| 配置键 | 默认值 | 作用 | 代码位置 |
|---|---|---|---|
| `data.netcdf.time_range` | `null` | `[start, end]` 时间范围筛选（基于坐标位置切片，不读变量值） | `netcdf_adapter.py:107-120`、`_range_slice:91` |
| `data.netcdf.latitude_range` | `null` | `[min, max]` 纬度范围 | 同上 |
| `data.netcdf.longitude_range` | `null` | `[min, max]` 经度范围 | 同上 |
| `data.netcdf.level_range` | `null` | `[min, max]` 气压层范围（level 不参与 stride，保垂直结构完整） | `netcdf_adapter.py:122-123` |
| `data.netcdf.chunks` | `{time: 16}` | dask 可用才启用 xarray 分块，否则退回按需读取 | `netcdf_adapter.py:43-52,:64-66` |
| `data.netcdf.engine` | `auto` | `auto/netcdf4/h5netcdf/scipy` | `netcdf_adapter.py:60-63` |
| `data.netcdf.max_in_memory_gb` | `2.0` | 体积守卫：估算字节数超过即**拒绝加载**并给降载建议 | `_DEFAULT_MAX_GB` :40、`_guard_memory:217-226` |
| `data.sampling.stride` | `1` | 时间/纬度/经度抽样步长 | `netcdf_adapter.py:123-131` |
| `data.sampling.max_samples` | `null` | 样本量上限：先沿第一维抽样（:227-233），load 后再等距裁剪（:244-247） | |
| `iter_batches(cfg, batch_samples=100_000)` | — | 沿第一维（通常 time）分批产出 Bundle，单批驻留内存可控 | `loader.py:70`、`netcdf_adapter.py:256-271` |

守卫失败时的报错会直接给出可操作建议（`netcdf_adapter.py:222-226`）：
增大 `sampling.stride/max_samples`、收窄 `*_range`、或改用 `iter_batches`。
对应测试：`tests/test_data.py:328 test_netcdf_memory_guard`、
`:318 test_netcdf_iter_batches_equals_load`（分批结果与整载一致）。

`cloudwater inspect` 对 netcdf **默认不读数值**（只读 header/坐标/契约），
要读值需显式 `--quality`，且仍受内存守卫保护（`cli.py:47-50,:71-83`）。

## 8. 变量登记表 `KNOWN_VARIABLES`

`schema.py:43-59`，共 12 个预留变量（数据源不必齐全，缺失即缺省、不填充）：

| 变量 | 单位 | role | nonnegative |
|---|---|---|---|
| `t` 温度 | K | feature | — |
| `q` 比湿 | kg kg-1 | feature | — |
| `rh` 相对湿度 | % | feature | — |
| `u` / `v` 水平风速 | m s-1 | feature | — |
| `w` / `omega` 垂直速度 | Pa s-1 | feature | — |
| `p` 气压 | hPa | feature | — |
| `z` 位势/几何高度 | m | feature | — |
| `clwc` 云液态水含量 | kg kg-1 | **target** | 是 |
| `ciwc` 云冰水含量 | kg kg-1 | **target** | 是 |
| `cwp` 云水路径 | kg m-2 | **target** | 是 |

`nonnegative=True` 是**物理非负提示**，供质量检查与物理约束参考，
不会自动裁剪数据（`schema.py:35` 注释）。
