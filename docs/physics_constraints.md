# 物理约束体系（CloudWater ML Platform v2.0）

> 核心原则（`configs/training.yaml` 的 `physics` 段注释）：
> **物理约束默认全部关闭，未验证的物理假设不得强制写入模型。**
>
> v2.0 实现了独立的 `cloudwater_ml.physics` 包：可插拔接口、单位感知、
> 硬/软约束分离、逐分量记录、独立的一致性校验器。

---

## 1. 为什么单位是核心问题

变量登记表（`src/cloudwater_ml/data/schema.py`）给出的单位：

| 变量 | 登记单位 | 物理量类型 |
|---|---|---|
| `clwc` 云液态水含量 | `kg kg-1` | **比量纲**（每千克空气含有的液态水质量） |
| `ciwc` 云冰水含量 | `kg kg-1` | **比量纲** |
| `cwp` 云水路径 | `kg m-2` | **柱积分量**（单位截面气柱的总水质量） |

陷阱：`clwc` 是「质量比」不是「密度」。直接对高度积分：

```
错误：CWP [kg/kg × m] ≠ kg/m²
```

量纲就错了。正确路径必须先换算成体积密度再积分：

```
正确：CWP [kg m-2] = ∫ clwc [kg kg-1] × ρ_air [kg m-3] dz [m]
```

等价的气压坐标写法：

```
CWP = ∫ clwc · (dp / g)        # dp/g 的单位是 kg m-2
```

**两种口径的数值差别**（本仓库用同一组数验证过）：

| 口径 | 公式 | 示例结果（clwc=[1e-3, 2e-3]，Δz=100 m，ρ=[1.2, 1.1]） |
|---|---|---|
| `kg/kg`（比含水量） | `∫ ρ·CLWC dz` | **0.34 kg/m²** |
| `kg/m³`（质量密度） | `∫ CLWC dz` | **0.30 kg/m²** |

在 `kg/m³` 口径下**再乘一次空气密度就会高估约 ρ 倍**；
`physics/integrals.py` 对此**显式报错拦截**，而不是默默算错。

---

## 2. 统一接口

```python
class PhysicsConstraint:
    name: str
    weight: float = 0.0        # 默认 0 = 关闭
    enabled: bool = True

    def compute(self, prediction, batch) -> Tensor: ...   # 标量惩罚
    def expects_units(self) -> dict[str, str]: ...        # 期望单位口径
    def requirements(self) -> list[str]: ...              # 还缺哪些前置条件
```

- `compute(prediction, batch)` 返回**标量惩罚**（越大越违例）；
- `batch` 可含 `Y` / `target_names` / `units` / `clwc_profile` / `layer_thickness` /
  `air_density` / `cwp_target`；
- **权重默认 0**：约束必须显式加权才生效（列出的权重全为 0 时框架会告警）。

### 硬约束 / 软约束

| 类型 | 机制 | 是否进损失 |
|---|---|---|
| `SoftConstraint` | `compute` 返回惩罚项 | 是，`L = L_data + Σ λ_i·L_i` |
| `HardConstraint` | `project()` 投影 / 重参数化（如 clamp、softplus） | 否（`compute` 恒 0） |

硬约束用于「结构上不可能违反」的量（如云水含量非负）：
直接用投影保证，而不是靠惩罚项慢慢学。

---

## 3. 可用约束

| 约束名 | 类型 | 说明 |
|---|---|---|
| `nonnegativity` | 软 | 对负预测惩罚 `mean(relu(-y))` |
| `unit_consistency` | 软 | 单位口径不符时报错（不默默算） |
| `nonnegativity_clamp` | 硬 | 把预测 clamp 到 ≥ 0 |
| `nonnegativity_softplus` | 硬 | 输出重参数化为 `softplus`，天然非负 |

> **硬约束怎么真正生效**：`clamp` 类投影若只作用于损失，会与推理不一致。
> 因此模型侧提供等价的输出层激活（`MLP(output_activation="softplus")`），
> 训练与推理**用同一个变换**；配置示例见 `configs/experiments/mlp_nonneg.yaml`。
> 实测：合成数据上 `negative_values` 检查由 **FAIL → PASS**
> （代价是 R² 从 0.976 降到 0.744，属「物理正确换精度」的显式取舍）。

### 预留约束（**构造即抛** `NotImplementedError`）

| 约束名 | 还缺什么 |
|---|---|
| `cwp_consistency` / `cwp_integral_consistency` | 空气密度廓线（或 `dp/g`）、层厚、单位口径确认 |
| `water_budget` | 水量收支边界条件（输入/输出通量） |
| `vertical_profile_structure` | 垂直廓线形状先验（层数、坐标、单调性定义） |
| `atmospheric_water_cycle` | 水循环闭合所需的完整变量集与边界 |

**刻意不实现**：单位口径与前置输入未确认前启用它们，等于把未验证的物理假设写进损失。
`requirements()` 会明确列出缺什么。

---

## 4. 与训练链路集成

```yaml
physics:
  constraints: [nonnegativity]
  nonnegativity_weight: 0.1     # 旧键
  # weights: {nonnegativity: 0.1, unit_consistency: 0.05}   # 新键：逐约束权重
```

`training/losses.py::build_constraints` **优先**使用 `physics` 包的新约束
（经 `physics/bridge.py` 适配到旧 `penalty()` 接口），新体系不可用时回退旧实现。

### 约束作用在**物理单位**上

标准化空间里的 z-score 不是物理量：把「低于均值」当成「负云水」是**科学错误**。
因此实验编排会把 Y 的反标准化变换注入训练配置
（`experiments/runner.py::_torch_inverse_y`），`CompositeLoss` 在计算物理惩罚前
先还原到物理单位；若无法还原则**告警**（而不是默默按标准化空间算）。

---

## 5. 物理一致性校验器（独立于损失）

```python
from cloudwater_ml.physics import PhysicsValidator, format_report_markdown

validator = PhysicsValidator(expected_units={"clwc": "kg kg-1"})
report = validator.validate(pred_tensor, {"Y": truth, "target_names": [...], "units": {...}})
print(format_report_markdown(report))
```

检查项：负值、量程、垂直积分一致性、变量单位、水量收支残差，
外加「预留约束」清单。

**四态输出，绝不含糊**：

| 状态 | 含义 |
|---|---|
| `PASS` | 该项检查通过 |
| `FAIL` | 确实违反 |
| `NOT_APPLICABLE` | 本次数据不适用（例如没有廓线数据） |
| `NOT_IMPLEMENTED` | 该检查尚未实现（预留约束） |

**总体状态不会在存在 `NOT_IMPLEMENTED` 时宣称 PASS。**

实验运行时会自动生成 `outputs/<exp>/logs/physics_report.md`，并在
`metrics.json` 的 `physics_report` 段保留结构化结果。

---

## 6. 诚实性约定

- 预留约束**构造即抛**，不提供「先跑起来再说」的假实现；
- 单位不符**报错**，不自动换算（换算需要人确认口径）；
- 硬约束与软约束**分开标注**，报告里能看出某项是投影保证还是惩罚项；
- 物理一致性报告的每一项都有状态，**未实现 ≠ 通过**。
