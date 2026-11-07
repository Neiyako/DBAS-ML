# 数学规格说明（Formulation）

> 本文是 CloudWater ML Platform v2.0 的**形式化规格**：把平台里用到的每一个数学模型、
> 损失、指标、约束与算法写成公式，并标注**实现位置**。
>
> 原则：**公式即代码**。凡是本文写出的式子，都能在标注的文件里找到对应实现；
> 凡是代码里没有的，本文不写。单位一律显式声明——本项目最危险的错误就是单位口径。

---

## 0. 符号表

| 符号 | 含义 | 单位 / 取值 |
|---|---|---|
| $n$ | 样本数 | — |
| $f$ | 输入特征维数 | — |
| $t$ | 目标个数 | — |
| $\mathbf{X}\in\mathbb{R}^{n\times f}$ | 特征矩阵 | 依变量 |
| $\mathbf{Y}\in\mathbb{R}^{n\times t}$ | 目标矩阵 | 依变量 |
| $\mathbf{y}_i\in\mathbb{R}^{t}$ | 第 $i$ 个样本的目标向量 | — |
| $\hat{\mathbf{y}}_i$ | 模型预测（**物理单位**） | 依变量 |
| $B$ | 批大小（batch size） | — |
| $A$ | 梯度累积步数 | 正整数 |
| $B_{\text{eff}}$ | 等效批大小 | $B\cdot A$ |
| $E$ | 训练轮数 | — |
| $\eta$ | 学习率 | — |
| $\lambda$ | 正则/权重衰减 | — |
| $\theta$ | 模型参数 | — |
| $z$ | 气压层高度（几何高度） | m |
| $\rho_{\text{air}}(z)$ | 空气密度廓线 | kg m⁻³ |
| $\mathrm{CLWC}(z)$ | 云液水含量 | **kg kg⁻¹**（比含水量）或 **kg m⁻³**（质量密度） |
| $\mathrm{CWP}$ | 云水路径 | kg m⁻² |
| $\Delta z_k$ | 第 $k$ 层厚度 | m |
| $p$ | 气压 | Pa（或 hPa） |

**单位纪律**：$\mathrm{CLWC}$ 的两种口径在公式上**不可互换**（见 §4.2）。
本仓库 `data/schema.py` 登记为 `kg kg-1`（比含水量）。

---

## 1. 数据与样本

### 1.1 样本载体

一个数据批次是

$$
\mathcal{B} = \bigl(\mathbf{X}_{[B\times f]},\ \mathbf{Y}_{[B\times t]},\ \mathbf{M}_{[B\times m]}\bigr)
$$

其中 $\mathbf{M}$ 是元数据表（`sample_id / time / latitude / longitude / level / source / source_origin / dataset`）。
实现：`data/schema.py::SampleBundle`。

### 1.2 空间场样本

对卷积类模型，一个样本是场而非向量：

| 模型 | 输入张量 | 输出张量 |
|---|---|---|
| CNN | $\mathbf{X}\in\mathbb{R}^{C\times H\times W}$ | $\hat{\mathbf{y}}\in\mathbb{R}^{t}$ |
| U-Net | $\mathbf{X}\in\mathbb{R}^{C\times H\times W}$ | $\hat{\mathbf{Y}}\in\mathbb{R}^{C_{\text{out}}\times H\times W}$ |
| ConvLSTM | $\mathbf{X}\in\mathbb{R}^{T\times C\times H\times W}$ | $\hat{\mathbf{Y}}\in\mathbb{R}^{T_{\text{out}}\times C_{\text{out}}\times H\times W}$ |

批维恒在最前：$[B,\dots]$。ConvLSTM 的标量读出为

$$
\hat{\mathbf{y}} = \mathrm{pool}_{\text{readout}}\bigl(\hat{\mathbf{Y}}\bigr),\qquad
\mathrm{pool}_{\text{last}}=\hat{\mathbf{Y}}[:,T_{\text{out}}-1]\ \text{对}(H,W)\text{取均值}
$$

再经（可选的）线性读出 $\mathbf{W}_r\in\mathbb{R}^{t\times C_{\text{out}}}$：$\hat{\mathbf{y}}\leftarrow\mathbf{W}_r\hat{\mathbf{y}}+\mathbf{b}_r$。
实现：`models/convlstm.py::ConvLSTM.predict / predict_field`。

### 1.3 数据划分（防时空泄漏）

设 $D=\{(x_i,y_i,\mathbf{m}_i)\}_{i=1}^{n}$，划分函数
$\pi:\{1,\dots,n\}\to\{\text{train},\text{val},\text{test}\}$ 必须满足互斥且完备：

$$
D_{\text{train}}\cap D_{\text{val}}=D_{\text{train}}\cap D_{\text{test}}=D_{\text{val}}\cap D_{\text{test}}=\varnothing,\qquad
|D_{\text{train}}|+|D_{\text{val}}|+|D_{\text{test}}|=n
$$

| 策略 | 判据 |
|---|---|
| `time` | $t_i\le T_{\text{train}}$ → train；$T_{\text{train}}<t_i\le T_{\text{val}}$ → val；$t_i>T_{\text{val}}$ → test |
| `year` | 按年份分组切 |
| `region` | 按经纬度矩形框切 |
| `source` | 按来源（real/simulated）切 |
| `random` | 均匀随机（**会泄漏时空自相关，仅用于链路验证**） |

互斥性由 `data/splitting.py::SplitResult.check_disjoint` 强制断言。

### 1.4 场景分组（防同源泄漏）

同一原始场景的派生样本必须整组留在同一集合。定义场景键

$$
\mathrm{scene\_id}_i = h\bigl(\text{dataset}_i,\ \text{time}_i,\ \text{level}_i,\ \text{lat}_i,\ \text{lon}_i\bigr)
$$

其中 $h$ 为 SHA-1 截断。要求：$\forall s,\ \pi$ 在 $\{i:\mathrm{scene\_id}_i=s\}$ 上取同一值。
实现：`data/sources.py::scene_ids`。

---

## 2. 归一化（只在训练集拟合）

### 2.1 标准化（Standard）

对第 $j$ 列，用**训练集**统计量：

$$
\mu_j=\frac{1}{|D_{\text{train}}|}\sum_{i\in D_{\text{train}}}x_{ij},\qquad
\sigma_j=\sqrt{\frac{1}{|D_{\text{train}}|}\sum_{i\in D_{\text{train}}}(x_{ij}-\mu_j)^2}
$$

$$
\tilde{x}_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j},\qquad
x_{ij}=\tilde{x}_{ij}\sigma_j+\mu_j
$$

$\sigma_j<\varepsilon$ 时取 $\sigma_j\leftarrow1$（常数特征不放大）。

### 2.2 最小-最大（MinMax）

$$
\tilde{x}_{ij}=\frac{x_{ij}-\min_j}{\max_j-\min_j},\qquad
x_{ij}=\tilde{x}_{ij}(\max_j-\min_j)+\min_j
$$

### 2.3 逆变换的一致性要求

指标必须在**物理单位**下计算：

$$
\hat{\mathbf{y}}^{\text{phys}} = \mathcal{T}_y^{-1}\bigl(\hat{\mathbf{y}}^{\text{norm}}\bigr)
$$

物理约束同理（§4.3）。实现：`data/preprocessing.py`（`StandardScaler/MinMaxScaler/NoScaler`）、
`experiments/runner.py::_torch_inverse_y`（把逆变换注入训练，使约束作用在物理量上）。

---

## 3. 模型

### 3.1 MLP

$$
\mathbf{h}^{(0)}=\mathbf{x},\quad
\mathbf{h}^{(\ell)}=\phi\bigl(\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)}+\mathbf{b}^{(\ell)}\bigr),\ \ell=1..L,
\quad
\hat{\mathbf{y}}=\psi\bigl(\mathbf{W}^{(L+1)}\mathbf{h}^{(L)}+\mathbf{b}^{(L+1)}\bigr)
$$

$\phi\in\{\mathrm{ReLU},\mathrm{GELU},\tanh\}$；$\psi$ 为输出激活：`linear` 或
$\psi(u)=\ln(1+e^{u})$（`softplus`，用于非负硬约束，§4.4）。

### 3.2 卷积 / U-Net

单层二维卷积：

$$
(\mathbf{K}*\mathbf{X})_{c',i,j}=\sum_{c}\sum_{u,v}K_{c',c,u,v}\,X_{c,\,i+u,\,j+v}+b_{c'}
$$

U-Net 的跳跃连接：设编码器第 $\ell$ 层特征 $\mathbf{E}^{(\ell)}$，解码器对应层

$$
\mathbf{D}^{(\ell)}=\mathrm{Conv}\Bigl(\bigl[\mathrm{Up}(\mathbf{D}^{(\ell+1)});\ \mathbf{E}^{(\ell)}\bigr]\Bigr)
$$

其中 $[\cdot;\cdot]$ 为通道拼接，$\mathrm{Up}$ 为双线性或转置卷积上采样。
要求 $H,W$ 可被 $2^{\text{depth}}$ 整除（否则显式报错，不静默裁剪）。

### 3.3 ConvLSTM

单元状态更新（$*$ 为二维卷积）：

$$
\begin{aligned}
\mathbf{i}_t&=\sigma(\mathbf{W}_{xi}*\mathbf{X}_t+\mathbf{W}_{hi}*\mathbf{H}_{t-1}+\mathbf{b}_i)\\
\mathbf{f}_t&=\sigma(\mathbf{W}_{xf}*\mathbf{X}_t+\mathbf{W}_{hf}*\mathbf{H}_{t-1}+\mathbf{b}_f)\\
\mathbf{o}_t&=\sigma(\mathbf{W}_{xo}*\mathbf{X}_t+\mathbf{W}_{ho}*\mathbf{H}_{t-1}+\mathbf{b}_o)\\
\mathbf{g}_t&=\tanh(\mathbf{W}_{xg}*\mathbf{X}_t+\mathbf{W}_{hg}*\mathbf{H}_{t-1}+\mathbf{b}_g)\\
\mathbf{C}_t&=\mathbf{f}_t\odot\mathbf{C}_{t-1}+\mathbf{i}_t\odot\mathbf{g}_t\\
\mathbf{H}_t&=\mathbf{o}_t\odot\tanh(\mathbf{C}_t)
\end{aligned}
$$

多步预报：以 $\mathbf{H}_T$ 为初态自回归生成 $T_{\text{out}}$ 步。

> **数学事实**（曾导致一个测试写错）：带偏置的单元在零输入零状态下
> $\mathbf{i}=\sigma(\mathbf{b}_i)\ne0$、$\mathbf{g}=\tanh(\mathbf{b}_g)\ne0$，
> 因此 $\mathbf{H}\ne0$；只有偏置为零时才成立 $\mathbf{H}=0$。

### 3.4 多任务

共享骨干 $g_\theta$ + 逐任务头 $h_{\psi_k}$：

$$
\hat{y}^{(k)}=\bigl(h_{\psi_k}\circ g_\theta\bigr)(\mathbf{x}),\qquad k=1..K
$$

总损失（§4.1）：$\mathcal{L}=\sum_{k}w_k\mathcal{L}_k$，$w_k$ 为逐任务权重（默认等权）。

---

## 4. 损失与物理约束

### 4.1 监督损失

$$
\mathcal{L}_{\text{MSE}}=\frac{1}{n t}\sum_{i=1}^{n}\sum_{j=1}^{t}\bigl(\hat{y}_{ij}-y_{ij}\bigr)^2
$$

$$
\mathcal{L}_{\text{MAE}}=\frac{1}{n t}\sum_{i,j}\bigl|\hat{y}_{ij}-y_{ij}\bigr|
$$

$$
\mathcal{L}_{\text{Huber}}=\frac{1}{nt}\sum_{i,j}
\begin{cases}
\frac12(\hat{y}_{ij}-y_{ij})^2, & |\hat{y}_{ij}-y_{ij}|\le\delta\\
\delta\,|\hat{y}_{ij}-y_{ij}|-\frac12\delta^2, & \text{otherwise}
\end{cases}
$$

### 4.2 CWP 柱积分（两种口径，不可混用）

**口径 A：CLWC 为比含水量（kg kg⁻¹）**

$$
\mathrm{CWP}\;[\mathrm{kg\,m^{-2}}]\;=\;\int_{z_b}^{z_t}\rho_{\text{air}}(z)\,\mathrm{CLWC}(z)\,\mathrm{d}z
\;\approx\;\sum_{k=1}^{K}\rho_k\,\mathrm{CLWC}_k\,\Delta z_k
$$

单位验证：$[\mathrm{kg\,m^{-3}}]\cdot[\mathrm{kg\,kg^{-1}}]\cdot[\mathrm{m}]=[\mathrm{kg\,m^{-2}}]$ ✓

气压坐标等价形式（静力平衡 $\mathrm{d}p=-\rho g\,\mathrm{d}z$）：

$$
\mathrm{CWP}=\frac{1}{g}\int_{p_t}^{p_b}\mathrm{CLWC}(p)\,\mathrm{d}p
\;\approx\;\frac{1}{g}\sum_k \mathrm{CLWC}_k\,\Delta p_k
$$

**口径 B：CLWC 为质量密度（kg m⁻³）**

$$
\mathrm{CWP}=\int_{z_b}^{z_t}\mathrm{CLWC}(z)\,\mathrm{d}z\;\approx\;\sum_k\mathrm{CLWC}_k\,\Delta z_k
$$

> **陷阱**：口径 B 下再乘 $\rho_{\text{air}}$ 会高估约 $\rho$ 倍（≈1.2×）。
> 实现对此**显式报错拦截**：`physics/integrals.py`（`water_content_convention` 判定口径，
> `cwp_integral` 按口径计算，口径与输入不符时抛 `UnitMismatchError`）。

数值示例（同一组数：$\mathrm{CLWC}=[10^{-3},2\times10^{-3}]$，$\Delta z=[100,100]$ m，$\rho=[1.2,1.1]$）：

| 口径 | 计算 | 结果 |
|---|---|---|
| kg/kg | $1.2\cdot10^{-3}\cdot100+1.1\cdot2\times10^{-3}\cdot100$ | **0.34 kg m⁻²** |
| kg/m³ | $10^{-3}\cdot100+2\times10^{-3}\cdot100$ | **0.30 kg m⁻²** |

### 4.3 组合损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \sum_{i=1}^{m}\lambda_i\,\mathcal{L}_{\text{phys},i}
$$

物理项作用在**物理单位**的预测上：$\mathcal{L}_{\text{phys},i}\bigl(\mathcal{T}_y^{-1}(\hat{\mathbf{y}}),\mathcal{B}\bigr)$。
逐分量记录 $\{\mathcal{L}_{\text{data}},\mathcal{L}_{\text{phys},i}\}$ 供诊断。
实现：`training/losses.py::CompositeLoss`（`component_tensors` 在设备上累加，每 epoch 一次同步）。

### 4.4 非负约束

$$
\mathcal{L}_{\text{nonneg}}=\frac{1}{nt}\sum_{i,j}\bigl(-y_{ij}\bigr)_{+},\qquad (u)_{+}=\max(0,u)
$$

**硬约束版本**（推荐，物理上恒成立）：令输出激活 $\psi=\mathrm{softplus}$，

$$
\hat{y}=\ln\bigl(1+e^{u}\bigr)>0\quad\forall u
$$

训练与推理用同一变换，因此预测必然非负（不是"尽量拉回"）。

### 4.5 预留约束（构造即抛，不实现）

| 约束 | 需要的额外输入 |
|---|---|
| CWP 积分一致性 | $\rho_{\text{air}}(z)$ 或 $\Delta p_k$、层厚、单位口径 |
| 水量守恒 | 收支方程形式、边界通量、$P/E$ 单位 |
| 垂直廓线结构 | 层坐标、$T$/$T_d$ 廓线、LCL/对流层顶定义 |
| 大气水循环 | 相态转换项、水汽输送通量、检验时间窗 |

`requirements()` 逐条列出缺失前置条件；`PhysicsValidator` 对它们输出 `NOT_IMPLEMENTED`。

---

## 5. 优化

### 5.1 参数更新

AdamW（$\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$）：

$$
\mathbf{m}_k\leftarrow\beta_1\mathbf{m}_{k-1}+(1-\beta_1)\mathbf{g}_k,\quad
\mathbf{v}_k\leftarrow\beta_2\mathbf{v}_{k-1}+(1-\beta_2)\mathbf{g}_k^2
$$

$$
\hat{\mathbf{m}}_k=\frac{\mathbf{m}_k}{1-\beta_1^k},\quad
\hat{\mathbf{v}}_k=\frac{\mathbf{v}_k}{1-\beta_2^k}
$$

$$
\theta_k\leftarrow\theta_{k-1}-\eta\left(\frac{\hat{\mathbf{m}}_k}{\sqrt{\hat{\mathbf{v}}_k}+\epsilon}+\lambda\,\theta_{k-1}\right)
$$

（解耦权重衰减放在更新项外，即 AdamW。）

### 5.2 梯度累积

设一个 epoch 有 $N_b$ 个 micro-batch，累积 $A$ 步。第 $k$ 个 micro-batch 的损失按 $1/A$ 缩放：

$$
\mathbf{g}_{\text{acc}}=\frac{1}{A}\sum_{j=1}^{A}\nabla_\theta\mathcal{L}\bigl(\theta;\mathcal{B}_{k+j}\bigr),\qquad
\theta\leftarrow\theta-\eta\,\mathrm{Update}(\mathbf{g}_{\text{acc}})
$$

**尾部处理**：若 $N_b \bmod A = P \ne 0$，最后 $P$ 个 micro-batch 的梯度按

$$
\mathbf{g}_{\text{tail}}=\sum_{j=1}^{P}\nabla_\theta\mathcal{L}/\bigl(A\bigr)
\ \longrightarrow\
\tilde{\mathbf{g}}_{\text{tail}}=\frac{A}{P}\,\mathbf{g}_{\text{tail}}
=\frac{1}{P}\sum_{j=1}^{P}\nabla_\theta\mathcal{L}
$$

修正后再做一次更新，**保证这些样本不被丢弃**（否则等效批大小悄悄变小）。
实现：`training/trainer.py::_run_epoch` 尾部 flush 分支。

### 5.3 梯度裁剪

$$
\mathbf{g}\leftarrow\mathbf{g}\cdot\min\Bigl(1,\ \frac{c}{\|\mathbf{g}\|_2+\epsilon}\Bigr),\qquad \|\mathbf{g}\|_2=\Bigl(\sum_k\|\mathbf{g}^{(k)}\|_2^2\Bigr)^{1/2}
$$

AMP 下**必须先 `unscale_` 再裁剪**（否则按缩放后的梯度算范数，阈值失效）。

### 5.4 混合精度（AMP）

前向权重 $\theta$ 保持 fp32，算子内部用低精度（fp16/bf16），损失缩放 $S$：

$$
\mathcal{L}_S=S\cdot\mathcal{L},\qquad \mathbf{g}_S=\nabla_\theta\mathcal{L}_S
$$

- **fp16 需要 GradScaler**（$S$ 动态调整：溢出则减半、连续成功则倍增），
  否则梯度下溢为 0；
- **bf16 不需要**（指数位宽与 fp32 相同）；
- 验证与指标计算**关闭 autocast**（用 fp32），避免指标数值受低精度影响。

### 5.5 学习率调度

$$
\eta_k=\eta_{\max}\cdot\frac12\Bigl(1+\cos\frac{k\pi}{K}\Bigr)\quad(\text{cosine})
$$

### 5.6 早停

监控量 $m_k$（默认验证损失）。若 $m_k < m_{\text{best}}-\delta$ 则更新 best 并重置耐心；
否则计数 $+1$，连续 `patience` 次未改善即停。

---

## 6. 评价指标

对目标列 $j$（或全部元素展平的 micro 汇总），令 $e_i=\hat{y}_i-y_i$，$\bar{y}=\frac1n\sum y_i$：

$$
\mathrm{MAE}=\frac1n\sum_{i=1}^{n}|e_i|,\qquad
\mathrm{MSE}=\frac1n\sum_{i=1}^{n}e_i^2,\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
$$

$$
\mathrm{Bias}=\frac1n\sum_{i=1}^{n}(\hat{y}_i-y_i)\quad(>0\text{ 表示高估})
$$

$$
R^2=1-\frac{\sum_i(\hat{y}_i-y_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

$$
\mathrm{NSE}=1-\frac{\sum_i(\hat{y}_i-y_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

> 注：NSE 与 $R^2$ 在本定义下**数值相同**（均以观测均值为基准）。
> 保留 NSE 是为了水文/资源领域的口径对齐，不是新信息。

$$
r=\frac{\sum_i(\hat{y}_i-\bar{\hat{y}})(y_i-\bar{y})}
{\sqrt{\sum_i(\hat{y}_i-\bar{\hat{y}})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}
$$

**退化保护**：$\sum_i(y_i-\bar y)^2<\varepsilon$（目标近似常数）时 $R^2=\mathrm{NSE}=\text{None}$；
$r$ 的分母为 0 时亦为 None。值一律 JSON 安全（NaN/Inf → None）。

实现：`evaluation/metrics.py`（唯一实现处）。

---

## 7. 不确定性（分割共形）

### 7.1 区间构造

给定校准集 $\mathcal{C}$（本项目用**验证集**），残差 $r_i=|\hat{y}_i-y_i|$，取

$$
q=Q_{1-\alpha}\bigl(\{r_i\}_{i\in\mathcal{C}}\bigr)
$$

则测试样本的预测区间

$$
\hat{C}(\mathbf{x})=\bigl[\ \hat{y}(\mathbf{x})-q,\ \ \hat{y}(\mathbf{x})+q\ \bigr]
$$

**有限样本覆盖保证**：若校准集与测试集可交换（同分布），则

$$
P\bigl(Y\in\hat{C}(\mathbf{x})\bigr)\ \ge\ 1-\alpha-\frac{1}{|\mathcal{C}|+1}
$$

### 7.2 可靠性指标

$$
\text{Coverage}=\frac1n\sum_{i=1}^{n}\mathbb{1}\bigl[\hat{y}_{lo,i}\le y_i\le\hat{y}_{hi,i}\bigr]
$$

$$
\text{MPIW}=\frac1n\sum_{i=1}^{n}\bigl(\hat{y}_{hi,i}-\hat{y}_{lo,i}\bigr)
$$

$$
\text{CWC}=\text{MPIW}\cdot\bigl(1-\text{Coverage}\bigr)\quad(\text{越小越好})
$$

### 7.3 CRPS（高斯预测分布）

对预测分布 $\mathcal{N}(\mu,\sigma^2)$ 与观测 $y$，令 $z=(y-\mu)/\sigma$：

$$
\mathrm{CRPS}=\sigma\Bigl[\,z\bigl(2\Phi(z)-1\bigr)+2\varphi(z)-\frac{1}{\sqrt{\pi}}\,\Bigr]
$$

其中 $\Phi$ 为标准正态 CDF、$\varphi$ 为其 PDF。**无覆盖率保证**的方法
（正态近似）在报告里必须显式标注 `guarantee: NONE`。

实现：`evaluation/uncertainty.py`。

---

## 8. 真实/仿真混合采样

设训练集按来源分为 $D_{\text{real}}$、$D_{\text{sim}}$，目标真实比例 $\rho\in[0,1]$，
总配额 $n_{\text{sel}}$。

**按比例**：

$$
n_{\text{real}}=\operatorname{round}(\rho\,n_{\text{sel}}),\qquad n_{\text{sim}}=n_{\text{sel}}-n_{\text{real}}
$$

若可用样本不足则名额在两类间**转移**（不丢样本、不改标签）：

$$
n'_{\text{sim}}=\min\bigl(|D_{\text{sim}}|,\ n_{\text{sim}}+(n_{\text{real}}-|D_{\text{real}}|)_{+}\bigr)
$$

实际比例 $\hat\rho=n_{\text{real}}^{\text{take}}/(n_{\text{real}}^{\text{take}}+n_{\text{sim}}^{\text{take}})$
必须如实记录；$\hat\rho\ne\rho$ 时写 notes（**不伪造标签**）。

**按权重**：$w_k>0$，$n_k=\operatorname{round}\bigl(n_{\text{sel}}\,w_k/\sum_j w_j\bigr)$。

**分层**：按层 $s$（年/季/区）分层，每层内按 $\rho$ 抽样：

$$
n^{(s)}_{\text{real}}=\operatorname{round}\bigl(\rho\,n^{(s)}\bigr)
$$

**不可解析来源**（全为 `unknown`）：保持全部样本（$\hat\rho$ 记 None），
**绝不因为筛 real/simulated 而把训练集清空**。

实现：`data/sources.py::MixedSampler`。

---

## 9. 性能与资源

### 9.1 吞吐

$$
\text{SPS}=\frac{n\cdot E}{T_{\text{wall}}}\ \ [\text{samples/s}]
$$

### 9.2 三段耗时分解

$$
T_{\text{wall}}\approx T_{\text{data}}+T_{\text{transfer}}+T_{\text{compute}}
$$

- $T_{\text{data}}$：从 DataLoader 取下一批的等待（含 worker 排队）；
- $T_{\text{transfer}}$：主机→设备拷贝（CUDA 下 pinned + non_blocking）；
- $T_{\text{compute}}$：前向 + 反向 + 优化器步进。

瓶颈判据：$\frac{T_{\text{data}}}{T_{\text{wall}}}>0.5$ → 数据供给瓶颈；
$\frac{T_{\text{compute}}}{T_{\text{wall}}}>0.7$ → 算力瓶颈。

### 9.3 显存估算（启发式，用于给建议）

可用预算 $M_{\text{avail}}=0.6\cdot M_{\text{free}}$；每样本占用粗估

$$
M_{\text{per-sample}}\approx\max\bigl(0.05\ \mathrm{MiB},\ f\times0.004\ \mathrm{MiB}\bigr)
$$

起步批大小

$$
B_{\text{rec}}=\mathrm{clip}\Bigl(\Bigl\lfloor\frac{M_{\text{avail}}\cdot1024}{M_{\text{per-sample}}}\Bigr\rfloor,\ 32,\ 1024\Bigr)
$$

$B_{\text{rec}}<64$ 时建议 $A=4$ 梯度累积维持等效批量。
**所有输入均为运行时实测**（不写死任何显卡型号/显存）。

实现：`benchmark/runner.py::recommend_config`。

### 9.4 峰值显存

$$
M_{\text{peak}}=\max_{k}\ \mathrm{reserved}(k)\quad(\text{CUDA: } \mathtt{max\_memory\_reserved})
$$

---

## 10. 复现性

同种子 + 同数据 + 同配置 + 同环境 ⇒ **统计一致**（不承诺位级一致）：

- 随机流：`python random` / `numpy` / `torch` / `torch.cuda` 全部播种；
- 续训恢复 `rng_state`，使 $\text{shuffle}$ 顺序与 dropout 掩码连续；
- GPU 非确定性算子（原子加、部分卷积反向）**无法保证**位级一致，
  `torch.use_deterministic_algorithms(True, warn_only=True)` 只告警。

---

## 11. 公式 ↔ 代码对照表

| 公式 | 实现 |
|---|---|
| §1.1 SampleBundle | `data/schema.py::SampleBundle` |
| §1.2 场输入契约 | `models/{cnn,unet,convlstm}.py` |
| §1.3 划分与互斥 | `data/splitting.py::split_indices / check_disjoint` |
| §1.4 场景分组 | `data/sources.py::scene_ids` |
| §2 归一化 | `data/preprocessing.py` |
| §3.1 MLP | `models/mlp.py::MLPNet` |
| §3.2 U-Net | `models/unet.py::UNetNet` |
| §3.3 ConvLSTM | `models/convlstm.py::ConvLSTMCell` |
| §3.4 多任务 | `models/multitask.py::MultiTaskNet` |
| §4.1 监督损失 | `training/losses.py::get_loss` |
| §4.2 CWP 积分 | `physics/integrals.py::cwp_integral` |
| §4.3 组合损失 | `training/losses.py::CompositeLoss` |
| §4.4 非负约束 | `physics/constraints.py::NonNegativityConstraint`；硬约束见 `models/mlp.py::_OUTPUT_ACTIVATIONS` |
| §5.1 AdamW | `training/optimizer.py` |
| §5.2 梯度累积 | `training/trainer.py::_run_epoch / _apply_gradients` |
| §5.3 裁剪 | `training/trainer.py::_apply_gradients` |
| §5.4 AMP | `hardware/precision.py` |
| §5.5 调度 | `training/scheduler.py` |
| §5.6 早停 | `training/callbacks.py::EarlyStopping` |
| §6 指标 | `evaluation/metrics.py::regression_metrics` |
| §7.1 共形区间 | `evaluation/uncertainty.py::conformal_interval` |
| §7.3 CRPS | `evaluation/uncertainty.py::crps` |
| §8 混合采样 | `data/sources.py::MixedSampler` |
| §9.2 三段耗时 | `hardware/profiling.py::Profiler` |
| §9.3 配置建议 | `benchmark/runner.py::recommend_config` |
| §10 随机流 | `utils/reproducibility.py` |

---

## 12. 已明确的局限（不写成没做的事）

- **TB 级流式训练已通，但分组泛化/质量报告仍需全量统计**：流式模式不产出这两项。
- **CWP 积分一致性、水量守恒、垂直廓线、水循环**四项为预留（构造即抛），
  因为缺少 $\rho_{\text{air}}(z)$ / $\Delta p$ / 边界条件等前置输入。
- **CRPS 仅实现高斯解析形式**；分位数回归/深度集成的 CRPS 需采样，尚未接入。
- **不确定性的覆盖保证依赖可交换性**：时间外推/空间外推场景下覆盖率会低于名义值，
  这是泛化问题，不是区间方法的问题——报告里必须同时给出划分策略。
