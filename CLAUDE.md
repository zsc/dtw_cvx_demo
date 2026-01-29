# SPEC：基于 Whisper 特征的凸优化 DTW 对齐 + 单调时间映射函数生成（不做静音裁剪）

## 0. 背景与目标

你有两段 WAV（同一段文字的不同 TTS），内容基本一致但语速/停顿/韵律不同。希望在 **Whisper 将 wav 转成的 feature matrix** 上做 DTW 对齐，并进一步生成一个 **单调非减的时间扭曲函数**：

- 输入：两段特征序列 `X[1..T1]`、`Y[1..T2]`
- 输出：单调函数 `t2 = f(t1)`（可用于把音频1的任意时间点映射到音频2）

**约束**：不做“去掉长静音”（不裁剪、不删除静音帧）。

本 SPEC 设计两阶段：
1) **凸优化版 DTW**：把 DTW 写成 **最小费用流（Min-Cost Flow）/线性规划（LP）**，求一条最优单调路径  
2) **单调函数生成**：把离散路径拟合为平滑的单调函数（凸二次规划 QP），输出连续可插值的 `f(t1)`

---

## 1. 术语与符号

- Whisper 特征序列：
  - `X = [x_1, ..., x_T1]`, `x_i ∈ R^d`
  - `Y = [y_1, ..., y_T2]`, `y_j ∈ R^d`
- 帧索引到归一化时间：
  - `u_i = (i-1)/(T1-1) ∈ [0,1]`
  - `v_j = (j-1)/(T2-1) ∈ [0,1]`
- 输出单调映射：
  - 离散：`v[i]`（对每个 `u_i` 的映射值）
  - 连续：通过 `(u_i, v[i])` 分段线性插值得到 `f_norm(u)`，再映射到真实秒数 `f(t1)`

---

## 2. 输入/输出规范

### 2.1 输入
- `wav1_path: str`
- `wav2_path: str`
- `feature_mode: str`：`"whisper_encoder"`（优先）或 `"log_mel"`（可选）
- 超参数（可配置，见后文默认值）：
  - `dist: str`：`"cosine"` 或 `"l2sq"`
  - `gamma_time: float`：时间偏离惩罚权重
  - `band_radius: float or None`：Sakoe-Chiba band 宽度（归一化尺度），例如 `0.08`
  - `step_penalty: dict`：水平/垂直/斜步惩罚
  - `qp_alpha, qp_beta: float`：QP 平滑项权重
  - `slope_min, slope_max: float or None`：可选斜率范围约束（防止过度拉伸/压缩）

> 注意：本 SPEC **不包含静音裁剪/删除**。允许保留静音帧参与对齐。

### 2.2 输出
- `mapping` 对象（建议 JSON 可序列化）：
  - `u: float[T1]`：归一化时间网格（0..1）
  - `v: float[T1]`：单调拟合后的映射（0..1）
  - `path: list[(i,j)]`：DTW 最优路径（1-based 或 0-based，需明确）
  - `config: dict`：实际使用的超参数
- 方法：
  - `warp_time(t1_seconds) -> t2_seconds`
  - `warp_frame(i) -> j_float`（可选）
  - （可选）`inverse_warp_time(t2) -> t1`：用交换 X/Y 重新跑一遍，或对 `(v,u)` 插值

---

## 3. 依赖与实现建议

### 3.1 Python 依赖建议
- 特征：
  - `openai-whisper` 或 `faster-whisper`（任选其一）
  - `numpy`
- 最小费用流（推荐）：
  - `ortools`（`SimpleMinCostFlow`）**优先**：高效、稳定
  - 备选：`networkx`（慢，不建议长序列）
- QP 求解：
  - `cvxpy` + `osqp`（或 `ecos`）
- 插值：
  - `numpy.interp` 或 `scipy.interpolate`

### 3.2 数值约定
- OR-Tools 的 min-cost-flow 成本通常需要 **整数**：
  - 将浮点代价 `C` 乘以 `cost_scale`（例如 `1e6`）并四舍五入到 int
- 所有向量/矩阵尽量使用 `float32` 以节省内存，但 QP 求解可用 `float64`

---

## 4. 特征抽取（Whisper → Feature Matrix）

### 4.1 `feature_mode="whisper_encoder"`（推荐）
- 流程：
  1) 读取 wav，重采样到 16k
  2) 计算 log-mel（Whisper 标准）
  3) 输入 Whisper Encoder，取 encoder 的时间序列输出作为特征矩阵
- 输出：
  - `X: (T1, d)`, `Y: (T2, d)`
  - 同时记录每帧对应的秒数 `frame_hop_sec`（或用归一化时间避免依赖）

> 说明：Whisper encoder 通常会下采样时间维度（例如 conv stride），导致特征帧率 ≠ mel 帧率。实现需从模型配置或实际张量长度推导每帧时间间隔，或者统一使用归一化 `u,v`。

### 4.2 `feature_mode="log_mel"`（可选）
- 直接使用 log-mel 的每帧向量作为特征（维度 ~80）

### 4.3 特征归一化（必须）
- 对每帧做 L2 normalize（用于余弦距离）：
  - `x_i ← x_i / (||x_i|| + eps)`，`y_j`同理
- 不做静音裁剪；不删除任何帧

---

## 5. 局部匹配代价矩阵 C 的定义

对每对帧 `(i,j)` 定义代价：

### 5.1 内容距离项
- 余弦距离（推荐）：
  - `C_content(i,j) = 1 - dot(x_i, y_j)`
- 或平方 L2：
  - `C_content(i,j) = ||x_i - y_j||^2`

### 5.2 时间偏离惩罚（可选但推荐）
用于防止路径过度偏离对角线（同一文本通常整体速率不会极端）：
- `C_time(i,j) = | i/(T1-1) - j/(T2-1) |`
- 总代价：
  - `C(i,j) = C_content(i,j) + gamma_time * C_time(i,j)`

### 5.3 Band 限制（强烈推荐，提升可用性）
仅计算满足下式的 `(i,j)`：
- `| i/(T1-1) - j/(T2-1) | <= band_radius`
- `band_radius=None` 表示全矩阵（长音频可能不可承受）

---

## 6. 凸优化 DTW：最小费用流（LP 等价形式）

### 6.1 网格图定义
节点：所有允许的 `(i,j)`（受 band 限制后）

允许的三种转移边：
- 右移（水平步）：`(i,j) -> (i, j+1)`
- 下移（垂直步）：`(i,j) -> (i+1, j)`
- 斜移（对角步）：`(i,j) -> (i+1, j+1)`

### 6.2 变量
- 对每条边 `e` 定义流量 `f_e >= 0`

### 6.3 约束（线性）
- 单位流从源到汇：
  - 源 `s=(0,0)` 供给 `+1`
  - 汇 `t=(T1-1, T2-1)` 需求 `-1`
  - 其他节点流守恒
- 容量（推荐但非必须）：
  - `0 <= f_e <= 1`

### 6.4 目标函数（线性）
最小化总费用：
- `min Σ_e cost(e) * f_e`

边的费用定义（推荐“到达节点费用 + 步型惩罚”）：
- 对边 `e: (i,j)->(i',j')`：
  - `cost(e) = C(i',j') + step_penalty[type(e)]`
- `step_penalty` 默认建议：
  - `diag: 0`
  - `horiz: lambda_h`
  - `vert:  lambda_v`
  - 通常 `lambda_h=lambda_v`，鼓励走对角以减少过度拉伸

### 6.5 解的性质（必须满足）
- 该 LP 是网络流问题，约束矩阵具备整数性性质：最优解会给出 0/1 的边流
- 从所有 `f_e=1` 的边可恢复一条单调路径 `P`

### 6.6 实现要点（OR-Tools SimpleMinCostFlow 推荐）
- 需要把节点映射为整数 id：
  - `node_id = i * T2 + j`（或对 band 子图用字典映射）
- 添加边：
  - `AddArcWithCapacityAndUnitCost(u, v, cap=1, cost=int(round(cost_float * cost_scale)))`
- 设置 supply：
  - `SetNodeSupply(source_id, +1)`
  - `SetNodeSupply(sink_id, -1)`
- 求解后读取：
  - 遍历 arcs，取 `Flow(arc) == 1` 的边，重建路径

---

## 7. 从 DTW 路径生成“原始映射观测” hat_v

DTW 路径 `P = [(i_k, j_k)]`（单调不减）

对每个 `i` 收集所有匹配到的 `j`：
- `J(i) = { j_k | (i_k == i) }`

定义原始映射：
- `hat_j(i) = median(J(i))`
  - 若 `J(i)` 为空（理论上不应出现；band 太窄可能出现），做邻近填充：
    - `hat_j(i) = hat_j(i-1)` 或线性插值修复
- 归一化：
  - `hat_v[i] = hat_j(i)/(T2-1)` （0-based）
  - 或 `hat_v[i] = (hat_j(i)-1)/(T2-1)`（1-based）

权重（用于 QP）：
- `w[i] = |J(i)|`（匹配次数越多权重越大）
- 或简化：`w[i]=1`

---

## 8. 凸 QP：拟合平滑单调函数 v[i]

目标：得到 `v[0..T1-1]`，满足：
- 单调非减：`v[i+1] >= v[i]`
- 边界固定：`v[0]=0`, `v[T1-1]=1`
- 贴合 `hat_v`
- 平滑（可控）

### 8.1 优化变量
- `v ∈ R^{T1}`

### 8.2 约束（线性）
- `v[0] == 0`
- `v[T1-1] == 1`
- `v[i+1] >= v[i]` for `i=0..T1-2`

可选斜率范围约束（线性）：
- 设 `Δu = 1/(T1-1)`，则
  - `slope_min*Δu <= v[i+1]-v[i] <= slope_max*Δu`
- 若不设，跳过该约束

### 8.3 目标函数（凸二次）
推荐组合（默认都开）：
- 贴合项：
  - `Σ_i w[i] * (v[i] - hat_v[i])^2`
- 一阶平滑：
  - `α * Σ_i (v[i+1]-v[i])^2`
- 二阶平滑（曲率惩罚）：
  - `β * Σ_i (v[i+2]-2v[i+1]+v[i])^2`

整体：
- `min  Σ w(i)(v-hat_v)^2 + α||D1 v||^2 + β||D2 v||^2`

### 8.4 求解器
- `cvxpy`：
  - 目标为二次凸，约束线性，推荐用 `OSQP`
- 注意：
  - `T1` 很大时 QP 仍可能变慢；但通常可接受（比全矩阵 LP 小很多）
  - 若需要进一步提速，可只在关键点（例如每 N 帧）拟合 v，再插值回全长（可作为优化项）

---

## 9. 连续时间映射函数 f(t1)

### 9.1 归一化函数
得到 `u[i]=i/(T1-1)` 与 `v[i]` 后，定义分段线性插值：
- `f_norm(u_query) = interp(u_query; u[], v[])`

### 9.2 秒级映射
设两段音频时长（秒）：
- `D1` = wav1 秒数
- `D2` = wav2 秒数

则：
- `t2 = f(t1) = D2 * f_norm(t1 / D1)`

### 9.3 输出 API（必须实现）
- `warp_time(t1: float) -> float`
  - 输入限定：`t1` 超出 `[0, D1]` 时，做 clamp 到边界
- `warp_u(u: float) -> float`（可选）
- 导出 `u,v` 数组，便于可视化与复现

---

## 10. 超参数默认值（建议）

> 实现需允许 CLI 参数覆盖

- `dist = "cosine"`
- `gamma_time = 0.1`（按实际特征尺度调，范围建议 `[0, 1]`）
- `band_radius = 0.08`（同文本 TTS 通常足够；若语速差很大可增大到 `0.15`）
- `step_penalty = { "diag": 0.0, "horiz": 0.2, "vert": 0.2 }`
- `cost_scale = 1_000_000`（OR-Tools 整数化）
- QP：
  - `qp_alpha = 1e-2`
  - `qp_beta  = 1e-2`
- `slope_min = None`, `slope_max = None`（默认不限制）

---

## 11. 复杂度与工程限制

- 若不做 band，图规模约 `O(T1*T2)`，可能不可用（内存/时间爆炸）
- **必须实现 band 模式**，并在 band 导致图不连通时：
  - 自动扩大 band 或回退到更宽 band（建议策略：每次 *1.5，最多尝试 K 次）
- 路径重建必须保证从源到汇连续、单调

---

## 12. 边界与异常处理（必须覆盖）

1) **band 太窄导致无可行路径**
   - 自动增大 `band_radius` 重试（或直接报错并给出建议）
2) **特征长度过短**
   - `T1<2` 或 `T2<2`：直接返回线性映射
3) **QP 不收敛/不可行**
   - 若不可行：检查是否设置了矛盾的 `slope_min/max`
   - 回退策略：去掉 slope 约束、仅保留单调 + 一阶平滑
4) **保持不裁剪静音**下可能出现“前后静音不一致”
   - 默认仍强制 `(0,0)->(T1-1,T2-1)` 对齐
   - （可选增强项）支持 `boundary_mode="relaxed"`：添加超级源/汇连接边界区域（带小惩罚），允许对齐起止点在边界附近漂移

---

## 13. 验收标准（Acceptance Criteria）

实现完成后必须满足：

1) **单调性**：`v[i+1] >= v[i]` 对所有 i 成立
2) **边界**：`v[0]=0`, `v[-1]=1`
3) **路径合法**：
   - `P` 从 `(0,0)` 到 `(T1-1,T2-1)`
   - 相邻点差分只能是 `(1,0),(0,1),(1,1)`
4) **可复现**：同输入同超参数输出一致
5) **可调用**：`warp_time()` 在 `[0,D1]` 上输出 `[0,D2]`，且单调非减
6) **不做静音裁剪**：实现中不允许删除帧/裁剪时间轴（可做归一化，但不可丢帧）

---

## 14. 建议的代码结构（文件/模块）

- `features.py`
  - `extract_whisper_features(wav_path, mode) -> (feat, duration_sec)`
- `cost.py`
  - `compute_cost_band(X, Y, gamma_time, band_radius, dist) -> iterator_of_cost_entries`
- `mincost_flow_dtw.py`
  - `solve_dtw_mincost_flow(cost_entries, T1, T2, step_penalty, cost_scale) -> path`
- `path_to_mapping.py`
  - `path_to_hat_v(path, T1, T2) -> (hat_v, w)`
- `monotone_qp.py`
  - `fit_monotone_smooth(u, hat_v, w, alpha, beta, slope_min, slope_max) -> v`
- `warp.py`
  - `build_warp(u, v, D1, D2) -> object with warp_time()`
- `cli.py`
  - 解析参数，串联 pipeline，导出 JSON（含 u,v,path,config）

---

## 15. 最小伪代码（端到端）

```python
X, D1 = extract_whisper_features(wav1, mode="whisper_encoder")
Y, D2 = extract_whisper_features(wav2, mode="whisper_encoder")

X = l2_normalize(X); Y = l2_normalize(Y)

cost_entries = compute_cost_band(X, Y, gamma_time, band_radius, dist)
path = solve_dtw_mincost_flow(cost_entries, T1=len(X), T2=len(Y),
                             step_penalty=step_penalty, cost_scale=1e6)

hat_v, w = path_to_hat_v(path, T1=len(X), T2=len(Y))
u = np.linspace(0, 1, T1)

v = fit_monotone_smooth(u, hat_v, w, alpha=qp_alpha, beta=qp_beta,
                        slope_min=slope_min, slope_max=slope_max)

warp = build_warp(u, v, D1, D2)
t2 = warp.warp_time(t1)
````

---

## 16. 输出格式建议（JSON）

```json
{
  "u": [0.0, 0.0007, ..., 1.0],
  "v": [0.0, 0.0006, ..., 1.0],
  "path": [[0,0],[1,1],...,[T1-1,T2-1]],
  "durations": {"D1": 12.34, "D2": 13.02},
  "config": {
    "feature_mode": "whisper_encoder",
    "dist": "cosine",
    "gamma_time": 0.1,
    "band_radius": 0.08,
    "step_penalty": {"diag":0.0,"horiz":0.2,"vert":0.2},
    "qp_alpha": 0.01,
    "qp_beta": 0.01
  }
}
```

---

## 17. 额外建议（非必须但推荐）

* 输出一个对齐可视化：`(u, v)` 曲线 + 路径热图（用于 debug）
* 提供 `--save-intermediate` 保存 `X,Y` 的长度与统计量，便于定位 band/超参数问题
* 在不裁剪静音的前提下，如果出现严重“静音拖拽”，优先通过：

  * 增大 `gamma_time`（更贴对角线）
  * 增大 `lambda_h/lambda_v`（减少水平/垂直拉伸）
  * 使用 `boundary_mode="relaxed"`（允许起止点漂移）
    来改善
