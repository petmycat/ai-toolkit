# TST V8 A1：Inverse-Rho 诊断轮实施计划

## 1. 目的

本计划用于实现并运行下一轮 **V8 A1-only 诊断实验**。

首轮 A1 已确认：

- 训练正常完成 200 steps；
- 4 个 virtual-token embeddings、36 层 Qwen `down_proj` rank-4 module LoRA、13 个 rank-2 tap adapters 均参与训练；
- 参数确实发生更新；
- 36 张训练图与 4 张 held-out 图的 split manifest 正确生成；
- 但 prediction-space 条件响应没有学到；
- 因此首轮 A1 checkpoint **不得进入 Phase B**。

本轮不更换 activator 架构，也不先做组件消融。优先验证一个更基础的问题：

> 首轮 A1 是否因为小 \(\rho\) 同时削弱了目标位移和目标方向梯度，导致完整 activator 没有得到公平强度的优化信号？

---

## 2. 首轮 A1 的最终诊断

### 2.1 已确认的失败

V8 A1 的 response target 为：

\[
T = B + \rho(Y-B),
\]

其中：

- \(A\)：active activator prediction；
- \(B\)：相同 token structure 下的 activator-bypass prediction；
- \(Y\)：diffusion reconstruction target；
- \(\rho\)：要求 activator 实现的目标方向响应系数。

首轮 schedule 为：

\[
\rho: 0.05 \rightarrow 0.20.
\]

但实际测得：

- 最大 \(\alpha\)：`0.00711`，出现在 step 136；
- 正 \(\alpha\) step 比例：`77.5%`；
- 最终 \(\alpha\)：`0.000760`；
- 最终 \(\rho\)：`0.199985`；
- 最终 \(\alpha/\rho\)：`0.00380`；
- \(\operatorname{corr}(\alpha, step) \approx 0.056\)。

结论：实际 target-direction response 仅达到最终请求值的约 `0.38%`，且没有稳定上升趋势。

### 2.2 位移主要是 off-direction

末尾典型值约为：

\[
\alpha \approx 0.00076,\qquad
\beta \approx 0.000484,\qquad
\omega \approx 0.000483.
\]

因为 \(\beta\) 几乎全部由 \(\omega\) 构成，所以 active-bypass prediction displacement 的绝大部分能量不在目标 \(B\rightarrow Y\) 方向上。

### 2.3 修正对 raw response loss 的解释

不能仅凭 raw response loss 随 \(\rho\) 上升，就断言训练发散。

当前 loss 为：

\[
L_{resp}=\lVert A-[B+\rho(Y-B)]\rVert^2.
\]

若 activator 完全不响应，即 \(A=B\)，则：

\[
L_{resp}=\rho^2\lVert Y-B\rVert^2.
\]

因此把 \(\rho\) 从 `0.05` 提高到 `0.20`，no-learning baseline 自然会放大：

\[
(0.20/0.05)^2=16.
\]

更准确的诊断量是 bypass-relative response ratio：

\[
R_{bypass}
=
\frac{\beta-2\rho\alpha+\rho^2}{\rho^2}.
\]

解释：

- \(R_{bypass}=1\)：与保持 bypass 一样差；
- \(R_{bypass}<1\)：正在接近目标 partial response；
- \(R_{bypass}=0\)：达到理想 target；
- \(R_{bypass}>1\)：active response 比完全不响应更差。

首轮最后 25 steps 平均约：

\[
R_{bypass}\approx1.008.
\]

所以最准确的结论是：

> activator 没有有效接近目标，但也没有明显爆炸；它基本收敛到了“近似 bypass”。

---

## 3. 首轮最可能被忽略的优化问题

原始 response loss：

\[
L_\rho=\lVert A-[B+\rho(Y-B)]\rVert^2.
\]

初始化附近 \(A\approx B\)，因此：

\[
\nabla_A L_\rho=2\rho(B-Y).
\]

普通 full-target reconstruction loss：

\[
L_{full}=\lVert A-Y\rVert^2
\]

的初始梯度是：

\[
\nabla_A L_{full}=2(B-Y).
\]

因此首轮 A1 的初始 target-directed gradient 约为普通 reconstruction gradient 的：

\[
\rho=5\%\text{ 至 }20\%.
\]

这意味着首轮实验把以下两个概念意外绑定在了一起：

1. 希望学习的小目标位移；
2. 用来找到该目标的优化推力。

它们应该被解耦。

---

## 4. 核心实现：保持 optimum，归一化优化强度

### 4.1 新的可配置 scaling

新增配置：

```yaml
conditional_response_v8:
  response_loss_scaling:
    mode: inverse_rho
    min_rho: 0.05
    max_multiplier: 20.0
```

第一版支持：

- `none`：保持现有行为，确保 v7/v8 旧实验可复现；
- `inverse_rho`：用 detached、clamped 的 \(1/\rho\) 缩放 response loss。

以后如有需要可添加：

- `inverse_sqrt_rho`：介于两者之间的优化强度。

但不属于本轮必须项。

### 4.2 数学定义

定义：

\[
m(\rho)=
\min\left(
\frac{1}{\max(\operatorname{detach}(\rho),\rho_{min})},
m_{max}
\right).
\]

新的 A1 response objective：

\[
L_{A1}=m(\rho)\cdot
\lVert A-[B+\rho(Y-B)]\rVert^2.
\]

当 `rho=0.05`、`min_rho=0.05`、`max_multiplier=20` 时：

\[
m(\rho)=20.
\]

初始化附近：

\[
\nabla_A L_{A1}
\approx
\frac{1}{\rho}\,2\rho(B-Y)
=
2(B-Y).
\]

因此：

- optimum 仍然是 \(A-B=\rho(Y-B)\)；
- 初始 target-directed gradient 不再随小 \(\rho\) 缩小；
- 不使用 \(1/\rho^2\)，避免小 \(\rho\) 下梯度按 \(1/\rho\) 爆增。

### 4.3 作用范围

本轮建议只在 **V8 A1 conditional-response objective** 启用 `inverse_rho`。

要求：

- `mode: none` 必须完全保持当前数值行为；
- Phase B 默认保持 `none`；
- A2 不因本轮改动而改变；
- 不修改 v7 loss 路径；
- scaling 只改变 response loss 的优化权重，不改变 diagnostics 中原始 \(\alpha,\beta,\omega\) 的定义。

### 4.4 多 category 的未来兼容

实现 helper 时应按 category rho 计算 multiplier，避免把 A1 特例硬编码进数学函数。

若未来 B 使用该 scaling，需要特别处理 `far.rho=0`。本轮默认 B 不启用；即使误启用，也必须通过 `min_rho` 和 `max_multiplier` 保证有限值。

---

## 5. 新增核心诊断指标

以下指标必须记录到 `v8_phase_a1_metrics.jsonl`，并保留原有 metrics。

### 5.1 Response efficiency

\[
\eta=\frac{\alpha}{\rho}.
\]

解释：

- \(\eta=0\)：没有学到请求位移；
- \(\eta=0.5\)：达到请求投影的一半；
- \(\eta=1\)：达到请求的 target-direction projection。

实现要求：

- 使用 safe rho denominator；
- 至少记录 local 和 shared 版本；
- metric 名称建议：
  - `.../response_efficiency_local`；
  - `.../response_efficiency_shared`。

### 5.2 Normalized off-direction contamination

\[
\chi=\frac{\omega}{\rho^2}.
\]

目标：

\[
\chi\rightarrow0.
\]

metric 名称建议：

- `.../normalized_off_direction`。

### 5.3 Bypass-relative response ratio

\[
R_{bypass}
=
\frac{\beta-2\rho\alpha+\rho^2}{\rho^2}.
\]

metric 名称建议：

- `.../bypass_relative_response_ratio`。

必须使用与相应 \(\alpha,\beta\) 相同的 decomposition 范围，不能混用 local alpha 与 shared beta。

### 5.4 Loss scaling diagnostics

记录：

- raw unscaled response loss；
- applied multiplier；
- scaled response objective；
- active scaling mode；
- effective clamped rho。

metric 名称建议：

- `.../response_loss_unscaled`；
- `.../response_loss_multiplier`；
- `.../response_loss_scaled`；
- `.../response_loss_effective_rho`。

保留现有 `response_loss` 时，必须明确它代表 raw 还是 scaled。建议保留其旧含义为 raw，避免破坏历史分析脚本。

### 5.5 Prediction-space norm diagnostics

每个 probe 至少记录：

\[
\lVert A-B\rVert,
\qquad
\lVert Y-B\rVert,
\qquad
\lVert A-T\rVert.
\]

建议同时记录归一化比值：

\[
\frac{\lVert A-B\rVert}{\rho\lVert Y-B\rVert+\epsilon}.
\]

metric 名称建议：

- `prediction_delta_norm`；
- `target_direction_norm`；
- `target_error_norm`；
- `prediction_delta_target_ratio`。

这些 norm 的 reduction、batch aggregation 和张量尺度必须在测试中固定并说明。

---

## 6. 修复 gradient diagnostics

### 6.1 现状

当前 `_write_trigger_binding_metrics()` 在 loss backward 之前写 metrics，因此 `parameter.grad` 仍为 `None`。现有 `gradient_norm: null` 并不能仅通过把 YAML 开关设为 `true` 解决。

现有 `trigger_selective_training.logging.debug_gradient_contributions` 逻辑属于旧 TST loss 分支，不能直接假定它已经覆盖 V8 conditional-response path。

### 6.2 本轮要求

为 V8 增加独立、清晰的 diagnostics 配置，或将现有 logging config 正式扩展到 V8。建议配置形态：

```yaml
conditional_response_v8:
  diagnostics:
    enabled: true
    gradient_steps: [1, 10, 25, 50, 75, 100]
    representative_te_layers: [0, 6, 12, 18, 24, 30, 35]
```

需要记录：

- embedding 总 gradient norm；
- module LoRA 全层 aggregated gradient norm；
- tap adapter aggregated gradient norm；
- representative module-LoRA layer gradient norms：0、6、12、18、24、30、35；
- 各组件 parameter norm；
- relative gradient norm：

\[
\frac{\lVert\nabla_\theta L\rVert}{\lVert\theta\rVert+\epsilon}.
\]

### 6.3 时序要求

不能继续依赖 pre-backward 的 `parameter.grad`。

可接受方案：

1. 在指定 diagnostic steps 使用 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` 对分组参数计算只读 gradient norms；或
2. 在正常 backward 之后、optimizer step/zero-grad 之前写入 diagnostics。

实现时优先选择对现有训练循环侵入最小、不会重复累积梯度且不会改变 optimizer 结果的方案。

### 6.4 Conditioning-path diagnostics

同时记录：

- trigger-span active conditioning delta norm；
- final Ideogram conditioning delta norm；
- prediction delta norm \(\lVert A-B\rVert\)。

目的是区分：

- activator 参数收到梯度，但 Qwen conditioning 几乎没变；
- Qwen conditioning 已变化，但 Ideogram final conditioning 被压弱；
- final conditioning 已变化，但 diffusion prediction 对其不敏感。

---

## 7. Held-out fixed probes 必须真正运行

### 7.1 现状

首轮 split manifest 正确包含：

- train：36 images；
- held-out：4 images；
- held-out IDs：12、18、19、6。

但 `validation.enabled=false`，所以没有产生 held-out response diagnostics。

另外，现有 `TriggerValidationConfig.every` 只表达周期，不足以精确表示本轮要求的离散 probe steps。

### 7.2 配置要求

扩展 validation config 支持：

```yaml
validation:
  enabled: true
  steps: [0, 10, 25, 50, 75, 100]
  seed: 42
  fixed_timesteps: [100, 500, 900]
  data_split_manifest: /data/train/models/ig4_TST_v8_a1_inverse_rho/data_split_manifest.json
  caption_sources: [structured]
```

要求：

- `steps` 优先于 `every`；
- 不配置 `steps` 时保持原有 `every` 行为；
- step 0 probe 在任何 optimizer update 前完成；
- 固定 image、caption、noise、timestep/sigma；
- train probe 与 held-out probe 分开输出；
- held-out items 永远不能进入训练 dataloader；
- probe 不得修改模型参数、optimizer state 或 RNG progression。

### 7.3 Probe 输出指标

train 与 held-out 都至少输出：

- rho；
- alpha local/shared；
- beta；
- omega；
- eta；
- chi；
- `R_bypass`；
- raw/scaled response loss；
- prediction delta norm；
- target direction norm；
- target error norm。

短 A1 run 的 gate 判断以 fixed probes 为主，普通随机训练 batch metrics 只作为辅助。

---

## 8. 下一轮实验配置

### 8.1 保持不变

继续使用完整 activator：

- 4 virtual tokens；
- semantic init：`illustration`；
- Qwen 每层 `Qwen3VLTextMLP.down_proj` rank-4 module LoRA；
- 13 个 rank-2 tap adapters；
- trigger-span token masking；
- JSON structured captions only；
- diffusion LoRA frozen；
- gamma 固定为 1.0；
- embedding / TE adapter / tap adapter 学习率先保持首轮值。

本轮不改变 adapter placement，不减少层数，不移除 taps。

### 8.2 修改项

建议创建新的诊断配置，不覆盖首轮配置：

```text
config/2026_08_16_ig4_r1X1dOn9mA2_v8_a1_inverse_rho.yaml
```

核心配置：

```yaml
phase_a1:
  steps: 100
  losses:
    conditional_response_v8:
      response_weight: 1.0
      hierarchy_weight: 0.0
      omega_weight: 0.0
      alpha_floor_weight: 0.0
      effect_consistency_weight: 0.0
      responses:
        trigger: {rho: 0.05}
      response_loss_scaling:
        mode: inverse_rho
        min_rho: 0.05
        max_multiplier: 20.0
      diagnostics:
        enabled: true
        gradient_steps: [1, 10, 25, 50, 75, 100]
        representative_te_layers: [0, 6, 12, 18, 24, 30, 35]
  save_steps: [10, 25, 50, 75, 100]
```

不再使用 ramping rho；固定：

\[
\rho=0.05.
\]

### 8.3 运行策略

- 第一观察点：step 10；
- 初始 gate：step 50；
- 只有 fixed probes 显示明确改善时才继续到 step 100；
- 若实现不支持自动 early stop，可保存 step 50 checkpoint 并人工决定是否继续；
- 本轮仍然 `stop_after_phase: a1`；
- 任何情况下都不自动进入 Phase B。

---

## 9. 实验验收门槛

### 9.1 实现正确性 gate

运行前必须通过：

- `mode:none` 与旧 response loss 数值一致；
- `inverse_rho` 不改变 target optimum；
- rho=0.05 时 multiplier=20；
- `min_rho` 与 `max_multiplier` clamp 正确；
- rho=0 不产生 NaN/Inf；
- raw 与 scaled loss 均被记录；
- eta、chi、`R_bypass` 数值与手算一致；
- gradient diagnostics 在指定 step 非 null；
- validation exact steps 正确触发；
- held-out 数据不进入训练。

### 9.2 Step 50 运行 gate

主要判断 fixed train/held-out probes，而不是单个随机 batch。

继续到 step 100 的最低信号：

- train probe \(\eta\) 有稳定正趋势；
- `R_bypass` 明显低于 1 并继续下降；
- \(\lVert A-B\rVert\) 有稳定增加；
- component gradient norms 非零且 finite；
- held-out probe 不表现为明显反向响应；
- loss、gradient 与 parameter norms 没有 NaN/Inf 或突增。

推荐的“有意义改善”参考线，不作为硬编码自动阈值：

- step 50 train-probe \(\eta\ge0.1\)；或
- step 50 train-probe \(R_{bypass}\le0.9\)；
- 并且多个固定 timestep 上方向一致。

### 9.3 成功判定

若到 step 100：

- \(\alpha\) 明显朝 `0.05` 接近；
- \(\eta\) 持续上升；
- `R_bypass` 持续下降；
- \(\chi\) 可控；
- held-out response 与 train response 同方向；

则说明完整 activator architecture 具备响应能力，首轮主要问题是 loss optimization strength。

### 9.4 失败判定

若 corrected scaling 下仍出现：

- \(\alpha\sim10^{-3}\)；
- \(\eta\) 长期接近 0；
- `R_bypass` 长期约等于 1；
- prediction delta 几乎不增长；

则下一步才进入 architecture/pathway 调查：

1. 检查 active/bypass gradient path；
2. 检查 trigger-span mask semantics；
3. 检查 embedding、module LoRA、tap residual 对 conditioning 的逐级影响；
4. 检查 Ideogram prediction 对 conditioning delta 的敏感度；
5. 再做 component ablations。

---

## 10. 本轮明确不做

为了保持实验可解释性，本轮暂不：

- 把首轮 A1 checkpoint 送入 Phase B；
- 修改 virtual token 数量；
- 修改 module LoRA parent/child placement；
- 移除或重排 tap adapters；
- 开始 embedding-only / TE-only / tap-only 消融；
- 启用 omega penalty、alpha floor、hierarchy loss 或其他辅助目标；
- 改变 diffusion LoRA；
- 改动 v7 objective；
- 使用 \(1/\rho^2\) scaling；
- 把 eta 解释为视觉风格百分比。

---

## 11. 实施 TODO

### P0：Loss scaling 与兼容性

- [ ] 在 V8 conditional-response 配置中解析 `response_loss_scaling`。
- [ ] 实现 `none` 与 `inverse_rho` 模式。
- [ ] 使用 detached rho、`min_rho` 与 `max_multiplier`。
- [ ] 保持 `mode:none` 与当前行为完全一致。
- [ ] 仅在新 A1 配置中启用 `inverse_rho`，B/A2 默认不变。
- [ ] 添加 scalar 与 batch/category rho 单元测试。

### P0：核心 metrics

- [ ] 记录 raw response loss 与 scaled response loss。
- [ ] 记录 applied multiplier 与 effective rho。
- [ ] 记录 \(\eta=\alpha/\rho\)。
- [ ] 记录 \(\chi=\omega/\rho^2\)。
- [ ] 记录 `R_bypass`。
- [ ] 记录 prediction、target-direction 与 target-error norms。
- [ ] 为所有比值增加 safe denominator 和 finite checks。
- [ ] 添加公式级数值测试。

### P0：真实 gradient diagnostics

- [ ] 确认并修复 metrics 写入时序，不能在 backward 前读取 `parameter.grad`。
- [ ] 增加 embedding aggregated grad norm。
- [ ] 增加 module-LoRA aggregated grad norm。
- [ ] 增加 tap-adapter aggregated grad norm。
- [ ] 增加代表层 0/6/12/18/24/30/35 的 grad norm。
- [ ] 增加各组件 relative gradient norm。
- [ ] 确保 diagnostics 不改变 optimizer gradients 或训练结果。
- [ ] 添加 non-null、finite、unused-parameter 测试。

### P0：Fixed train/held-out validation

- [ ] 给 `TriggerValidationConfig` 增加离散 `steps`。
- [ ] `steps` 优先于 `every`，并保持旧配置兼容。
- [ ] 接通 step 0/10/25/50/75/100 probe 调度。
- [ ] 固定 image、caption、noise、timestep/sigma 与 seed。
- [ ] 输出 train 与 held-out 独立 JSONL。
- [ ] 输出 eta、chi、`R_bypass` 与 prediction norms。
- [ ] 验证 held-out IDs 不进入训练 dataloader。
- [ ] 验证 probe 不改变训练 RNG、参数或 optimizer state。

### P1：Conditioning-path diagnostics

- [ ] 记录 trigger-span embedding delta norm。
- [ ] 记录 Qwen/tap active conditioning delta norm。
- [ ] 记录 final Ideogram conditioning delta norm。
- [ ] 将 conditioning delta 与 prediction delta 关联到相同 fixed probe。
- [ ] 明确张量维度、mask reduction 和 batch reduction。

### P1：新 A1-only 配置

- [ ] 新建 inverse-rho 诊断 YAML，不覆盖首轮 YAML。
- [ ] 固定 `rho=0.05`，移除 A1 rho ramp。
- [ ] 设置 100 steps 与 save steps `[10,25,50,75,100]`。
- [ ] 启用 validation 与 gradient diagnostics。
- [ ] 使用独立 output root，避免复用首轮 artifacts。
- [ ] 保持完整 activator、JSON-only、A1-only。

### P1：测试与运行 gate

- [ ] 运行新增 loss/metrics/config/validation 单元测试。
- [ ] 运行现有 V7/V8 regression tests。
- [ ] 做一个最小 synthetic backward test，确认三组件 gradient 非零。
- [ ] 启动新 A1 run，先检查 step 0/1/10 diagnostics。
- [ ] step 50 按 eta、`R_bypass`、chi、gradient 与 held-out probes 决定是否继续。
- [ ] 未通过 gate 时停止，不进入 B。

---

## 12. 实施顺序

严格按以下顺序执行：

1. Loss scaling helper 与配置解析；
2. eta、chi、`R_bypass` 和 raw/scaled loss metrics；
3. 真实 gradient diagnostics；
4. exact-step fixed validation；
5. conditioning-path diagnostics；
6. 新 A1-only inverse-rho YAML；
7. 单元测试与 regression tests；
8. 运行 50-step gate；
9. 仅在指标改善时继续到 100 steps；
10. 通过 A1 gate 后，再规划 Phase B。

---

## 13. 最终决策原则

本轮要回答的唯一首要问题是：

> 在保持小目标响应 \(\rho=0.05\) 的同时，给予近似 full-strength target-directed gradient 后，完整 V8 text activator 能否让 \(\alpha\) 明显接近 0.05？

如果能，优先保留当前架构并重新设计后续 rho curriculum。

如果不能，再怀疑 text-side pathway、module placement、mask semantics 或 Ideogram sensitivity，而不是提前根据首轮弱梯度实验否定整个 activator 架构。
