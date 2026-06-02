---
name: mimickit-g1-usd-material-fix
description: MimicKit g1.usd 全白渲染的真根因 + 修复 — MJCF per-geom rgba 在 USD 转换时被压成单一 DefaultMaterial，要从 g1.xml 反推 per-mesh 颜色重绑 36 个 mesh
metadata:
  type: feedback
---

MimicKit (xbpeng/MimicKit) 的 `dependencies/MimicKit/data/assets/g1/g1.usd` 在 Isaac Lab 里渲染会全白/泛白，看不到 teaser_g1.gif 里那种黑色 head/joints + 浅灰 panels 的真实 Unitree G1 外观。

**Why:** 不是 `visual_material=None` 剥掉了材质、也不是 PBR 调色不对、也不是 shader 输入名（OmniPBR vs UsdPreviewSurface）的问题。**真根因**：g1.usd 在从 MJCF 转换的过程中把每个 visual `<geom>` 的 `rgba` 属性压成了单一 `DefaultMaterial=(1,1,1)` 白；35 个 mesh 全部绑同一个 white material，原本 `(0.2 0.2 0.2)` 和 `(0.7 0.7 0.7)` 两色信息全部丢失。

**How to apply:**
- 看到 MimicKit G1 渲染异常单色，**优先检查 `data/assets/g1/g1.xml` 的 per-geom `rgba`** 是不是 USD 里没体现到 material binding 上
- 修法在 `scripts/g1_usd_recolor.py`（已落地）：parse g1.xml → 取 per-mesh rgba → 在 USD 里建 N 个 UsdPreviewSurface 材质 → 每个 mesh prim 用 `UsdShade.MaterialBindingAPI.Apply().Bind()` 重绑 → 导出 `g1_textured.usd`
- 切入方式：`export MIMICKIT_G1_USD=/abs/path/to/g1_textured.usd` — 配合 codex patch 写进 engine 的 `_parse_usd_path` env 桥（保留 `MIMICKIT_G1_USE_LOCAL_USD=1` 回退本地白材质，`MIMICKIT_G1_USD=...` 显式指定）
- 千万别直接给 `char_color = np.array([...])` 当万能药 — 它会覆盖 USD 自带 binding，永远拿不到 per-link 多色
- IsaacLab Nucleus 官方 `/Isaac/Robots/Unitree/G1/g1.usd` 虽然有真材质，但 body 顺序与 MimicKit 自带 g1.xml 不兼容（`body_common2sim.index(0)` 找不到），不能直接换

**⚠️ 引擎默认分支不能回落 Nucleus（2026-06-02 闪退根因）：**
- `_parse_usd_path` 三段优先级：① `MIMICKIT_G1_USD` env 显式覆盖 → ② `MIMICKIT_G1_USE_LOCAL_USD=1` 强制白 g1.usd → ③ **默认**
- codex 初版把默认设成 Nucleus G1 USD → 任何不设 env 的人启动 ~18s 在 `initialize_sim → _build_sensor_order_tensors` 崩 `ValueError: 0 is not in list`（body 顺序不兼容）
- 修复：默认改为「同目录有 `g1_textured.usd` 就用它（同拓扑 + 真材质），否则用本地白 `g1.usd`，**永不回落 Nucleus**」
- 这样 3 个入口（notebook 一行 / 本地训练 / HF 下载）都不必手动设 env，引擎自挑对的 USD
- HF 下载用户本地无 textured 文件 → `eval_chain.sh` 的 HF 分支设 `MIMICKIT_G1_USD` 指向 cache 里的 textured（走优先级①兜底）

**核心代码片段：**
```python
# parse_mjcf_colors() — 取 visual geom (contype=0 conaffinity=0 group=1) 的 rgba
# 然后 stage.TraverseAll() 找 Mesh prim, prim.GetName() 即 link name
# 用 UsdShade.MaterialBindingAPI.Apply(prim).Bind(material) 重绑
```

**关联记忆：**
- [[mimickit-lafan-fight-training-plan]] — 这个 fix 的用例，4 个 LAFAN motion eval 时用
- [[feedback-html-doc-rules]] — 详细写在 doc/mimickit_lafan_training.html
