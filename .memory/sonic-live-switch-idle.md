---
name: sonic-live-switch-idle
description: SONIC live demo运行时切prompt不重启+idle站立机制(VLA_PROMPT_FILE控制文件+sonic_say.sh+idle哨兵绕过模型);BonesSeed 5.2a
metadata:
  type: project
---

# 🎮 SONIC live demo 运行时切动作 + idle站立(BonesSeed 5.2a,2026-06-24)

**机制**:`vla_live_injector.py`(gear-sonic **patch 0004**)`eval_step` 每5步轮询 `VLA_PROMPT_FILE`,内容变了即切 `self.prompt`+强制重查询 → **无需重启 viewer/server**。控制模式(`pf` set)下 timeline 的 segment-entry 不覆盖 prompt(否则单段循环每 ~200 步重入 segment0 → flip-flop 回启动 prompt)。

- **切换**:`scripts/sonic_say.sh <motion>` 写 prompt 串到控制文件(motion→PROMPT 映射复用 `gr00t_build_sequence.PROMPT`,未知键当 raw prompt);`sonic_say.sh stand` 回 idle。
- **idle 站立**:prompt ∈ {stand,idle,none,空,站着} → `_is_idle()` → **跳过 query + 注入"捕获的复位站立 token"**(首次 decode 抓 `token_flattened[0].detach().cpu().numpy()`;启动用 squat clip → 帧0=直立站姿)。**"stand" 从不送进模型**(模型没训过 stand)——纯 injector 哨兵绕过推理,这是它能 work 的原因(用户揪过这点)。
- **BonesSeed 5.2a 单元**:▶️启动(`subprocess.Popen(start_new_session=True)` 后台非阻塞——**jupyter 不支持 `!...&`**=OSError;种控制文件=`stand`→启动只站立)→ 7动作切换(`sonic_say.sh <m>`)→ 🧍站立/停下 → ⏹️关闭(`gear_sonic_stop.sh` 杀 viewer+server)。
- **实测**:idle 站立 z0.79/query0 → 切 squat dpose3.9/z→0.55 → 切 stand query 停、z 回0.79。**kick/jump/walk 切过去仍冻**(=[[sonic-closeloop-freeze-rootcause]] 的根因限制,与切换机制无关)。
- **坑**:`_stand_token` 须存 cpu numpy(cuda tensor 进 `np.asarray` 崩 `can't convert cuda tensor to numpy`)。injector 文件 **gitignored**,真源 = patch 0004 经 `apply_gear_sonic_patches.sh` 还原(改了要 `git add -fN` 重生成 patch 才持久,见 [[git-submodule-gitlink-gotcha]])。
- 另:flow2/flow3 已全段 `bootstrap_steps:0`(零回放,`gr00t_build_sequence` 加 per-segment 覆盖,bs=0 时连 dump npz 都不需要)。

关联 [[sonic-closeloop-freeze-rootcause]]、[[umbrella-leisaac-repo-boundary]]、[[gear-sonic-preview-setup]]。
