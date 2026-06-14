---
name: flowdp-e1-e2-live-state
description: Flow-matching baseline 落地态(2026-06-13);E1=DP高清(resize[480,640])在westc跑中(tmux e1hires,14000/1400,loss健康降,config已验clean);E2=LeIsaac/FlowHeads submodule已建(commit53c789b未push,FlowMatchingModel覆写DP的compute_loss+conditional_sample);**最终可用配方=克隆starvla env白嫖torch2.6+nvidia+lerobot0.4.4+transformers降4.49+本地rsync v3.0数据集+pyav后端+HF_HUB_DISABLE_XET**(破"盒子下>100MB必卡死");E2起训前FlowDP的import要从0.5.x适配到0.4.4
metadata:
  type: project
---

# Flow-Matching baseline 落地态（2026-06-13）

承接 [[flow-matching-policy-survey]] 的结论(自研最小 CFM head + 先 E1 DP 高清重测)。

## 🐛 eval "卡死"两个真根因(2026-06-13,已修)
新 FlowHeads 变体(flowdit/flowditx/dit)eval 反复"卡死",拆出**两个独立 bug**:
1. **打包 bug(主因)**:lerobot-v044 里 `flowheads` 是 editable 安装,生成的 `__editable__.flowheads` finder **MAPPING 冻结={flowdp,flowact}**(安装时新包还没建)→ eval server(不 cd 进 FlowHeads)报 `No module named 'flowdit'` → server 崩 → client `Failed to connect` → 假"卡死"。flowdp/flowact 能评因在 MAPPING 里;本地冒烟过因 `cd FlowHeads`(cwd 在 path)。**修=写 `$SP/flowheads_all.pth` 内容为 FlowHeads 根目录** → 5 包全可导入(免重装)。**新增任何 FlowHeads 子包必须确认它在 server env 可导入**(.pth 已覆盖整个 FlowHeads 根,新包自动生效)。
2. **Isaac teardown 死锁(放大器)**:client 抛异常(如连不上)→ 解释器关闭时 Isaac headless `SimulationApp` 卡 teardown → run_one 永不返回、日志冻结、GPU 钉住 ~12min(仅 720s 硬超时能解)→ dp_quick_eval 的快速重试失效。**修=`policy_inference.py` 的 `__main__` 用 `os._exit`**:main() 异常→print traceback+`os._exit(1)`(秒级死→快重试),正常返回→`os._exit(0)`(避免成功路径 teardown 卡)。metrics 在 eval 中增量写,os._exit 不丢。
**复发处置**:某 ckpt 长无 metrics+bench 日志冻结(末行=Replicator removal)→ 先看 `logs/lerobot_server.log` 末行的真错(import? OOM? 段错误?),不是只看 client 端"Failed to connect"。按 PID 杀整条卡链(留 watcher),watcher 自动重评。

## 🔑 box SSH 接入(westc:31709)
- 用户名是 **`root@`** — `ssh -p 31709 root@connect.westc.seetacloud.com`。漏掉 `root@` 会用本地用户名 `david` 登录 → `Permission denied`,易误判成"密码失效/box 重启换密码"。先确认用户名再怀疑密码。
- 密码:`pass autodl/westd`(注意 westc 主机对应的 pass 条目名是 `westd`;另有 `autodl/west-c` 是别的)。
- **已装本机公钥** `~/.ssh/id_rsa.pub` 到 box `authorized_keys` → 现在免密直连,巡检不必再走 sshpass。

## E2 = LeIsaac/FlowHeads submodule（2026-06-13 用户重命名 FlowDP→FlowHeads 伞包，含两变体）
**重命名**:原 `LeIsaac/FlowDP` → **`LeIsaac/FlowHeads`**(伞包);远端原 `git@github.com:vitorcen/FlowDP.git`(GitHub repo 可能也已改名 FlowHeads,push 前核对)。FlowHeads 现含**两个 flow-matching 变体**(各只换 head、backbone 逐字复用,赢负归因到目标不归因 backbone):
- **`flowdp/`**(我写的 E2)= **DP conv-UNet** backbone(ResNet18+SpatialSoftmax+FiLM 1D-UNet)+ flow head;锚定 **DP 8.3%**(E1 高清重测≈同档,峰 2/15)。`FlowMatchingModel(DiffusionModel)` 只覆写 `compute_loss`(DDPM ε→线性插值 `x_τ=(1-τ)x0+τx1`+速度场 `x1-x0`)+ `conditional_sample`(DDIM→Euler N 步);encoder/UNet/processor/normalization 全继承;**无 Transformer/DiT**(60-demo 小数据 conv-UNet 胜)。
- **`flowact/`**(用户新写)= **ACT transformer** enc-dec + ResNet backbone + flow head(CVAE L1+KL → flow velocity);锚定 **ACT 43.3%(最强 baseline)**。**ACT≫DP→flowact 比 flowdp 有前途得多**(强 backbone 上换 flow head)。文件 configuration/modeling/processor/train_flowact.py。
- 本地 commit `53c789b`(旧 FlowDP 内容);LeIsaac gitlink 待用户重新 commit/push;**push 顺序**:先 push FlowHeads(origin main)→ 再 commit/push LeIsaac gitlink。
- 启动:`FlowHeads/scripts/train_e2.sh`(或对应 flowact 脚本);native 480×640, AUTO_EVAL off。

## 🆕 DiT Policy 两变体已落地 + 三方评审(2026-06-13 ~19:40)
FlowHeads 现含 **4 个变体**(flowdp/flowact 已有,新增两个 DiT):
- **`flowdit/`** = flow head + **DiT denoiser(adaLN-Zero),全局 FiLM 条件**。继承 flowdp 的 `FlowMatchingModel`(flow loss + Euler 采样),**只换 denoiser conv-UNet→DiT**。单变量 A/B vs flowdp。5.04M 参数(DP 同档)。这是**干净对照**但天花板≈flowdp(8-13%)。
- **`flowditx/`** = DiT + **cross-attention 到空间 vision token**(tap DP 的 `rgb_encoder.backbone` 在 SpatialSoftmax pool 之前的 (B,C,H,W) 特征图→token,cross-attn)。保留全局 FiLM(条件是 flowdit 的超集)。7M denoiser/18M 含 ResNet。**非干净 A/B,是真正能打的变体**——橙子 10-40px 在特征图里存活但被 32-keypoint pool 压死,给 denoiser 直接空间访问是能抬 DP 系天花板的那一招。
- **三方评审一致(codex gpt-5.5 + mimo)**:正确性 ✓(global_cond_dim/adaLN-Zero/timestep 都对);FIX-FIRST 已修(horizon 断言、删 dead expand、inference_seed 确定性、action_clamp、脚本预算 100k→14000/1400 否则 44 epoch 纯过拟合);战略=全局-FiLM flowdit 只确认已知瓶颈,**真杠杆是 cross-attn(flowditx)**。用户决定:**对照都跑 + 建空间 DiT**(flowact 完→跑 flowdit 对照→flowditx 真竞争者)。
- **重要纠正评审前提**:DP 视觉塔是 **SpatialSoftmax(32 keypoints→64维)非 average pool**,保留部分空间(keypoint 位置)但只 32 个。
- 全部静态冒烟过:flowdit(determinism seed=0→同 chunk ✓)+ flowditx(backbone probe 512ch/vision_tokens reshape/compute_loss/generate_actions 端到端 ✓)。eval 注册已加(factory + SUPPORTED_POLICIES 含 flowdit/flowditx)。**完整训练链未在 box 跑过**。

## ⚠️ 翻案(2026-06-14 ~12:00):strict 20-round 推翻 quick 初判 → flowdp≈ACT,骨干才是杠杆
**三 best ckpt strict 20-round(EVAL_ROUNDS=20/EP_LEN=120/wall_cap=180/HORIZON=8,runner=`LeIsaac/scripts/evaluation/strict20_best.sh`):**
| 变体 | 骨干×目标 | strict 20r | 满轮/20 | quick峰 |
|---|---|---|---|---|
| **flowdp** | conv-UNet×flow | **45.0%**(27/60) | **3** | 46.7% |
| **ACT**(锚) | transf×CVAE+L1 | 43.3% | 有 | — |
| dit | DiT×DDPM | 15.0%(9/60) | 0 | 20% |
| flowditx | DiT×flow | 10.0%(6/60) | 0 | 26.7% |

- **核心翻案**:初版"生成头全失败,只ACT判别头能用"**错了**。根因=quick的`wall_cap=90s`系统性截断**慢但能干**的flowdp(成功episode要~136s,90s判死)。strict 180s下flowdp翻45%≈ACT43.3%。
- **新定论=骨干一阶,目标/头二阶**:能干骨干 conv-UNet(flowdp45%)与ACT-transf(43.3%)打平;DiT骨干无论DDPM(dit15%)还是flow(flowditx10%)都弱+0满轮。**分档线在骨干不在目标**。flow≈判别头(flowdp flow ≈ ACT判别)。
- **方向相反的偏差证confound真**:flowdp quick≈strict(本就强);dit/flowditx strict(15/10)<quick峰(20/26.7)=quick峰是乐观抽样噪声。→**quick峰值不可信,strict定论**。
- **满轮rounds_succ比橙子%更硬**:只flowdp(3)+ACT有满轮,DiT全0。
- **方法学教训**:quick-eval紧wall_cap误判慢策略→定论必须strict长wall_cap(≥180s),quick只配快筛"能不能动"。**推论:flowact(ACT×flow 13.3%)/DP-hires(13.3%)也只quick数,可能被低估,要strict复测才定档**。
- 文档已重写 `doc/flowheads_architecture_objective_matrix.html`(strict定论版)。strict指标 results/benchmark/{flowdp-9800,dit-14000,flowditx-11200}-s20.metrics.json。

## ✅ (上一版,已被上方strict翻案)2026-06-14 ~07:20:七路 quick-eval 全完 → doc/flowheads_architecture_objective_matrix.html
**全 20 个 sweep ckpt(dit 10 + flowditx 10)evaled,加既有 flowdp/flowact/flowdit/DP/ACT。** 总结 HTML(中文+SVG 矩阵)= `doc/flowheads_architecture_objective_matrix.html`。
**七路最终结果(HORIZON=8 quick-eval 5-round 峰值;ACT=strict 锚):**
| 变体 | 架构 × 目标 | 峰值 | 满轮 |
|---|---|---|---|
| **ACT**(基线) | Transformer × CVAE+L1 判别 | **43.3% strict** | 有 ✅ |
| flowdp | conv-UNet × flow | 7/15 (46.7%)@9800 | 0 |
| flowditx | DiT-crossattn × flow | 4/15 (26.7%)@11200 | 0 |
| dit | DiT-adaLN × DDPM | 3/15 (20.0%)@14000 | 0 |
| DP-hires | conv-UNet × DDPM | 2/15 (13.3%) | 0 |
| flowact | ACT × flow | 2/15 (13.3%) | 0 |
| flowdit | DiT-FiLM × flow | 0/15 (0%) | 0 |
| DP-lowres | conv-UNet × DDPM | 0/15(视觉对照) | 0 |

- **三条核心结论**:①**判别头是一阶因素**——只有 ACT 的 CVAE+L1 可靠完成整任务;同骨干换 flow(flowact)从 43.3%→13.3%。所有生成式头(flow/DDPM × convUNet/DiT)`rounds_succ` 全 0,只散落单颗。②**DiT 架构 60-demo 无增益**(0–26.7%,与 conv-UNet 同档;global-FiLM 还把视觉稀释到 0)。③**flow 不比 DDPM 差但救不回**——conv-UNet 上 flowdp(46.7%)反超 DP(13.3%),DiT/ACT 上 flow 无效;切目标是二阶。高清 DP 13.3% vs 低清 0% 也证伪"分辨率才是 DP 瓶颈"。
- **诚实声明**:FlowHeads/DP 是单 ckpt 乐观峰值(15ep 高方差,wall_cap 90s 宽松),ACT 43.3% 是 strict;绝对%不对等但"可靠完成 vs 零星散落"鸿沟稳健。flowdp 46.7% 是 9-ckpt 挑最优,余多在 0–2/15。
- **box 可关机**(训练全完早已拉回);本机 eval 队列已清空(全 20 完)。指标 `LeIsaac/results/benchmark/{dit,flowditx,flowdp,flowact,flowdit}-*.metrics.json`。
- **共享 GPU 纪律(本次新增)**:`flow_sweep_watcher.sh` 给整段 eval 包了 `flock /tmp/leisaac_gpu_eval.lock`,解决两 watcher(dit+flowditx)同时过 GPU 门双订 4090;**flock 子shell `( flock 200 )` 会让 pgrep 看到"成对"PID,那是机制不是重复 watcher——dedup 只数 PPID=systemd 的 top-level,绝不 kill 运行中 eval 子树**。详见 [[feedback-shared-gpu-eval-queue-orphan-discipline]]。

## ⭐ 最终可用配方(零大文件下载,破"盒子下>100MB必卡死")
此盒到 aliyun/pytorch-wheels 下**任何 >100MB 文件都卡死**(torch/nvidia 都栽,11 版试错全因此;HF 经 turbo 64MB/s 反而快)。破法:
1. **克隆 starvla env 白嫖 torch+nvidia**:`/root/miniconda3/bin/conda create -y --clone /root/autodl-tmp/envs/starvla -p /root/autodl-tmp/envs/lerobot`(py3.10,自带 torch2.6.0+cu124+全套 nvidia,可用)。**env 必须在 autodl-tmp(50G)不是 `/`(30G overlay,装依赖必 ENOSPC)**。
2. **lerobot 用 0.4.4**(非 0.5.x):0.5.x 需 torch≥2.7(下不动);0.4.4 只需 torch≥2.2,starvla 的 torch2.6 满足;且与 DP baseline 同框架科学正确(见 [[act-framework-drift-root-cause]])。`pip install -i aliyun "lerobot==0.4.4"`(小依赖,aliyun 快)。
3. **transformers 降 4.49**:克隆带来的 transformers 5.2.0 需 hf_hub≥1.3,但 lerobot 钉 hf_hub<0.36 → lerobot eager import groot→transformers→`is_offline_mode` 崩。`pip install "transformers==4.49.0"` 修(DP 不用 transformers,只要 import 过)。
4. **数据集要 v3.0**(不是 v2.1!**lerobot 0.4.4 已用 v3.0 格式**,HF 的 v2.1 会被拒"not backward compatible")→ **rsync 本地 v3.0**:`/home/david/work/isaaclab-experience/LeIsaac/datasets/raw/leisaac-pick-orange`(665M,v3.0)→ box `/root/autodl-tmp/datasets/leisaac-pick-orange`(本地→box rsync ~2MB/s 约 5min)。HF 直下会撞 **xet 401**(`HF_HUB_DISABLE_XET=1` 绕,但版本还是 v2.1 不能用)。
5. **视频后端必 pyav**:lerobot 0.4.4 默认 torchcodec 要系统 `libavdevice.so.58`(盒子没有)→ 崩;`--dataset.video_backend=pyav`(env 有 av 15.1.0,PickOrange 是 AV1,pyav 自带 dav1d 能解,见 [[starvla-av1-dav1d-thread-leak-enomem]])。
6. **训练前 export `HF_HUB_DISABLE_XET=1`**。
- 备用:盒子有 `/root/mihomo`(32M)+ 本地 hysteria 配置 `/home/david/hysteria-us.yaml`(Clash 格式,hysteria2 节点),要下大文件可在盒子起 mihomo 当快代理走 pypi.org;但当前配方零下载用不上。
- **pkill -f 自匹配坑**:SSH 命令行含 `bin/pip`/`pip install` 等字样时 `pkill -f`/`pgrep -f` 会匹配当前 shell 自己→静默杀会话(无输出)。只用 tmux kill-session 或精确数字 PID。

## ✅ 本机 DP eval env(lerobot-v044)+ quick-eval 链(2026-06-13 14:00)
**坑:0.4.0 装不了 0.4.4 训的 ckpt** —— 0.4.4 config 多了 `use_peft/resize_shape/crop_ratio/compile_model/compile_mode` 5 字段,draccus 严格解析 → `DecodingError`;且 **`resize_shape` 正是高清实验核心**,0.4.0 根本不支持 resize,强行删字段=原生分辨率跑、和训练不匹配 → **eval 必须用 0.4.4**。
- **建 env**:`conda create --clone lerobot-v040 -n lerobot-v044`(继承 torch2.7.1+cu126)→ `pip uninstall lerobot && pip install --no-deps lerobot==0.4.4`(--no-deps 不动 torch)→ 重打 **DP async patch**(`predict_action_chunk` 补 `populate_queues`+image stacking;box 的 0.4.4 与 0.4.0 此法逐字相同,同 diff 干净套用;见 [[lerobot-dp-async-server-bug]])。**不动 lerobot-v040(ACT baseline 锁 0.4.0,见 [[act-framework-drift-root-cause]])**。
- **eval 调用**:`cd LeIsaac; LEROBOT_PYTHON=.../lerobot-v044/bin/python EVAL_ROUNDS=5 EPISODE_LENGTH_S=60 MAX_ROUND_WALL_S=90 bash scripts/benchmark/run_one.sh dp-hires-<step> lerobot-diffusion 8 <ckpt-dir> lerobot "<label>"`。server(:8080,v044)持久复用,客户端每次 `--policy_checkpoint_path` 发路径→server 按路径重载,不服旧策略。sim 侧 conda env=isaaclab。STEP_HZ 默认 30(DP 非 GR00T 的 60)。**只拉 ckpt 的 `pretrained_model/`(~1.07G)不拉 optimizer(整 ckpt 3G)**。
- **eval 策略(user 定)**:训练中只 **5-round quick**(run_one.sh,**非** strict);从 **~1.5ep 起评**(002800=1.23ep→004200=1.85ep→...每个 ckpt);E1高清+E1低清+E2 全跑完看 quick 曲线选赢家,**只对赢家排一次 20-round strict**。abort=连续 3 ckpt 全 0-orange + 臂 range<0.1。
- **⚠️ retracted_middle 假阳性坑(早期 DP ckpt 必踩)**:harness `policy_inference.py` 的 `retracted_middle` 检测器=臂回 rest pose(shoulder_lift≈-100°/elbow_flex≈90°)+ 放置数 5s 不变 → 判"任务结束"提前砍轮。**早期 DP ckpt(动作幅度小,臂停 reset pose 附近)会一上来 ~15s 误触发**→ 每轮被砍、根本没尝试任务 → 假 0/15(GUI 一闪而过=整 eval 只 ~75s)。代码 L666-668 自标此假阳性 + 提供 `LEISAAC_DISABLE_RETRACT_DETECT=1` 关掉。**DP-from-scratch 全链 eval 必须带 `LEISAAC_DISABLE_RETRACT_DETECT=1`** 让每轮跑满。注:跑满 `wall_cap` 的 ckpt(2800/4200)没碰这检测器,无需重评。
- **自动清显存包装脚本**`LeIsaac/scripts/evaluation/dp_quick_eval.sh <slug> <ckpt-dir> [label]`:跑 run_one 后自动①按 PID 杀残留 policy_inference(pgrep 排除自身+wrapper argv 不含模式串=无自匹配)②`policy_server.sh stop lerobot` 释放常驻 :8080 server 的 ~2.7G → GPU 回 ~0(下个 eval run_one 自动重起 server)。后续 eval 都走这个 wrapper。
- **E1 高清完整 quick 曲线(5-round, retract OFF)**:2800=0,4200=**1**,5600=0,7000=0,8400=**1**,9800=0,11200=0,12600=**2**(峰,5.6ep),14000=1(6.2ep)。**峰值 2/15=13.3%,均值~5%**。vs 原低清 DP baseline 8.3% → **高清没显著救起 DP**(6ep 预算下,峰值勉强同档,在噪声内)。初步证伪"DP 8.3% 是视觉分辨率瓶颈"——更像小数据(60 demo)+ DP 从零 6ep 偏短的共同限制。待 E1 低清对照(同 6ep 直接对比)+ E2 FlowDP(同分辨率换 flow head)完整三方对比再定论。

## ⚠️ box 50G 盘 ENOSPC 纪律(低清训练已踩)
**DP/lerobot ckpt = pretrained_model(~1G model.safetensors)+ training_state(~2G optimizer)= 3G/个**。10 ckpt=30G,加 envs 15G+数据 1.4G → 50G 盘必爆。**低清训练 step1400 即 ENOSPC 崩**(伪装,实为盘满,见 [[feedback-cloud-env-reuse-disk-cleanup]])。
- **训练完成的 run:立即删全部 `checkpoints/*/training_state`**(optimizer 不再需要,省 2G/ckpt;eval 只用 pretrained_model)。`rm -rf .../checkpoints/*/training_state`。
- **eval 暂存策略**:每 ckpt 只拉 `pretrained_model/`(~1G)到本机 → box 上该 ckpt 可删。**hires eval 完/拉全后删 box 整个 hires checkpoints 腾 10G**。
- **进行中的 run**:patrol 每轮删 lowres `checkpoints/` 里**除最新外**的 training_state(留最新可 resume-on-crash);或训练完一把删。
- **崩后 resume 坑**:ENOSPC 崩时最后那个 ckpt 的 training_state 可能没存上(只有 pretrained_model)→ 无法 resume,只能重训(本案低清丢 1400 步重启)。

## 🔬 flowact 三方评审(codex 完整/mimo hung)+ 必修已应用(2026-06-13 17:00)
codex(gpt-5.5)审 flowact 出 10 条,**确认无 VAE/GT-action 泄漏**(use_vae=False→vae_encoder 不建,latent token 恒 0,GT action 只经 x1 构 flow target,推理无不可用通路)。mimo 卡 9 行 >20min hung 已杀。**已修(代码在 LeIsaac/FlowHeads/flowact/,验证 import+实例化过)**:
- **P0 闭环确定性**:`generate()` 原每次 fresh `torch.randn` 无 generator→同观测出不同动作=闭环抖动(ACT 原本近确定)。加 `config.inference_seed=0`+`generate(generator=)`,predict_action_chunk 用固定 seed generator(PickOrange 单峰→固定采样对);seed<0 重随机。
- **P1 masked loss**:原 `(mse*mask).mean()` 被 padding 稀释→改 `(per*mask).sum()/(mask.sum()*action_dim).clamp_min(1)` 除有效标量数。
- **P1 Euler clamp**:加 `config.action_clamp=3.0`,Euler 每步 clamp normalized x→防模型误差把 x 推出训练支撑=超关节范围饱和。
- **P2** `_batch_size_and_device` 补 OBS_STATE fallback(原只 image/env_state);`flow_source` 加 `__post_init__` 校验只允许 gaussian(visual 未实现);EMA/early-stop 注释名不副实→改实话(没 wire EMA)。
- **flowdp/flowact train.py 0.4.4 适配**:0.4.4 train 入口是 `lerobot.scripts.lerobot_train.main`(非 0.5.x 的 `lerobot.scripts.train`),flowdp 的 PreTrainedConfig 在 `lerobot.configs.policies`(非 `lerobot.configs`);都加 try/except(flowact 用户已写好,flowdp 我补)。
- **✅ P0 eval 注册已解决(flowdp/flowact 通用)**:async server 用 **client 发的 policy_type**(非 ckpt config.type)选 class:client `policy_inference.py` 把 `--policy_type=lerobot-flowdp` 拆成 `"flowdp"` 发给 server;server 先查 `SUPPORTED_POLICIES` 白名单(`lerobot/async_inference/constants.py`)→ 再 `get_policy_class(type)`(`lerobot/policies/factory.py`)→ `class.from_pretrained(ckpt)`。**修法=直接 patch lerobot-v044 site-packages 两处(和 DP async patch 同模式,不动 policy_server.sh):①`factory.py get_policy_class` 加 `flowdp→FlowDPPolicy`/`flowact→FlowACTPolicy` 分支(import 时连带注册 config subclass)②`constants.py SUPPORTED_POLICIES` 加 "flowdp","flowact"**。+ `pip install -e FlowHeads` 进 v044。验证 get_policy_class("flowdp")→FlowDPPolicy(DiffusionPolicy 子类,用 flow Euler 的 conditional_sample 非 DDPM)。**eval 调用**:`POLICY_TYPE=lerobot-flowdp bash scripts/evaluation/dp_quick_eval.sh flowdp-<step> <ckpt> "<label>"`(dp_quick_eval.sh 已加 POLICY_TYPE 参数+stuck off)。注:server 不传 config.type,纯靠 client policy_type 选 class→ckpt 用对的 lerobot-flowdp 即可。
- **narrative(非 bug,改 README)**:x_tau 进 decoder query 改了 ACT 数据流(不是纯"只换头");tau 均匀采样 low-NFE 早步方向(ablation)。

## E2 FlowDP 起训注意(0.4.4 适配)
FlowDP 代码按 **lerobot 0.5.x import 写**(`lerobot.utils.constants` 等),但 box 是 0.4.4 → 起 E2 前需对照 box 上 `/root/autodl-tmp/envs/lerobot/lib/python3.10/site-packages/lerobot/policies/diffusion/modeling_diffusion.py` 改 FlowDP 的 import(DiffusionModel/DiffusionPolicy/constants 位置;0.4.4 constants 可能在 `lerobot.constants`;compute_loss/conditional_sample 签名可能微差)。改完冒烟 3 步验,再 bash scripts/train_e2.sh(同 480×640+pyav,14000/1400)。

关联:[[flow-matching-policy-survey]]、[[feedback-cloud-env-reuse-disk-cleanup]]、[[cn-pypi-mirror-aliyun]]、[[autodl-uv-sync-cn-strategy]]、[[lerobot-dp-async-server-bug]]
