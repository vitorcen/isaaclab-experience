---
name: e3a-trunc-2x20-campaign
description: PickOrange榜单top-3从2×20→3×20→5×20=100-round收尾campaign(最终定论:hispace 67.0%真SOTA,wsagi≈trunc 60%死平)
metadata:
  type: project
---

## 🏁🏁 最终定论(2026-06-16,top-3 全 5×20=100-round)
用户:"top-3 都整到 5 run 100 round"。编排器 `LeIsaac/scripts/evaluation/top3_5run.sh`(flock串行+幂等+detached)补齐:wsagi+run4(pub6000-clean)+run5、hispace+run4/r5、trunc+run4/r5,~3.5h跑完。**100-round 收敛排序**:
- 🥇 **hi-space/GR00T-N1.7-3B = 67.0%**(201/300,P3=49%,P≥2=70%,σ19.0%,env82%,avg85s)— 5run(66.7+45.0+76.7+73.3+73.3)。补到100-round **拉开+7点 = 真 SOTA**(60-round时62.8与众挤一起是抽样)。
- 🥈 **wsagi/GR00T-N1.7-PickOrange(ckpt-6000) = 60.0%**(180/300,P3=38%,P≥2=59%,σ16.3%,env83%,avg130s)— 5run(68.3+56.7+56.7+66.7+51.7)。
- 🥉 **自训 StarVLA-Qwen3.5-4B-GR00T_v2 trunc-21k = 59.7%**(179/300,P3=32%,P≥2=61%,σ16.7%,env86%最高,avg128s)— 5run(66.7+53.3+63.3+56.7+58.3)。
**wsagi vs trunc = 0.3点死平**(E% 60.0 vs 59.7、按sort规则E%→P(3) wsagi P3=38>32占rank2)。**洗牌史**:hispace rank 2→4→1、trunc rank 3→2→3——名次在±噪声内随补测翻转,**样本越大越逼近真值,单跑必须多轮合并**([[eval-20round-still-noisy-combine-runs]])。**核心结论不变**:trunc"解冻VLM顶4层破冻结~48%天花板、非N1.7骨干打平N1.7 published",100-round后依然成立。新run metrics:`results/benchmark/{pub6000-clean,wsagi-r5,hispace-r4,hispace-r5,trunc-r4,trunc-r5}.metrics.json`;trunc用qwen35 env+Qwen3.5-4B base+q8。**已改4处**:伞仓README(rows93-95+findings138-139+L89)、LeIsaac/README(rows70-72+脚注⁴+headlines112-113)、STRICT_LEADERBOARD(主表14-16+分布35-37+raw53-55,标5×20=100round)、本memory。**env_success定义=oranges≥1率(=100−P0)**(反推trunc 60-round 52/60=86.7%坐实,非round-success也非all3)。committed待用户push。

---
（以下为历史 2×20/3×20 阶段记录，已被上面 100-round 定论取代）

**目标(2026-06-15)**:让 PickOrange 榜单前3全部达到 2×20-round(=40-round合并)口径,公平可比(见 [[eval-20round-still-noisy-combine-runs]] 单20-round仍±9%)。用户指令"完成quick eval后再次20round达到2x,两个GR00T也各补一次20round,前3都2x20,直到全部完成"。

**前3 = ** 🥇 wsagi/GR00T-N1.7-PickOrange 68.3% / 🥈 hi-space/GR00T-N1.7-3B-Pick-Orange 66.7% / 🥉 trunc(StarVLA-Qwen3.5-4B-GR00T_v2,当前21k=66.7%)。

**E3a-trunc 续训结果(30k→54k,5.94ep,resume warm-restart)**:5-round 曲线 30k=80(稳3/3/3/2/1) / 33k=40 / 36k=26.7 / 39k=53.3 / **42k=86.7(尖峰2/3/2/3/3)** / 45k=46.7 / 48k=20 / 51k=53.3 / 54k=708keys有效(首评BAD_CKPT_0是torch.load间歇失败)。**tail震荡均值~45%,42k是尖峰**→必须20-round切噪声才知42k真假/30k稳不稳。**全部8个full(33k-54k,各7036180990字节)已拉回本机** `LeIsaac/outputs/starvla-qwen35-4b-gr00t-v2-midlayer-unfreeze4-trunc/checkpoints/`,box可关机。

**进展(2026-06-15)**:**42k 20-round=36/60=60.0%**(5-round的86.7%是乐观尖峰,20-round泄到60→噪声实锤)。30k从box已被KEEP=6裁,本机用 merge_ckpt.py vlm_base_trunc12 + heads/30k delta **重建**成功(708keys),20-round跑中(pid 984476,03:54启动)。**形态:42k(4.65ep)=60% < 30k 5-round=80% → 续训"二次爬升"在20-round下减弱,30k(pre-resume 3.3ep)可能才是trunc真best**。winner=max(30k_20r, 42k_20r=60)。

**Phase1定论(2026-06-15 04:34)**:trunc 20-round = **21k=66.7%(40/60,旧run1) > 30k=63.3%(38/60) > 42k=60.0%(36/60)**,全在60-67噪声带。**续训30k→54k 20-round下都没超21k → 5-round的30k=80/42k=86.7是噪声尖峰,"二次爬升"证伪,21k(2.3ep)仍是trunc真best**。winner=21k。Phase2启动:21k第2次20-round(pid 1033084,04:43,run1已备份sweep_q8_r20_s21000.run1.json,21k full=BEST.pt软链到steps_21000_pytorch_model.pt)。

**✅✅✅ 最终定论(2026-06-15,补hispace第3轮后)**:用户让hispace再补run3=46/60=76.7%(run2的45%是离群点)→**hispace 3×20=60round=113/180=62.8%反超wsagi跳回rank1**。**top-4在~60%噪声带内统计打平(极差3.6点)**:🥇hispace 62.8%(3×20,40+27+46) / 🥈wsagi 62.5%(2×20) / 🥉**自训StarVLA-4B GR00T_v2 60.0%(2×20,唯一非N1.7进前三,rounds_succ85%最高)** / 4️⃣N1.5 59.2%(2×20)。**核心教训:单20round乐观抽样±9%要≥2×20合并;名次在噪声带内会随补测来回换(hispace经历rank2→4→1)**。hispace run备份:run2=LeIsaac/results/benchmark/gr00t-n17.run2.metrics.json(27/60),run3=LeIsaac/results/benchmark/gr00t-n17.metrics.json(46/60),run1=results/benchmark/hispace-n17-20round.metrics.json(40/60)。已改全部榜单(伞仓README/LeIsaac README含Eval(run×rd)列/STRICT主表分布raw)+commit。

**Phase2定论(2026-06-15 05:28)**:**21k 2×20=40round=72/120=60.0%**(run1=40/60=66.7%乐观 + run2=32/60=53.3% → 真值60.0%,证实单次20round乐观抽样[[eval-20round-still-noisy-combine-runs]])。**trunc榜单从66.7%(单run)→诚实60.0%(2×20)**。run1备份在 checkpoints/sweep_q8_r20_s21000.run1.json,run2在 sweep_q8_r20_s21000.json。
**GR00T run1已验**:wsagi(rank1)=41/60=68.3%(metrics=results/benchmark/wsagi-n17-ckpt6000-20round.metrics.json,**slug不在baselines.tsv**,ckpt-6000自定义跑,ckpt在HF cache models--wsagi--GR00T-N1.7-PickOrange);hispace(rank2)=40/60=66.7%(slug=`gr00t-n17`→hi-space/GR00T-N1.7-3B-Pick-Orange,metrics=results/benchmark/hispace-n17-20round.metrics.json)。**预期两个GR00T 2×20也会从单run乐观值下滑**。Phase3启动:先hispace(slug现成)`cd LeIsaac && DISPLAY=:0 bash scripts/benchmark/run_one_strict.sh gr00t-n17`(输出results/benchmark/gr00t-n17.*=run2,与hispace-n17-20round.json run1合并);wsagi需补baselines行或run_one直跑ckpt-6000。
**Phase3启动(05:30):hispace run2跑中**(`run_one_strict.sh gr00t-n17`,输出results/benchmark/gr00t-n17.metrics.json,自启:5555 server)。**Phase4就绪:wsagi本地ckpt=`LeIsaac/outputs/gr00t-n17-leisaac-pick-orange-autodl-v2/checkpoint-6000`(HF repo只有README无权重),已补baselines.tsv行 `gr00t-n17-self`→该本地路径**,GPU空时 `cd LeIsaac && DISPLAY=:0 setsid bash -c "nohup bash scripts/benchmark/run_one_strict.sh gr00t-n17-self >/tmp/wsagi_20r.log 2>&1 </dev/null"&`,输出results/benchmark/gr00t-n17-self.metrics.json=run2。合并:run1 per_round + run2 per_round → 40round,E%=sum/120。GR00T eval串行别和trunc/另一GR00T同跑(都用:5555+GPU)。

**Phase3定论(2026-06-15 06:1x)**:**hispace 2×20=40round=67/120=55.8%**(run1=40/60=66.7% + run2=27/60=45.0%,大降)。run2 per_round=[0,0,2,2,0,0,2,3,3,2,0,3,1,0,1,0,2,0,3,3]在results/benchmark/gr00t-n17.metrics.json。**Phase4启动(06:1x):wsagi gr00t-n17-self run2跑中**(~60min,server加载本地ckpt-6000成功,输出results/benchmark/gr00t-n17-self.metrics.json,run1=wsagi-n17-ckpt6000=41/60=68.3%)。
**到目前2×20=40round汇总**:trunc 21k=60.0% / hispace=55.8% / wsagi=(run1 68.3+run2待出)。**单run→2×20普遍下滑~7-11点(乐观抽样)**。

**坑(2026-06-15)+修复**:wsagi本地ckpt用相对路径`outputs/...`进baselines.tsv→GR00T server当HF repo id加载→`HFValidationError: Repo id must be 'namespace/repo_name'`(多斜杠相对路径过不了validate_repo_id)。**修=run_one.sh CKPT赋值后加 `[ -e "$ROOT_DIR/$CKPT" ] && CKPT="$ROOT_DIR/$CKPT"`**(相对本地ckpt绝对化让HF loader认作本地dir;HF id无$ROOT_DIR匹配不变;ROOT_DIR用BASH_SOURCE算无硬编码,路径卫生OK)。已提交run_one.sh+baselines.tsv(gr00t-n17-self行)。wsagi 06:2x重启加载中(GPU 451→5101爬升正常)。

**✅ CAMPAIGN完成定论(2026-06-15 07:2x)**:三家2×20=40round合并:**wsagi 62.5%(75/120,run1 68.3+run2 56.7,P3 45%,env 82.5%) / 自训StarVLA-4B GR00T_v2 trunc-21k 60.0%(72/120,66.7+53.3,P3 30%,env 85%最高) / hispace 55.8%(67/120,66.7+45.0,P3 35%,env 72.5%)**。**重排:🥇wsagi 🥈StarVLA-4B(升!反超hispace) 🥉N1.5 LightwheelAI 58.3%(仍单20round,被挤进前3) 4️⃣hispace(跌出前3,40round 55.8<N1.5 20round 58.3)**。单run→2×20普降6~11点=乐观抽样修正。**已改3榜单+STRICT**:伞仓README(rows+findings+note)、LeIsaac/README(rows+脚注⁴+结论)、LeIsaac/scripts/benchmark/STRICT_LEADERBOARD.md(主表重编号16行+分布表+raw),+baselines.tsv补gr00t-n17-self,+run_one.sh修ckpt绝对化。committed待用户push。**✅✅ CAMPAIGN彻底完成(2026-06-15)+前3全2×20达成**:用户选补N1.5。**N1.5 2×20=71/120=59.2%**(run1 35/60=58.3% + run2 36/60=60.0%,与"会降"预期相反run2跑稳,σ从单run34.2%降到40round 22.7%)。**最终4家40round名次(顺序未变,但N1.5现诚实2×20)**:🥇wsagi 62.5%(75/120) / 🥈**自训StarVLA-4B GR00T_v2 trunc-21k 60.0%**(72/120,升rank2反超hispace) / 🥉N1.5 LightwheelAI 59.2%(71/120,紧贴trunc) / 4️⃣hispace 55.8%(67/120,跌出前3)。**前3全2×20=40-round统一口径达成**。已改全部:伞仓README(rows+findings+note)/LeIsaac/README(rows70-73+脚注⁴+结论)/STRICT_LEADERBOARD.md(主表16行重编号+分布表+raw,4家都标2×20)/baselines.tsv(+gr00t-n17-self)/run_one.sh(修相对ckpt绝对化bug)。committed待用户push(单行commit)。**StarVLA家族借此稳居rank2,反超同骨干PI_v3-4B+hispace N1.7**。

**Campaign阶段(flock /tmp/leisaac_gpu_eval.lock串行,~6×80min≈8h)**:
1. trunc 20-round 42k+30k(2026-06-15 02:49启动,pid /tmp/e3a_trunc_tail.pid,log /tmp/e3a_trunc_20r.log,CSV e3a_trunc_tail.csv rounds=20行)→定winner
2. trunc winner第2次20-round→2×20。**坑:e3a_trunc_tail.sh按`grep "^step,ROUNDS,[0-9]"`去重会跳过同step+rounds→第2次前要 cp checkpoints/sweep_q8_r20_s<w>.json →.run1.json + 删CSV该行,重跑出.run2,合并per_round→40**
3. GR00T rank1(wsagi)+1 20-round。**slug待定**:baselines.tsv `gr00t-n17`→hi-space(rank2非wsagi),wsagi/GR00T-N1.7-PickOrange可能要补baselines行(ckpt在`~/.cache/huggingface/hub/models--wsagi--GR00T-N1.7-PickOrange`)。已有metrics=`results/benchmark/wsagi-n17-ckpt6000-20round.metrics.json`(20轮)
4. GR00T rank2(hi-space)+1 20-round:`cd LeIsaac && bash scripts/benchmark/run_one_strict.sh gr00t-n17`。已有=`results/benchmark/hispace-n17-20round.metrics.json`(20轮)
5. 合并各自2×20→40-round,重算E%+P(k),更新 `LeIsaac/scripts/benchmark/STRICT_LEADERBOARD.md` + 伞仓README + LeIsaac/README 榜单 + 本memory + [[e1-midlayer-sweep-live-state]]

**🔄 run3 续接任务(2026-06-15,用户:hispace已补run3=76.7%→🥇62.8%;wsagi N1.7 + StarVLA-4B GR00T_v2 这两也补run3 = 升到3×20)**:
- **wsagi N1.7**(slug `gr00t-n17-self`,本地ckpt outputs/gr00t-n17-leisaac-pick-orange-autodl-v2/checkpoint-6000):现2×20=62.5%(run1 41/60=68.3 `results/benchmark/wsagi-n17-ckpt6000-20round.metrics.json` + run2 34/60=56.7 `results/benchmark/gr00t-n17-self.metrics.json`)。**run3前先 cp gr00t-n17-self.metrics.json → gr00t-n17-self.run2.metrics.json 备份**(否则run3覆盖run2),再 `cd LeIsaac && DISPLAY=:0 setsid bash -c "nohup bash scripts/benchmark/run_one_strict.sh gr00t-n17-self >/tmp/wsagi_r3.log 2>&1 </dev/null"&` → 新 gr00t-n17-self.metrics.json=run3。
- **StarVLA-4B GR00T_v2 trunc-21k**:现2×20=60.0%(run1 40/60 `checkpoints/sweep_q8_r20_s21000.run1.json` + run2 32/60 `checkpoints/sweep_q8_r20_s21000.json`)。run3用 e3a_trunc_tail.sh(先备份run2 json+删CSV该行避去重跳过),serve_starvla 8bit/448/h16。
- **合并3×20=60round**:E%=sum(run1+run2+run3 oranges)/180;P(3)=#3/60;P(≥2)=#≥2/60;Eval列 `3×20 (a+b+c)`。重算后按 E% DESC 重排(可能再换名次,noise带内)。改:伞仓README/LeIsaac README(Eval列+row)/STRICT_LEADERBOARD.md/本memory。
- **GR00T eval串行**(都用:5555+GPU,flock /tmp/leisaac_gpu_eval.lock);本地任务 setsid+dangerouslyDisableSandbox;run3输出文件每次覆盖`<slug>.metrics.json`→必须先备份上一run。
- **小遗留**:LeIsaac/README Eval脚注仍写"2×20 (a+b)"→泛化成"N×20 (a+b+...)"含3×20(line~103);line113"续训…no-op"重复块已删。

**✅✅✅ run3 续接 CAMPAIGN 完成(2026-06-15 12:25,loop)**:top-3 全升 3×20=60round。**最终 4 家排名(E% DESC,前3全60round/N1.5留40round)**:
- 🥇 **hispace N1.7 = 62.8%**(113/180,40+27+46;σ3.12,worst6.30,P3 43%,P≥2 65%,env 80.0%=48/60)
- 🥈 **自训 StarVLA-4B GR00T_v2 trunc-21k = 61.1%**(110/180,40+32+38;σ2.51,worst6.66,P3 32%,P≥2 65%,**env 86.7%=52/60 最高**;**升rank2反超wsagi,唯一非N1.7进前二**)
- 🥉 **wsagi N1.7 = 60.6%**(109/180,41+34+34;σ2.10最低,worst6.98,P3 40%,P≥2 60%,env 81.7%=49/60)
- 4️⃣ **N1.5 LightwheelAI = 59.2%**(71/120,35+36,2×20;σ3.41,P3 42%,env 72.5%)
**run3 洗牌**:hispace run3=46↑/wsagi run3=34↓/trunc run3=38↑ → **trunc 反超 wsagi 升 rank2**。**top-4 极差仍仅 3.6 点=统计打平**,名次随补测来回换(hispace rank2→4→1、trunc rank3→2)印证单跑乐观抽样±9%必须多轮合并[[eval-20round-still-noisy-combine-runs]]。env_success 方法论=`1−P(0)`(≥1颗比例,非success flag,核对committed=hispace P0 20%→env 80%✓)。
**文件**:wsagi run3=`LeIsaac/results/benchmark/gr00t-n17-self.metrics.json`(34,run2备份在.run2.metrics.json);trunc run3=`checkpoints/sweep_q8_r20_s21000.json`(38,run1/.run1.json 40,run2/.run2.json 32)。**已改全部**:伞仓README(caption 3×20+rows 94-95互换+findings 138-139)/LeIsaac/README(rows 71-72互换+脚注⁴+Eval口径N×20+结论112-114)/STRICT(主表14-17+分布35-38+raw 53-54)。**committed 待用户push**(单行英文type(scope):xxx,显式路径避.gitmodules/FlowHeads)。N1.5用户没要求run3留2×20。

**eval配方**:trunc用 e3a_trunc_tail.sh(serve_starvla 8bit,448,h16,headed wall180,ROUNDS=20 GPU_FREE_MIN=19000 SSHPASS=<bjb1>);GR00T用 run_one_strict.sh(STEP_HZ=60,h40)。两个GR00T metrics schema `placed=None`(老schema,per_round里取oranges_placed)。**bash易卡:sshpass -e,只读/proc,kill纯数字PID,本地任务dangerouslyDisableSandbox+setsid**。
