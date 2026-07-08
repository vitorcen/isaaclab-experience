---
name: so101-leader-follower-hardware-setup
description: 本机一对物理SO-101主从臂;主/从绑USB板序列号识别(主5B3D041438/从5B61036495,/dev/serial/by-id)别绑电压端口号;脚手架LeIsaac/scripts/so101(run.py+so101.sh)notebook极简无绝对路径;血泪:遥操startup冲极限拉垮供电(修max_relative_target)+主从对不齐是校准中位不一致非装歪+"主臂自己动"是端口重枚举缓存旧端口驱动错臂+kill-9打挂CH340只能SIGINT+首连no-status-packet要重试;权限udev根治;lerobot0.5.2 CLI(补feetech-servo-sdk)
metadata:
  type: project
---

# 🦾 本机 SO-101 主从臂上手(2026-07-08)

工作站(RTX 4090)接了一对物理 **SO-101 leader–follower** 臂,用于遥操录 LeRobot 数据集。

## 硬件映射(靠电压区分,别拔插)
两只臂同用 CH340 串口板(`1a86:55d3`)→ `/dev/ttyACM0`、`/dev/ttyACM1`,**VID:PID 相同无法靠 USB 描述符分**。判角色读 `Present_Voltage`:
- **`/dev/ttyACM0` ~5.2V = LEADER**(手动拖动、torque off)→ `--teleop.type=so101_leader --teleop.id=leader_arm`
- **`/dev/ttyACM1` ~12.2V = FOLLOWER**(带电持位)→ `--robot.type=so101_follower --robot.id=follower_arm`
- 各 6× Feetech STS3215,ID 已出厂配好 1..6(无需重设)。ACM 编号开机可能互换 → 认电压不认编号。

## 前置坑(已根治)
1. **串口权限反复丢**:`david` 不在 `dialout` 组,`/dev/ttyACM*` 属 `root:dialout`。**而且从臂堵转会 power-cycle → USB 重枚举 → 新设备节点权限重置(chmod 白做)**。根治=**udev 规则**(已装 `/etc/udev/rules.d/99-so101-serial.rules`:`SUBSYSTEM=="tty",ATTRS{idVendor}=="1a86",ATTRS{idProduct}=="55d3",MODE="0666"` + `udevadm control --reload && udevadm trigger`),插拔/掉线自动有权限。sudo 需密码=交互。
2. **lerobot env 缺电机 SDK**:`lerobot` env(`/home/david/miniconda3/envs/lerobot`,0.5.2 editable @ `dependencies/lerobot`)默认没 `feetech-servo-sdk`(=`scservo_sdk`)→ FeetechMotorsBus 报 `require_package`。已补装。

## 校准(源码实锤:同时校"位置零点"+"幅度量程")
`so_follower.calibrate()` 两步写 3 字段:①**"摆中位按回车"→`set_half_turn_homings()`=设 `homing_offset`(位置零点)**;②**"每关节推到两端"→`record_ranges_of_motion()`=设 `range_min/max`(幅度)**;`drive_mode` 这版硬编码 0(不校方向)。**wrist_roll 特例:被排除量程扫描、range 写死 0/4095,唯一校的是中位** → 它对不齐 100% 是中位问题。产物 `~/.cache/huggingface/lerobot/calibration/{robots/so_follower/follower_arm,teleoperators/so_leader/leader_arm}.json`(注意子目录是 `so_follower`/`so_leader` 非 `so101_*`)。CLI:`lerobot-calibrate`/`lerobot-teleoperate`/`lerobot-record`。

## 🔴 两个血泪根因(调了半天)
1. **遥操 startup 冲极限拉垮供电**:基础 `lerobot-teleoperate` 无软启动,第一帧就命令从臂到主臂当前姿全扭矩执行。开机时**从臂不在默认位、主从姿差大** → 从臂猛冲到关节极限**堵转过流 → 12V 拉垮 → CH340 复位 → USB 掉线**(报 `device disconnected`/`no status packet`,ttyACM 消失)。**解药=`--robot.max_relative_target=10`**(每步最多 10°,分步缓慢逼近不冲;日志 `clamped to be safe` 是正常工作非报错)+ 启动前对齐两臂姿态。
2. **主从对不齐 80°+ = 校准不一致(非舵机安装!)**:两臂独立校准但**中位摆在不同物理角**(wrist_roll 差 143°=1635tick)或**量程扫不全**(follower shoulder_lift 只录 1900tick vs leader 4095)→ 同一物理姿归一化出不同值。校准**就是**用来吸收每臂机械/盘齿差异的,只要中位一致+量程扫全就对齐,**别怀疑装歪**。修=重校两臂**用同一中位基准**(各关节居中的 L 形,别拿收起/贴极限当中位)+ 每关节推满两端(**盯 MIN/MAX 拉开再回车**,只按回车没推=`min==max` ValueError)。重校后 max|Δ| 83°→15°。启动前必读对齐(`get_observation` vs `get_action` 比关节角,<15° 才放行)。

## Notebook + 脚手架(2026-07-08 重构)
逻辑下沉 `LeIsaac/scripts/so101/`(`run.py` 全逻辑 + `so101.sh` 薄壳),**notebook 极简只 `!bash scripts/so101/so101.sh <cmd>`,不含任何 `/home/david` 绝对路径**(run.py 路径自解析:仓库根=`__file__` 上溯三层、env bin=`dirname(sys.executable)`;so101.sh 用 `$HOME`/conda 解析 lerobot python)。命令:detect/calibrate-follower/calibrate-leader/align/teleop-start/teleop-stop/act-start/act-stop。后台起=`subprocess.Popen`+`start_new_session`+pidfile+stdin 写 `\n\n\n` 自动应答校准提示;停=SIGINT 优雅关扭矩(**teleop 常驻别用 `!` 会卡死 cell**)。

## 🔴 血泪教训(这次踩的)
1. **主/从识别绑 USB 板序列号,别绑电压/端口号**:两只臂 CH340 各有唯一序列号(`/dev/serial/by-id/usb-1a86_USB_Single_Serial_<SN>`:主 `5B3D041438`/从 `5B61036495`),跟着控制板走。**"启动遥操作主臂自己动了"根因=端口重枚举(拔插后 ACM0/1 对调)+ cell 缓存了旧端口→驱动错臂**(不是电源接错;电压=接了哪个电源,不是臂身份)。每次启动都 `resolve_ports()` 按序列号重解析,别缓存。换臂只改 run.py 顶部两常量。
2. **停机械臂进程只能 SIGINT,绝不 `kill -9`**:`kill -9` 打断串口传输会把 **CH340 打挂、整个掉出 USB**(lsusb 里消失),要拔插重枚举。停止格用 `os.killpg(pgid, SIGINT)`。
3. **首连接偶发 `no status packet`**:`connect()`/首个 `read` 偶发抖动(尤其两个 kernel 抢串口)→ 包一层重试(`_connect_retry` 4×);端口探测 `subprocess.run` 必带 `timeout`(否则串口卡→cell 永久挂死)。**只开一个 kernel**。
4. **Jupyter 改磁盘不自动重载**:我在磁盘改了 ipynb,用户界面跑的还是旧内存副本→假故障(旧错误重现)。改完必须 **File→Reload Notebook from Disk**。

## GR00T N1.7 真机自主(路 A:lerobot 原生,已打通)
真机跑 SOTA GR00T = `so101.sh gr00t-start/gr00t-stop`(run.py `cmd_gr00t_start`),`lerobot-rollout --policy.path=<转换后ckpt> --strategy.type=base`,ckpt=`outputs/gr00t-n17-v10-curve/checkpoint-4500`(V2 SOTA sim 81%)。相机 key **必须 `front`/`wrist`**(=训练的 video.front/wrist),state 自动切 single_arm(5)+gripper(1),task 字符串同 sim。`max_relative_target=15`、fps=30、base strategy 不录数据。
- **🔴 原生 GR00T ckpt 不能直接喂 lerobot-rollout,必须先转 lerobot 格式**(脚手架 `scripts/so101/convert_groot_ckpt.py`,run.py 首次自动转+缓存 `checkpoint-4500-lerobot/`)。根因:`--policy.path` 走 draccus 解析要 config.json 有 `type` 字段,原生 ckpt 是 `architectures:[Gr00tN1d7]` 无 type→解析崩;`--policy.path` 与 `--policy.type` **互斥**(`filter_path_args` 报 `Cannot specify both`),`--policy.type=groot` 单给又是 fresh config 缺 input_features→`validate_features` 崩;`is_raw_groot_n1_7_checkpoint` 又硬要 `"type" not in config`→加 type 会让 processor 走序列化分支要 `policy_preprocessor.json`。**唯一解=完整转换**:`GrootPolicy.from_pretrained(原生)`(config 带上 features)+`make_pre_post_processors(pretrained_path=原生)`(is_raw 分支从 statistics.json 现建)→ `policy.save_pretrained`+`pre/post.save_pretrained` 落成完整 lerobot ckpt(config 带 type + model.safetensors 12.5G + policy_{pre,post}processor.json),之后 `--policy.path` 走序列化分支正常加载。转换/加载全链已 python 级实测通过,真机 rollout 待连臂测。
- **env 分离**:GR00T 跑在独立 env **`lerobot-060`**(lerobot 0.6.0),校准/遥操/ACT 仍用 `lerobot`(0.5.2)。`so101.sh` 按命令前缀 `gr00t-*` 自动切 env(选 `$HOME/miniconda3/envs/<env>/bin/python`),run.py 的 `BIN=dirname(sys.executable)` 随之指向对应 `lerobot-rollout`。**lerobot-060 也要 `pip install feetech-servo-sdk`**(=scservo_sdk,真机驱动电机必需,同 0.5.2 env 那个坑)。
- **🔴 lerobot 0.6.0 装法坑**:①**PyPI(官方+镜像都)最高只到 0.4.4**,0.5/0.6 只在 GitHub 打 tag → 换 pip 源没用,只能 `pip install "lerobot[groot] @ git+https://github.com/huggingface/lerobot@v0.6.0"`;②**硬要 python≥3.12**(3.11 直接被 pip 拒);③它自己拉 torch 2.11.0+cu130(nvidia cu13 wheels,别预装 aliyun cu128 torch,aliyin 源没 py312 的 torch)。装完实测:`GrootPolicy.from_pretrained(ckpt)` 16s load 成功 3.14B,cuda OK;唯一 MISSING `backbone.model.lm_head.weight`=Qwen 未用 LM head(`_tie_unused_qwen_lm_head`),无害。N1.5 走 `lerobot==0.5.1`(见 [[gr00t-multi-release-env-split]])。
- **🔴 CLI 逐坑清单(真机首跑一路踩过,全已修)**:①`feetech-servo-sdk`+`deepdiff`+`pyserial` lerobot-060 也要装(git 装无硬件 extras;SO101 只用这仨,dynamixel-sdk/python-can 用不上);②**视觉特征 mismatch**:GrootPolicy 对空 input_features 兜底成单个 `observation.images.camera`(modeling_groot.py:252),但真实是 embodiment 的 video keys(`new_embodiment`=front/wrist)→ 转换脚本 `_fix_camera_input_features` 从 processor_config 读真实 keys 覆盖(检查只比 key 集合非 shape);③电压预探测的 connect/disconnect 紧接 rollout 重连撞 CH340 复位窗口→`Incorrect status packet`→gr00t-start 用 `check_volt=False`;④**`--inference.type=rtc` 必需**:GR00T 输出 chunked 相对动作,默认 sync 逐 tick 解会被 `GrootN17ActionDecodeStep` 拒(要整块 predict→decode→queue);GR00T 原生支持 RTC(`_prepare_n1_7_rtc_inputs`/`rtc_ramp_rate`),默认 execution_horizon=10 够用。
- **✅ 真机跑通(2026-07-08)**:全链打通,GR00T N1.7 经 RTC **连续自主驱动从臂**(policy loaded cuda→robot connected→RTC 引擎→控制环产 goal+`max_relative_target=15` 钳位),稳定跑到用户主动急停。**推理速度 ~12Hz<30fps target(3B VLA 在 4090 正常,控制略顿挫,可 --fps 15 贴近)**。⚠️**别把急停误判成掉电**:按急停切从臂电机供电→`sync read Present_Position ids=[1..6] no status packet`=6 电机同时失联,长得和 12V 掉电/CH340 复位一模一样,但 CH340 适配器 USB 单独供电不掉出→`/dev/serial/by-id` 还在=急停非真掉电(真掉电才需拔插)。急停后 teardown 在断电总线上 disable_torque 失败=cosmetic 级联报错。区分:真掉电=CH340 掉出 USB(by-id 空)+ 常伴堵转;急停=by-id 还在。

## 文档 + 商家资料
项目文档 `LeIsaac/docs/training/so101_real_arm_bringup.html`(权威 LeRobot 流程 + 本机实测)。商家飞书文档(`tcnppips4y7o.feishu.cn/wiki/...`)是**登录墙 WebFetch 抓不到**,商家特有接线/上电/限位待用户贴来补进 doc §7。

关联 [[vla-pickorange-vision-resolution-selection]]、[[umbrella-leisaac-repo-boundary]]。
