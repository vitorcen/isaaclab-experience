---
name: gr00t-multi-release-env-split
description: GR00T N1.5/N1.6/N1.7 三个 release 拆 3 submodule + 3 venv，外部 clone 进垃圾桶，policy_type 统一 gr00t；N1.7 1-round 3/3 验证
metadata: 
  node_type: memory
  type: project
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# GR00T 多 release 环境分离 (2026-05-24)

## 触发
原 setup：N1.5 用 `~/work/Isaac-GR00T-N1.5/`（conda gr00t-n15 editable）；N1.6+N1.7 共用 `~/work/Isaac-GR00T/.venv`。后者 transformers 单方面 pip install 飘到 4.57.3 → N1.6 cached HF code 4 个 ImportError 连击。

## 解法
**1 release 1 submodule + 1 venv**：
- `dependencies/Isaac-GR00T-N1.5/` @ n1.5-release (4af2b62) — conda gr00t-n15 editable 重指向
- `dependencies/Isaac-GR00T-N1.6/` @ n1.6.1-release (5dc80c4) — uv venv，transformers **4.51.3**
- `dependencies/Isaac-GR00T/` @ n1.7-release-2 (3df8b38) — uv venv，transformers **4.57.3**

server_kind=gr00t-n{15,16,17} 路由对应 GR00T_DIR（policy_server.sh `start_gr00t_n1{5,6,7}` 各 export）。

## Wire bug 副作用
n1.7-release-2 `Gr00tSimPolicyWrapper.check_observation` 严格 validate flat obs key（`video.front` not nested）。LeIsaac `Gr00t16ServicePolicyClient` 发 nested → KeyError。

Fix: baselines.tsv 4 个 GR00T row 的 policy_type 从 `gr00tn1.6` 改为 **`gr00t`** → dispatch 到 `Gr00tServicePolicyClient`（flat-wire + `GR00T_WRAP_OBSERVATION=1` envelope 切换，已有 4 个 wire bug fix）。

## 验证
N1.7 1-round: hi-space ckpt → **3/3 oranges, 51.9s, 100% success** ✅。N1.6 venv ready（pyproject 4.51.3，cached code 对齐）。

## 5 个改动
| # | 文件 | 层 |
|---|---|---|
| 1 | `.gitmodules` + `dependencies/Isaac-GR00T-N1.5/` | root |
| 2 | `.gitmodules` + `dependencies/Isaac-GR00T-N1.6/` | root |
| 3 | `server/start_server.sh` + `init_server.sh` + `scripts/{policy_server,run_one,realtime_gr00t_viewer,check_start_gr00t,preview_gr00t_inference}.sh` — GR00T_DIR default 改 submodule + 加 gr00t-n17 case | root |
| 4 | `scripts/benchmark/baselines.tsv` — policy_type=`gr00t`，gr00t-n17 row server_kind=`gr00t-n17` | root |
| 5 | `LeIsaac/scripts/evaluation/policy_inference.py` — 加 `gr00t` 别名（同时增了 `DreamZeroServicePolicyClient` 79 行） | LeIsaac submodule，**直接 commit** `cf838c6 fix(gr00t): Use separate envs for GR00T N1.5, N1.6 and N1.7`（push 到 vitorcen/LeIsaac-Training 自己 fork） |

## 关键 gotcha
- `git submodule add --reference <ext_clone>` + 然后 trash ext_clone → submodule .git/objects 指向死路径，submodule 操作全 fail。修法：`git fetch origin --tags --force` 强制 re-download objects + 再 checkout tag 即可（**勿删 alternates 然后 repack**：repack 报 "bad object refs/heads/main"，因为 objects 真的没了）。
- N1.5 conda env 是 PEP 660 editable install（`__editable__.gr00t-1.1.0.pth`），direct_url.json url 字段指向外部 clone → trash 前必须 `pip install -e dependencies/Isaac-GR00T-N1.5 --no-deps` re-anchor。
- 真正运行时依赖外部 clone 的只有 6 处（`grep WORK_DIR/Isaac-GR00T server/ scripts/`）：start_server.sh / init_server.sh / policy_server.sh / realtime_gr00t_viewer.sh / check_start_gr00t.sh / preview_gr00t_inference.sh。其他都是 docs/URL ref。

## Submodule 维护方式定式
| Submodule | upstream | 方式 |
|---|---|---|
| LeIsaac | `vitorcen/LeIsaac-Training` (own fork) | **直接 commit + push** |
| lerobot | `huggingface/lerobot` | detached HEAD + local commit (93 个) |
| dependencies/Isaac-GR00T (N1.7) | `NVIDIA/Isaac-GR00T` | detached HEAD + 4 maintained local commit |
| dependencies/Isaac-GR00T-N1.5 / N1.6 | 同上 | 静态绑 `n1.5-release` / `n1.6.1-release` tag，无 local commit |

`patches/leisaac/` archive 是 2026-05-14 历史遗留（fork 未建前的归档方式），现已切到直接 commit。

## 关联
- HTML 全文档: `doc/gr00t_multi_release_env_split.html` (含 SVG 路由图)
- LeIsaac commit: `cf838c6` (vitorcen/LeIsaac-Training)
- [[gr00t-n17-leisaac-wire-debug]] — 之前 4 个 wire bug 已修在 `Gr00tServicePolicyClient`，本次仅 dispatch 路由改动
- [[gr00t-n17-hi-space]] — hi-space N1.7 验证（当时用外部 clone）
- [[per-model-action-horizon]] — N1.6=40, N1.7=40
