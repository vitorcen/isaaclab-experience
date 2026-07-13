# lerobot 本地补丁(上游 huggingface/lerobot 无推送权限,fresh clone 后按序 git am)

- `0001-fix-diffusion-...patch` — `predict_action_chunk` 队列预热后按 **queues**(非原始 batch)stack:
  select_action 预热注入的合成键 `observation.images` 不在原始 batch 里,漏掉则 flowdp/DP
  异步 serving 与离线 eval 报 KeyError。应用:`cd dependencies/lerobot && git am ../../patches/lerobot/0001-*.patch`

另有两处**未提交**的临时实验改动(factory.py return_uint8 / modeling_act.py L1 mean,标注 REVERT AFTER EXPERIMENT,
2026-05-22 ACT 漂移验证用)——刻意不入 patch,过期后应 revert。
