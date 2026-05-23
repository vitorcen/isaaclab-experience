| Policy | Rounds ✅ (strict) | Rounds (env) | 🍊 (n/9) | Pick rate | Avg round time | Peak VRAM | Mean GPU util | Per-round detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT (self) — wsagi/ACT-PickOrange | 0/3 | 0/3 | 2/9 | 22.2% | 129.58s | 10.4 GB | 24.7% | 0🍊@106s / 0🍊@180s / 2🍊@103s |
| ACT (other) — shadowHokage/act_policy | 0/3 | 0/3 | 1/9 | 11.1% | 126.6s | 8.6 GB | 24.6% | 0🍊@157s / 1🍊@77s / 0🍊@146s |
| DP (self) — wsagi/DiffusionPolicy-PickOrange | 0/3 | 0/3 | 2/9 | 22.2% | 107.72s | 10.6 GB | 22.3% | 0🍊@159s / 2🍊@105s / 0🍊@60s |
| SmolVLA (self) — wsagi/SmolVLA-PickOrange | 1/3 | 1/3 | 5/9 | 55.6% | 355.0s | 10.0 GB | 23.0% | 3🍊@158s✅ / 0🍊@552s / 2🍊@355s |
| SmolVLA (other) — edge-inference | 0/3 | 0/3 | 0/9 | 0.0% | 167.99s | 10.2 GB | 23.4% | 0🍊@180s / 0🍊@167s / 0🍊@157s |
| GR00T N1.5 — LightwheelAI/leisaac-pick-orange-v0 (step_hz=60) | 0/3 | 0/3 | 4/9 | 44.4% | 104.52s | 16.2 GB | 36.1% | 1🍊@180s / 1🍊@55s / 2🍊@79s |
| GR00T N1.6 — hi-space/GR00T-N1.6-3B-Pick-Orange (step_hz=60) | 2/3 | 2/3 | 6/9 | 66.7% | 95.66s | 17.3 GB | 31.1% | 3🍊@39s✅ / 0🍊@180s / 3🍊@68s✅ |
| π0.5 (self) pt-v3 final_lora | 0/3 | 0/3 | 0/9 | 0.0% | 180.07s | 18.7 GB | 25.2% | 0🍊@180s / 0🍊@180s / 0🍊@180s |

**Success criteria**:
- **Rounds ✅ (strict)** = all 3 sticky `put_orange_to_plate` fired during round (EE-near + gripper-open + xy-in-plate at any point). Lower bound — misses fast transients (<33ms).
- **Rounds (env)** = `task_done` (orange xyz in plate box + arm rest). Can false-positive when an orange is knocked off plate edge to nearby table at similar height.
- **🍊 (n/9)** = sticky-counted oranges across all 3 rounds.
- ⚠️ `env-only` tag = env said success but sticky undercounted (mismatch case).
