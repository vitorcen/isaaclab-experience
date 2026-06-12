---
name: feedback-commit-message-oneline
description: commit 注释只写简短一行 `type: xxx`，绝不换行/带作者后缀；且只有用户能 push，我从不 push
metadata:
  type: feedback
---

# commit 注释 = 单行 `type: xxx`，无任何后缀；push 只能用户做

用户要求：写 commit message **只能是简短一行**，形如 `feat: xxx` / `fix: xxx` /
`docs: xxx`（Conventional Commits 前缀）。**不可以换行、不可以带 body、不可以带
`Co-Authored-By` 或任何作者/工具署名后缀**。

**push 铁律：只有用户能 push，我从不 push**（任何 repo、任何 remote、fork 也不行）。
我最多把工作区/本地 commit 准备好，push 一律交给用户。

**Why**：用户自己负责所有 git 提交与署名，多余的后缀（尤其 `Co-Authored-By: Claude…`）
是噪音、污染开源仓库的提交历史。这条**覆盖**默认的「commit 末尾加 Co-Authored-By」习惯。

**How to apply**：
- 我代为 `git commit` 时（如给 fork 批量打 patch commit），`-m "feat: 简短描述"` 一条到底，
  英文、祈使句、聚焦单一改动；**不加 `-m` 第二段、不加 trailer**。
- 一个改动 = 一个聚焦 commit（见 [[feedback-refactor-smoke-pathhygiene]] 的「逐 patch」纪律）。
- 同样适用于 PR 标题；PR body 才可展开。
- **从不执行 `git push`**：连「push 一个 fork 分支」这种也只给用户命令、由用户跑。

关联：[[feedback-style]]（协作风格）、[[umbrella-leisaac-repo-boundary]]（fork 提交场景）。
