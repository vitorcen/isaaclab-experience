---
name: git-submodule-gitlink-gotcha
description: 加 submodule 时只提交了 .gitmodules、漏了 tree 里的 gitlink → GitHub 不显示 submodule、本地路径变未跟踪；验证/修复法
metadata:
  type: feedback
---

# submodule 要显示在 GitHub，`.gitmodules` 条目 + tree 里的 gitlink **两样都要**

2026-06-14 实测:LeIsaac-Training 上看不到 `FlowHeads` submodule。根因 = 那笔
"add FlowHeads submodule" 的 commit **只写进了 `.gitmodules` 条目,却没把 tree 里的
gitlink(mode `160000` commit 对象)提交进去**。结果:
- 本地 `FlowHeads/` 变 `?? FlowHeads/`(未跟踪)
- GitHub 拿到孤儿 `.gitmodules` 条目,渲染不出 submodule(页面什么都不显示)

**Why**:GitHub 渲染 submodule 需要**两个东西同时存在**:`.gitmodules` 里的 `[submodule]`
条目 **+** 该路径在 tree 里是一个 `160000` 的 gitlink。少了 gitlink = 孤儿配置。
常见触发 = `git add <path>` 时那次没真正 stage 进 gitlink(如先 `git rm --cached` 过、
或 reset/重做 commit 时漏掉),只剩 `.gitmodules` 被提交。

**验证**(任何"submodule 不显示"先跑这条):
```bash
git ls-tree HEAD <path>          # 必须看到: 160000 commit <sha>  <path>
                                 # 空 = gitlink 缺失(就是本 bug)
git ls-files --stage <path>      # index 里是否有 160000
```

**修复**:
```bash
git add <path>                   # 把当前 submodule HEAD 作为 gitlink 补进 index
                                 # (警告 "adding embedded git repository" 是正常的,
                                 #  因为 .gitmodules 已有条目 → 它就是正规 submodule)
git ls-files --stage <path>      # 确认 160000 <sha>
git commit -m "fix(<sub>): commit missing submodule gitlink"
```

**还要确认 gitlink 指向的 commit 已推到子模块远端**(否则 GitHub submodule 链接悬空):
`git ls-remote --heads <submodule-url>` 看远端 branch SHA == gitlink SHA。

**push 顺序铁律**(嵌套 submodule):**先推子模块仓 → 再推父仓**,否则父仓 gitlink 指向
远端不存在的 commit = 悬空。关联 [[feedback-commit-message-oneline]]、[[umbrella-leisaac-repo-boundary]]。
