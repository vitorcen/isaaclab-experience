---
name: feedback-html-doc-rules
description: 写中文 HTML 设计/复盘文档时必须遵守的样式规则（颜色、SVG、自包含）
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# HTML 文档样式规则（写中文设计/复盘文档时必带）

写在 `LeIsaac/docs/training/*.html` 之类的中文复盘 / 设计 / 训练计划 HTML 文档时，**强制**遵守这些规则，否则用户在 dark mode 浏览器下打开会"背景有问题字看不清"。

**Why:** 我写 `gr00t_n17_sim_wire_protocol_debug.html` 时只设 `body { color: #1d1d1f }` 没设 background，用户在 dark mode 浏览器打开，浏览器把 body 默认背景变成黑色（dark mode UA），结果黑底 + 深灰文字 = 不可读；2026-05-23 用户直接报"背景有问题字看不清"。

**How to apply:** 任何 HTML 文档第一段 CSS 必须包含下面 3 行，所有手写的 background/color 都要成对配。

## 强制样式规则

1. **强制浅色主题**，禁止浏览器 dark mode 自动反色：
```css
:root { color-scheme: light; }
html, body { background: #ffffff; }
body { ... color: #1d1d1f; background: #ffffff; }
```
不能只设 `color` 不设 `background`，反之亦然。同一元素必须 **成对** 设。

2. **SVG 容器** 也要显式 `background`（`<svg style="background:#fafafa">`），不然 dark mode 下 SVG 透明背景 = 黑底，曲线/文字看不见。

3. **代码块**（pre / code）：
   - `pre { background: #1d1d1f; color: #f5f5f7; }`（深色块 + 浅色文字）
   - `pre code { background: transparent; color: inherit; }`（不要嵌套覆盖）
   - inline `code { background: #f5f5f7; color: inherit; }`（浅色 chip + 默认深字）

4. **彩色 callout**（`.tldr` / `.stage` / `.warn-box` 等）：每个都必须显式写 `background` + `color`，不能依赖继承。
```css
.tldr { background: #fff7e6; color: #1d1d1f; border-left: 5px solid #ff9500; }
.stage { background: #f5f5f7; color: #1d1d1f; }
```

5. **不要外链图片/字体**。HTML 文档"单文件可分享、可离线浏览"是默认目标 ([[feedback-style.md]])。框图用内嵌 SVG（`<svg>...</svg>` in body, 不是 `<img src=...>`）。

6. **代码块字号** ≥ 13px，行高 1.5；中文字号 ≥ 14px，行高 1.7。手机/平板可读。

7. **table 必带 border** + `background: #f5f5f7` 给 th；`vertical-align: top` 给 td。

## 模板片段（可直接复用）

```html
<style>
  :root { color-scheme: light; }
  html, body { background: #ffffff; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         max-width: 1080px; margin: 0 auto; padding: 24px 32px;
         line-height: 1.7; color: #1d1d1f; background: #ffffff; }
  h1 { border-bottom: 3px solid #0a84ff; padding-bottom: 10px; }
  h2 { color: #0a84ff; border-left: 4px solid #0a84ff; padding-left: 10px; margin-top: 36px; }
  code { background: #f5f5f7; color: inherit; padding: 1px 6px; border-radius: 4px;
         font-family: SF Mono, Menlo, monospace; font-size: 0.9em; }
  pre { background: #1d1d1f; color: #f5f5f7; padding: 14px 18px; border-radius: 8px;
        overflow-x: auto; font-size: 13px; line-height: 1.5; }
  pre code { background: transparent; color: inherit; padding: 0; }
  .tldr { background: #fff7e6; color: #1d1d1f; border-left: 5px solid #ff9500;
          padding: 14px 18px; border-radius: 0 6px 6px 0; margin: 16px 0 24px; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 0.95em; }
  th, td { border: 1px solid #d2d2d7; padding: 8px 12px; text-align: left; vertical-align: top; }
  th { background: #f5f5f7; color: #1d1d1f; font-weight: 600; }
</style>
```

## 关联

- [[feedback-style]] — HTML + 中文 + inline SVG + 单文件可分享是默认目标
- HTML 实战修复例子：`LeIsaac/docs/training/gr00t_n17_sim_wire_protocol_debug.html` (2026-05-23 修复 dark mode 黑底深字 bug)
