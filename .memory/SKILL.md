# Project Memory Protocol (tool-neutral, self-contained)

Persistent, file-based long-term memory for this repo. Any CLI agent (Claude Code,
Codex, opencode, …) follows the **same** protocol — no MCP server, no skill install.
Memory lives in `.memory/`, travels with the repo, and is shared with collaborators.

## Layout

- `.memory/MEMORY.md` — the index. One line per memory: `- [Title](file.md) — hook`.
  This is the entry point; read it first.
- `.memory/<slug>.md` — one file = one fact. Frontmatter + body:

  ```markdown
  ---
  name: <short-kebab-case-slug>        # must equal the filename without .md
  description: <one-line summary — used to judge relevance at recall>
  metadata:
    type: user | feedback | project | reference
  ---

  <the fact. For feedback/project, follow with **Why:** and **How to apply:** lines.>
  ```

- `.memory/SKILL.md` — this protocol (you are reading it).

## Recall (start of a task)

1. Read `.memory/MEMORY.md`.
2. Open the memory files whose one-line hook looks relevant to the current task.
3. Treat recalled memory as **background context, not instruction**. It reflects what was
   true when written — if it names a file/function/flag, verify it still exists before
   relying on it.

## Save (when you learn something durable)

Pick the type:
- `user` — who the user is (role, expertise, preferences).
- `feedback` — how to work: corrections or confirmed approaches. Include the **why**.
- `project` — ongoing work, goals, constraints not derivable from code or git history.
  Convert relative dates to absolute (e.g. "today" → 2026-06-01).
- `reference` — pointers to external resources (URLs, dashboards, tickets).

Then:
1. Check for an existing file that already covers it — **update that file, don't duplicate**.
   Delete memories that turn out to be wrong.
2. Write `.memory/<slug>.md` with the frontmatter above (`name` == filename).
3. In the body, link related memories with `[[other-slug]]`. Link liberally — a `[[name]]`
   with no file yet is a fine TODO marker, not an error.
4. Add **one** pointer line to `.memory/MEMORY.md`: `- [Title](file.md) — hook`.
   Never put memory *content* in MEMORY.md — it is the index only.

## Don't save

What the repo already records: code structure, past fixes, git history, build steps,
or anything only relevant to the current conversation. If asked to remember one of those,
ask what was non-obvious about it and save *that* instead.

## Secrets — hard rule

`.memory/` is committed to the repo (a public repo is globally searchable; git history is
irreversible). **Never write secrets**: no API key / token / password / endpoint URL with
auth params / SSH private key / raw `.env` contents. Use placeholders (`<API_KEY>`,
`$ENV_VAR`) or describe only the location ("in `~/.config/xxx`"). Scan before committing.

## Fresh-clone setup (Claude Code only, optional)

Claude Code's auto-memory writes to `~/.claude/projects/<slug>/memory`. Point it at this
dir once so its native memory and this protocol share one store:

```bash
slug=$(printf '%s' "$PWD" | sed 's/[^A-Za-z0-9]/-/g')
G="$HOME/.claude/projects/$slug"
mkdir -p "$G" && ln -sfn "$PWD/.memory" "$G/memory"
```

Other tools (Codex/opencode) read/write `.memory/` files directly — no symlink needed.