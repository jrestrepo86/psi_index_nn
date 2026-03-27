---
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git commit:*)
description: Generate a commit message from staged changes, ask for approval, then commit
---

## Context

- Current branch: !`git branch --show-current`
- Staged changes (what will be committed): !`git diff --cached`
- Working tree status: !`git status`
- Recent commits (style reference): !`git log --oneline -10`

## Your task

**Step 1 — Draft a commit message**
Analyze the staged diff above and write a commit message:

- Subject line: imperative mood, ≤72 characters
- Add a blank line + body only if the change genuinely needs more explanation

**Step 2 — Show and ask for approval**
Display the proposed message in a code block, then ask:

> ¿Aprobás este commit? Respondé: **sí** / **no** / **editar: \<nuevo mensaje\>**

**Step 3 — Act on the response**

- `sí` → run `git commit -m "<proposed message>"`
- `no` → abort, do nothing, tell the user the commit was cancelled
- `editar: <texto>` → use `<texto>` as the commit message and run `git commit -m "<texto>"`

**Rules:**

- Do NOT run `git add` — the user stages changes manually
- Do NOT commit before receiving explicit approval in Step 3
