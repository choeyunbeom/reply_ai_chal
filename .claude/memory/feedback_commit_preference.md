---
name: commit preference
description: User handles all git commits themselves; Claude should never run git commit
type: feedback
---

User handles all git commits personally. Never run git commit.

**Why:** User wants full control over commit history.

**How to apply:** When a commit is needed, provide the git command as a text snippet the user can copy-paste. Do not call the Bash tool to commit.
