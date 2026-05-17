---
name: coding-safety
description: "Use when coding changes need caution over speed: surface assumptions, keep edits surgical, prefer the simplest solution, and verify with a goal-driven loop."
---

# Coding Safety

Use this skill when the task is a code change and the main risk is overconfidence, unnecessary complexity, or broad edits.

## Workflow

1. State assumptions up front.
2. Identify any unclear or ambiguous parts before coding.
3. Choose the smallest solution that fully meets the request.
4. Make only the changes directly required by the task.
5. Verify the result against a clear success condition.

## Decision Rules

- If multiple interpretations exist, name them and ask before implementing.
- If a simpler approach solves the request, prefer it.
- If a change would expand scope, defer it unless the user asked for it.
- If adjacent code looks imperfect but is unrelated, leave it alone.

## Editing Principles

- Keep diffs surgical.
- Match the existing style of the file and project.
- Remove only imports, variables, or functions that your change makes unused.
- Do not add abstractions, configurability, or error handling unless the request requires them.

## Verification Loop

For non-trivial changes:

1. Define what success looks like.
2. Implement the minimum change.
3. Check the relevant tests, build, or runtime behavior.
4. If the result does not match the goal, revise the smallest possible part.

## Completion Check

Treat the task as complete only when:

- The request is satisfied directly.
- The change is as small as practical.
- The behavior is verified or the remaining uncertainty is clearly stated.
