# How to Use Claude Code Optimally

Claude Code works best when you treat it like a focused senior pair programmer, not a general chat assistant. Give it a narrow task, the relevant files, and a clear definition of done.

## 1. Start With Context, Not Vague Goals

Good input:

- What you want changed
- Where the change should happen
- Constraints that matter
- How you will verify success

Example:

```text
Refactor the FastAPI endpoint in `api.py` to use a shared validation helper.
Keep behavior unchanged.
Add or update tests if needed.
```

Avoid:

- "Improve this code"
- "Make it better"
- "Fix everything"

## 2. Ask for Small, Complete Chunks

Claude Code is most reliable when each task is bounded.

Best practice:

- One feature
- One bug
- One file group
- One test target

If the work is large, split it into phases:

1. Inspect and summarize the current implementation.
2. Propose the minimum safe change.
3. Implement it.
4. Verify with tests or a quick run.

## 3. Give It the Right Files

When possible, point Claude Code at the exact files involved.

Useful input:

- Relevant source files
- Related tests
- Config files that affect behavior
- Error logs or stack traces

This reduces guessing and speeds up the first useful response.

## 4. Be Explicit About Constraints

State what must not change.

Examples:

- Preserve public API behavior
- Keep the diff small
- Do not add new dependencies
- Match existing style
- Avoid broad refactors

Claude Code can make stronger decisions when the boundaries are clear.

## 5. Ask for Verification, Not Just Edits

Optimal use is edit plus proof.

Ask it to:

- Run tests
- Check formatting
- Explain any failing step
- Summarize the behavior change

If you cannot run tests, ask it to tell you exactly what to run.

## 6. Use Iteration, Not One-Shot Prompts

The best workflow is usually:

- First pass: inspect
- Second pass: change
- Third pass: verify and refine

This is better than asking for a perfect final answer on the first prompt.

## 7. Keep Prompts Concrete

Strong prompt pattern:

```text
In `linked_list.py`, add a `delete_at_index` method.
Follow the style already used in the file.
Update or add tests if they exist.
Do not change unrelated methods.
```

Weak prompt pattern:

```text
Make the linked list class more robust.
```

## 8. Prefer Direct Instructions Over Over-Explaining

Claude Code does not need long background unless the context is actually important.

Use short instructions such as:

- "Patch only the bug"
- "Do not rewrite the module"
- "Preserve compatibility"
- "Show the exact files changed"

## 9. Review the Diff Before Merging

Even strong models can over-edit.

Check for:

- Unnecessary refactors
- Behavior changes outside scope
- Missing edge cases
- Test gaps
- Style drift

If the diff is larger than expected, tighten the task and rerun it.

## 10. A Practical Workflow

1. Describe the task in one paragraph.
2. Point Claude Code to the relevant files.
3. Specify constraints and success criteria.
4. Let it propose or make a minimal change.
5. Verify with tests or a quick manual check.
6. Ask for a short explanation of what changed and why.

## 11. Common Mistakes

- Asking for too much at once
- Not providing file paths
- Forgetting to mention constraints
- Accepting a huge diff without review
- Skipping verification

## 12. A Good Default Prompt

```text
You are working in this codebase.
Inspect the relevant files first.
Then make the smallest safe change to solve the issue.
Keep the existing design intact.
Run or suggest verification steps.
Report exactly what changed.
```

## Bottom Line

Claude Code is most effective when you give it:

- Clear scope
- Relevant context
- Tight constraints
- A verification step

That combination produces better code and less cleanup.
