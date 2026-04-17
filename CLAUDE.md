# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository is in an **initial / empty state**. As of the first commit, it contains only:

- `README.md` — project title only (`# OmniOptimizer`), no description yet
- `LICENSE` — MIT License
- `.gitignore` — standard Python template (covers `__pycache__`, `.venv`, `pytest`, `mypy`, `ruff`, `uv`, `poetry`, `pdm`, `marimo`, Django/Flask artifacts, etc.)

There is no source code, no build system, no dependency manifest (`pyproject.toml` / `requirements.txt`), and no tests yet. The project's purpose, scope, and architecture are **not yet defined in-repo**.

## Implications for Claude Code

- **Do not invent architecture, modules, or commands.** Until real code lands, there is nothing to document beyond this file. If asked about how the project is structured, say so rather than guessing.
- **Language assumption:** The `.gitignore` is Python-focused, so new code should be assumed to be Python unless the user states otherwise. No specific Python version, package manager, test runner, or linter has been chosen yet — ask the user before picking one.
- **When the first real code is added,** update this file with:
  - The chosen package manager (`uv` / `poetry` / `pdm` / `pip`) and the exact install/run/test/lint commands
  - The top-level module layout and what "OmniOptimizer" actually optimizes (problem domain, inputs, outputs)
  - Any non-obvious architectural decisions that span multiple files

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
