<!--
  Auto-generated guidance for AI coding agents.
  This file was created because no existing agent instructions were found in the repository.
  Edit sections below when you discover project-specific conventions.
-->
# Copilot instructions — repository onboarding (scaffold)

Purpose
- Help an AI coding agent become productive quickly by describing discovery steps, common workflows, and where to look for project-specific conventions.

Note about this repository
- At the time this file was created there were no source files or README detected in the repository root. The agent MUST run the discovery checklist below to identify the project type and update this document with concrete commands once the code is found.

Discovery checklist (first actions)
1. List the repository root and important files:
   - Look for: `package.json`, `pyproject.toml`, `setup.py`, `requirements.txt`, `Pipfile`, `go.mod`, `Cargo.toml`, `pom.xml`, `build.gradle`, `Makefile`, `Rproj`, `DESCRIPTION`, `renv.lock`.
2. Identify language and framework from those manifests.
3. Inspect top-level directories: `src/`, `lib/`, `pkg/`, `app/`, `tests/`, `spec/`, `R/`, `notebooks/`.
4. Open the primary entry points (for example `src/index.*`, `main.*`, `app.py`, `server.go`).
5. Search for CI workflows (`.github/workflows/*.yml`) to find build/test commands used by maintainers.

Build & test mapping (what to try after discovery)
- Node.js / TypeScript: if `package.json` exists, prefer `npm`/`pnpm`/`yarn` script names in `scripts` (look for `build`, `test`, `lint`). Run the exact script defined, e.g.:
  - `npm run build` (or `npm test`, or `pnpm test`)
- Python: if `pyproject.toml`/`setup.py`/`requirements.txt` exist, look for `tox.ini`, `pytest` usage, or `scripts` in `pyproject`. Run `pytest -q` where tests are present.
- Go: if `go.mod` exists, run `go test ./...` and `go build ./...` in module root.
- Rust: if `Cargo.toml` exists, run `cargo test` and `cargo build`.
- R: if `DESCRIPTION` or `.Rproj` exists, run `R CMD check` or use `devtools::test()` when `tests/testthat` present.

Where tests and examples live
- Look in `tests/`, `spec/`, `__tests__/`, `tests/test_*.py`, `examples/`, and `notebooks/`. Prefer fast unit tests first.

Project-specific conventions — how to detect and follow them
- Coding style: search for linter configs (`.eslintrc`, `pyproject.toml` `[tool.black]`, `.clang-format`, `.prettierrc`) and follow the configured formatter/linter rules.
- Versioning & changelog: look for `CHANGELOG.md`, `version` fields in manifests. Match the existing version bump strategy in commits.
- Branching/PR: try to follow existing branch name patterns if present (search commit messages or open PRs). If none found, use `feat/<short-desc>` or `fix/<short-desc>`.

Integration points & external dependencies
- Check manifests for external services or SDKs (AWS/GCP/Azure clients, database client libs). Look for environment files (`.env.example`, `docker-compose.yml`) to infer local integration flows.
- If there are `Dockerfile` or `docker-compose.yml` files, prefer reproducing the containerized dev flow.

Editing rules for agents
- Only change files relevant to the requested task. Keep diffs small and explain intent in the PR description.
- Run the project's tests and linters locally before opening a PR. If tests fail, include failing output and a short plan to fix.

What to update in this document
- After the agent discovers build/test commands, update the "Build & test mapping" section with exact commands and any environment variables or setup steps specific to this repo (for example, `make bootstrap && npm ci && npm test`).

When you are blocked
- Report the exact file(s) or error output causing the blockage. If credentials or private services are required, document the minimal mock or emulator needed and suggest a fall-back (unit test with mocks, or environment variable-based short-circuits).

Where to look for examples
- Helpful places to update first: `README.md`, `.github/workflows/*`, `Makefile`, `CONTRIBUTING.md` — these often contain canonical commands.

Final note
- This repository had no discoverable source files when this scaffold was created. Please run the discovery checklist and update this file with exact commands and example files once code appears.
