# LEARN

What was built in each phase, why it is designed that way, and what the alternative would have been.

## Phase 0: Machine setup and repository

**What we built.** An empty but complete project skeleton: the folder layout, a README stub, an MIT license, a `.gitignore`, an empty `.env.example`, and a public GitHub repository with `main` pushed.

**Why it is designed this way.**
- *Folder layout first.* Deciding where things live before writing code keeps each file small and with one job (`data.py` only loads data, `baseline.py` only does the baseline, and so on). The `src/intentbench/` "src layout" means tests import the installed package, not a stray file from the current folder, which catches packaging mistakes early.
- *`.gitignore` before the first commit.* Once a file is committed, it lives in git history forever, even if you delete it later. Ignoring `.env` (secrets), model weights (hundreds of MB) and caches from day one means they can never be committed by accident.
- *`.env.example`.* Shows anyone cloning the repo which settings they must provide, without exposing real values. The real `.env` stays only on your machine.
- *`.gitkeep` files.* Git tracks files, not folders. An empty placeholder keeps the planned folders visible on GitHub.
- *MIT license.* Without a license, nobody is legally allowed to reuse public code. MIT is the shortest common permissive license, so it's a sensible default for a portfolio project.

**Alternatives.** A project template generator (such as cookiecutter) could have produced this skeleton, but it adds files you didn't choose and can't explain. A flat layout (code at the repo root) is simpler, but it lets tests pass by accident because they import local files instead of the installed package.
