# Contributing

## Branches

| Branch | Purpose | How it changes |
| --- | --- | --- |
| `main` | Tested releases | Pull request from `develop` only |
| `develop` | Integration branch, always runnable | Pull requests from task branches |
| `<name>/<issue>-<topic>` | One task or issue | Push freely |

Both `main` and `develop` are protected: no direct pushes, no force pushes, no deletion.

## Pull requests

Every pull request into `main` or `develop` needs:

1. **CI passing.** The `CI passed` check must be green.
2. **One approval from a teammate.** You cannot approve your own pull request. A new push after approval needs a fresh approval.
3. **All review conversations resolved.**

In the description, link the issue (`Closes #12`), list the scenes you changed, and say how and on which machine you tested it (lab PC, laptop desktop mode, or mock sidecar). The template asks for all of this.

Keep pull requests small and merge them daily where possible. Use **squash merge** for task branches into `develop`, and a **merge commit** for `develop` into `main`. Branches are deleted automatically after merging.

## Scene ownership

Only the named owner edits a Godot scene (`.tscn`). To change someone else's scene, open an issue or ask them. [.github/CODEOWNERS](.github/CODEOWNERS) requests a review from the owner automatically when their files change.

| Owner | Scenes and files |
| --- | --- |
| Emre | `sidecar/` |
| Gloria | `MainXR.tscn`, `CaptureRig.tscn`, `project.godot`, OpenXR action map, `scripts/xr/`, `scripts/network/` |
| Biswarup | `DemoWorld.tscn`, `shaders/`, `scripts/overlay/` |
| Yernur | `ComparisonLens.tscn`, `LatencyHUD.tscn`, `MRMode.tscn`, `scripts/ui/` |

## Issues

Use the **Task** template for planned work. Give each issue its ID from the plan's issue register (E1, G2, B1 …), its owner, its due date, its dependencies and the acceptance IDs it closes (A1 to A12). Use the **Bug** template for defects.

## CI

CI runs on every pull request and on every push to `main` and `develop`. It currently checks:

- no merge conflict markers
- no files larger than 50 MB
- no committed `.godot/` cache, `.env` files or private keys
- Python tests in `sidecar/`, once that folder has `requirements.txt` and `tests/`
- Godot 4.7.2-stable imports `godot/` headless and parses every `.gd` file without errors, once `godot/project.godot` exists

When you add a CI job, add it to the `needs` list of the `CI passed` job, so the branch rules never need changing.

## Versions

Everyone uses **Godot 4.7.2-stable** (standard build, not .NET). Don't open the project in another version: it rewrites project and scene files. To change the version, update `GODOT_VERSION` in [.github/workflows/ci.yml](.github/workflows/ci.yml) and this section in the same pull request.

## Releases

Pushing a tag such as `v0.1.0-rc1` on `main` creates a GitHub release with generated notes. Record the acceptance evidence (commit, versions, machine, latencies) in the release description.
