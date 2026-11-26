# See/Saw

A Meta Quest 3S VR proof of concept for probing Vision Transformer attention in a 3D scene. The user aims a controller and captures a view. A Python sidecar runs a frozen DINOv2 ViT-S/14 on the capture, groups the patch features into eight clusters, upsamples them with FeatUp, and returns the groups with their attention. The Godot client paints those groups back onto the scene geometry and colors them by attention. A world-space lens shows a second model's grouping of the same capture, so the two models can be compared on the same surfaces.

The project reimplements *See/Saw: Embodied Probing of Vision Transformer Attention in Virtual Reality* (Xie and Lin, SIGGRAPH Posters 2026) for the Augmented and Virtual Reality course, University of Bordeaux. Everything is built from scratch in this repository.

## Status

Sprint: 25 November to 9 December 2026. G1 now has a standalone desktop/XR capture harness. Open `godot/project.godot` in **Godot 4.7.2-stable** and run it; desktop is the default and requires no headset. See [G1 controls, interfaces and pending acceptance](docs/g1-capture.md). Depth rendering, backend integration and physical Quest acceptance remain pending.

| Date | Milestone |
| --- | --- |
| 1 Dec | Single-model end-to-end VR loop on the lab PC with measured latency; model B chosen |
| 8 Dec | All mandatory features integrated and tested |
| 9 Dec | Clean-checkout acceptance and release candidate |
| 15 / 18 Dec | Defense / final submission |

## Plan

The full implementation plan is in [docs/plan/SeeSaw_Implementation_Plan_v2.pdf](docs/plan/SeeSaw_Implementation_Plan_v2.pdf). It covers scope, hardware, ownership, architecture, the `/probe` API contract, the model B options, the day-by-day schedule, the issue register and the acceptance matrix (A1 to A12).

## Team

| Member | Area |
| --- | --- |
| Emre Candir ([@emrecandir9](https://github.com/emrecandir9)) | Python sidecar: DINOv2, attention, K-means, FeatUp, matching, model B, mock mode |
| Gloria García Cuenco ([@gloriacuenco](https://github.com/gloriacuenco)) | XR, capture rig, project settings, networking, final assembly |
| Biswarup Mukherjee ([@BiswarupMukherjee1](https://github.com/BiswarupMukherjee1)) | Demo scene, shaders, projection and overlay |
| Yernur Polatbek ([@KurosJS](https://github.com/KurosJS)) | Comparison lens, latency HUD, optional MR, test documentation |

Each Godot scene has exactly one editor. Ask the owner to change a scene you don't own.

## Working on this repository

- `main` holds tested releases only. Day-to-day work goes into `develop`.
- Branch from `develop` as `<name>/<issue>-<short-description>`, for example `emre/E1-sidecar-baseline`.
- Open a pull request into `develop`. CI must pass and one teammate must approve before it can merge.
- Releases go from `develop` to `main` through a pull request and are tagged `vX.Y.Z` (for example `v0.1.0-rc1`).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full rules.
