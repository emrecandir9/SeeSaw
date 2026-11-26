# Adopted project context

The user adopted `docs/plan/SeeSaw_Implementation_Plan_v2.pdf` (25 Nov 2026) as the binding ownership and milestone plan on 26 Nov. Its SHA-256 is `a3df17c2eda2417445a681cbaa35c50962bb753713a84f06f6b29f3de4bf15d3`; it exactly matches the uploaded PDF. Earlier plan versions and demo branches are not the baseline.

Gloria owns MainXR, CaptureRig, project settings, OpenXR actions and XR/network scripts. Never edit Biswarup's DemoWorld or Yernur's ComparisonLens, LatencyHUD or MRMode scenes; coordinate via written interfaces. B1 owns rendering/capture depth; G1 defines the camera and stores the returned depth. See CONTRIBUTING and CODEOWNERS for the complete map.

Work from develop on owner/task branches and PR back into develop. Required policy: green `CI passed`, one current teammate approval, resolved review conversations; fresh changes need fresh approval. Do not push to protected branches or merge stacked work without checking prerequisites. Squash task PRs into develop; preserve main for tested releases.

G1 physical test is due 28 Nov on the lab PC and Quest 3S. The user does not currently have the headset and can obtain it soon. Desktop development continues; A1/A2 remain unverified until hardware and depth integration evidence exists. API freeze: 27 Nov. Milestones: 1 Dec single-model loop/model B choice, 8 Dec full integration, 9 Dec acceptance/release candidate. MR and post-release camera work stay off the critical path.

Use a dedicated 224×224 camera following head pose, saved capture context, top-left RGB PNG and capture depth. G2 must use one request in flight and match against the last **applied** result per session/model. No measured laptop timing is target-machine acceptance evidence.
