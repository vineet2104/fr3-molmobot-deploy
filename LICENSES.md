# Origins & licensing — REVIEW BEFORE PUSHING TO A PERSONAL/PUBLIC REPO

This repo was assembled by copying/adapting files from internal working systems.
**Confirm redistribution rights for each origin before committing to a public or
personal git remote.** If a component cannot be redistributed, replace it with a
clean-room reimplementation from documented behavior, or fetch it at install
time from its authorized source.

| File(s) | Origin | Notes / action |
|---|---|---|
| `services/franky_service.py` | internal `franky_service` (`droid_plus`) | verify license; uses `franky-control` (3rd-party). |
| `services/robotiq_gripper_service.py` | internal `franky_service` (`droid_plus`) | verify license; only needed for Robotiq. |
| `services/franka_hand_service.py` | **new here** (uses `franky.Gripper`) | written for this repo. |
| `client/run_molmobot_official.py`, `client/run_molmobot_ft.py` | internal `sim2real-deploy/real_serve` | verify license. |
| `client/client_example.py`, `server/serve_bridge.py`, `server/bridge_policy.py` | internal `sim2real-deploy/sim_serve` (MolmoBot serving glue) | verify license/attribution to the MolmoBot project. |
| `service_console.py` | internal `sim2real-deploy` | verify license. |
| `tools/stitch_task_videos.py` | internal `sim2real-deploy/tools` | written for the reference project. |
| `docs/*` | internal design docs | sanitize hostnames/serials before publishing. |

**Do NOT commit:** model checkpoints, credentials/tokens, internal hostnames or
IPs, private package indexes. Real values live only in `configs/robot.yaml` /
`configs/robot.env` (git-ignored).

Third-party dependencies retain their own licenses (`franky-control`, libfranka,
FastAPI/uvicorn, numpy, opencv, websockets, msgpack-numpy, openpi-client, ZED
SDK / pyrealsense2, and the MolmoBot/`olmo` model code on the GPU host).
