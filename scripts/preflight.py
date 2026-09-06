"""Non-motion preflight for the workstation. Verifies services + model reachable
before a rollout. Exits nonzero on failure. Never commands motion.

    python scripts/preflight.py
Reads endpoints from the environment (source configs/robot.env first).
"""
import os
import sys
import time
import urllib.request
import json


def _get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode()
            try:
                return r.status, json.loads(body)
            except Exception:
                return r.status, body
    except Exception as e:
        return None, str(e)


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(' — ' + str(detail)) if detail else ''}")
    return ok


def main():
    franky = os.environ.get("FRANKY_URL", "http://127.0.0.1:54321")
    hand = os.environ.get("HAND_URL", "http://127.0.0.1:54324")
    robotiq = os.environ.get("ROBOTIQ_URL", "")
    gpu_host = os.environ.get("MOLMOBOTFT_A100_HOST", "")
    gpu_port = os.environ.get("MOLMOBOTFT_A100_PORT", "8000")
    ok_all = True

    print("== arm service ==")
    st, js = _get(f"{franky}/health")
    a = check(f"franky /health @ {franky}", st == 200, js)
    if a and isinstance(js, dict):
        check("  robot_connected", bool(js.get("robot_connected")), js.get("last_control_error"))
        if js.get("stop_latched"):
            print("  [warn] stop_latched=True — clear via /go_home before motion")
    st, js = _get(f"{franky}/joint_state")
    check("franky /joint_state (7 positions)",
          st == 200 and isinstance(js, dict) and len(js.get("positions", [])) == 7, js if st != 200 else "")
    ok_all &= a

    print("== gripper service ==")
    gbackend = os.environ.get("GRIPPER_BACKEND", "franka_hand")
    if gbackend == "robotiq" and robotiq:
        st, js = _get(f"{robotiq}/health"); check(f"robotiq /health @ {robotiq}", st == 200, js)
    elif gbackend == "none":
        print("  [skip] gripper disabled")
    else:
        st, js = _get(f"{hand}/health"); check(f"franka_hand /health @ {hand}", st == 200, js)

    print("== model server (GPU host) ==")
    if gpu_host:
        st, body = _get(f"http://{gpu_host}:{gpu_port}/healthz")
        ok_all &= check(f"model /healthz @ {gpu_host}:{gpu_port}", st == 200, body)
    else:
        print("  [warn] MOLMOBOTFT_A100_HOST not set — set it in configs/robot.env")

    print("== cameras ==")
    try:
        import pyzed.sl as sl
        ds = sl.Camera.get_device_list()
        want = {os.environ.get("ZED_EXO_FRONT_SN"), os.environ.get("ZED_WRIST_SN")}
        want = {int(x) for x in want if x and x.isdigit()}
        avail = {d.serial_number for d in ds if str(d.camera_state) == "AVAILABLE"}
        for d in ds:
            print(f"  ZED {d.serial_number} {d.camera_model} {d.camera_state}")
        if want:
            check("required ZEDs available", want.issubset(avail), f"need {want}, have {avail}")
    except Exception as e:
        print(f"  [warn] ZED enumeration skipped: {e}")

    print("\nPREFLIGHT:", "PASS" if ok_all else "FAIL (see above)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
