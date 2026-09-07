#!/usr/bin/env python3
"""Capture one RGB snapshot from each attached ZED and RealSense camera.

This does not contact the robot or model server. Cameras must not be owned by
another process. Images are written under camera_snapshots/ by default.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def capture_zed(out: Path) -> list[dict]:
    try:
        import pyzed.sl as sl
    except Exception as exc:
        print(f"[ZED] unavailable: {exc}")
        return []

    results = []
    devices = list(sl.Camera.get_device_list())
    print(f"[ZED] found {len(devices)} device(s)")
    for device in devices:
        serial = int(device.serial_number)
        camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.set_from_serial_number(serial)
        status = camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] SN {serial}: open failed: {status}")
            continue
        try:
            image = sl.Mat()
            runtime = sl.RuntimeParameters()
            for _ in range(30):
                if camera.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                    camera.retrieve_image(image, sl.VIEW.LEFT)
                time.sleep(0.01)
            rgba = np.asarray(image.get_data())
            if rgba.size == 0:
                raise RuntimeError("empty frame")
            bgr = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
            path = out / f"zed_{serial}_left.png"
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError("cv2.imwrite failed")
            info = camera.get_camera_information()
            result = {
                "backend": "zed",
                "serial": serial,
                "model": str(info.camera_model),
                "shape": list(bgr.shape),
                "file": str(path),
            }
            results.append(result)
            print(f"[ZED] SN {serial}: saved {path} shape={bgr.shape}")
        except Exception as exc:
            print(f"[ZED] SN {serial}: capture failed: {exc}")
        finally:
            camera.close()
    return results


def capture_realsense(out: Path) -> list[dict]:
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        print(f"[RealSense] unavailable: {exc}")
        return []

    results = []
    context = rs.context()
    devices = list(context.query_devices())
    print(f"[RealSense] found {len(devices)} device(s)")
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(config)
            color = None
            for _ in range(30):
                frames = pipeline.wait_for_frames(3000)
                frame = frames.get_color_frame()
                if frame:
                    color = np.asanyarray(frame.get_data()).copy()
            if color is None or color.size == 0:
                raise RuntimeError("empty color frame")
            path = out / f"realsense_{serial}_color.png"
            if not cv2.imwrite(str(path), color):
                raise RuntimeError("cv2.imwrite failed")
            result = {
                "backend": "realsense",
                "serial": serial,
                "model": name,
                "shape": list(color.shape),
                "file": str(path),
            }
            results.append(result)
            print(f"[RealSense] SN {serial}: saved {path} shape={color.shape}")
        except Exception as exc:
            print(f"[RealSense] SN {serial}: capture failed: {exc}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output or root / "camera_snapshots" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results = capture_zed(out) + capture_realsense(out)
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[done] {len(results)} snapshot(s); manifest: {manifest}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
