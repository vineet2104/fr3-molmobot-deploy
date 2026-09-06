"""Enumerate ZED cameras + their serials (run on the workstation)."""
import pyzed.sl as sl
for d in sl.Camera.get_device_list():
    print(f"SN={d.serial_number}  model={d.camera_model}  state={d.camera_state}")
