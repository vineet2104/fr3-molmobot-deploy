"""Franka Desk web-session client.

Drives the same HTTPS API that the Desk web UI uses. Endpoints lifted from
franky.RobotWebSession; trimmed to the calls START needs (login, take control,
unlock brakes, enable FCI). See:
    https://github.com/TimSchneider42/franky/blob/main/franky/robot_web_session.py

Credentials are read from ~/.config/franka/desk.json:
    {"host": "10.20.13.189", "user": "...", "password": "..."}
File should be chmod 600.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import ssl
import time
import urllib.parse
from http.client import HTTPSConnection
from pathlib import Path

CREDS_PATH = Path.home() / ".config" / "franka" / "desk.json"


def _encode_password(user: str, password: str) -> str:
    digest = hashlib.sha256(f"{password}#{user}@franka".encode("utf-8")).digest()
    bs = ",".join(str(b) for b in digest)
    return base64.encodebytes(bs.encode("utf-8")).decode("utf-8")


class DeskError(RuntimeError):
    pass


class DeskSession:
    def __init__(self, host: str, user: str, password: str, timeout: float = 15.0) -> None:
        self.host = host
        self.user = user
        self._password = password
        self._timeout = timeout
        self._conn: HTTPSConnection | None = None
        self._token: str | None = None
        self._control_token: str | None = None
        self._control_token_id: int | None = None

    # ----- context -----

    def __enter__(self) -> "DeskSession":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._control_token is not None:
                self.release_control()
        finally:
            self.close()

    # ----- low level -----

    def _request_once(
        self, method: str, path: str, headers: dict | None = None, body: str | None = None
    ) -> bytes:
        assert self._conn is not None
        h: dict = {}
        if self._token is not None:
            h["Cookie"] = f"authorization={self._token}"
        if headers:
            h.update(headers)
        self._conn.request(method, path, headers=h, body=body)
        r = self._conn.getresponse()
        data = r.read()
        if r.status != 200:
            raise DeskError(f"{method} {path} -> {r.status} {r.reason}: {data[:200]!r}")
        return data

    def _request(
        self, method: str, path: str, headers: dict | None = None, body: str | None = None
    ) -> bytes:
        last: Exception | None = None
        for _ in range(3):
            try:
                return self._request_once(method, path, headers=headers, body=body)
            except http.client.RemoteDisconnected as e:
                last = e
        assert last is not None
        raise last

    def _control_request(
        self, method: str, path: str, headers: dict | None = None, body: str | None = None
    ) -> bytes:
        if self._control_token is None:
            raise DeskError("No control token. Call take_control() first.")
        h = {"X-Control-Token": self._control_token}
        if headers:
            h.update(headers)
        return self._request(method, path, headers=h, body=body)

    # ----- lifecycle -----

    def open(self) -> None:
        self._conn = HTTPSConnection(
            self.host, timeout=self._timeout, context=ssl._create_unverified_context()
        )
        self._conn.connect()
        body = json.dumps(
            {"login": self.user, "password": _encode_password(self.user, self._password)}
        )
        self._token = self._request(
            "POST", "/admin/api/login", headers={"content-type": "application/json"}, body=body
        ).decode("utf-8")

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
                self._token = None

    # ----- ops -----

    def get_system_status(self) -> dict:
        return json.loads(self._request("GET", "/admin/api/system-status").decode("utf-8"))

    def has_control(self) -> bool:
        if self._control_token_id is None:
            return False
        active = self.get_system_status().get("controlToken", {}).get("activeToken")
        return active is not None and active.get("id") == self._control_token_id

    def take_control(self, force: bool = True, wait_timeout: float = 30.0) -> None:
        """Request the control token; if force=True the user must press the
        physical button on the FR3 pendant within ``wait_timeout`` seconds."""
        if self.has_control():
            return
        res = self._request(
            "POST",
            f"/admin/api/control-token/request{'?force' if force else ''}",
            headers={"content-type": "application/json"},
            body=json.dumps({"requestedBy": self.user}),
        )
        payload = json.loads(res)
        self._control_token = payload["token"]
        self._control_token_id = payload["id"]
        if force:
            print(
                f"[desk] FORCE: press the circle button on top of the FR3 "
                f"pendant within {int(wait_timeout)}s to confirm control."
            )
        start = time.time()
        while time.time() - start < wait_timeout:
            if self.has_control():
                return
            time.sleep(0.5)
        raise DeskError(f"Timed out waiting for control after {wait_timeout}s.")

    def release_control(self) -> None:
        if self._control_token is None:
            return
        try:
            self._control_request(
                "DELETE",
                "/admin/api/control-token",
                headers={"content-type": "application/json"},
                body=json.dumps({"token": self._control_token}),
            )
        finally:
            self._control_token = None
            self._control_token_id = None

    def unlock_brakes(self) -> None:
        self._control_request(
            "POST",
            "/desk/api/joints/unlock",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    def lock_brakes(self) -> None:
        self._control_request(
            "POST",
            "/desk/api/joints/lock",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    def enable_fci(self) -> None:
        assert self._control_token is not None
        token_b64 = urllib.parse.quote(base64.b64encode(self._control_token.encode("ascii")))
        self._control_request(
            "POST",
            "/desk/api/system/fci",
            headers={"content-type": "application/x-www-form-urlencoded"},
            body=f"token={token_b64}",
        )

    def disable_fci(self) -> None:
        self._control_request("DELETE", "/desk/api/system/fci")


def load_creds() -> tuple[str, str, str]:
    if not CREDS_PATH.exists():
        raise FileNotFoundError(
            f"Desk credentials not found at {CREDS_PATH}. "
            'Create with {"host":"10.20.13.189","user":"...","password":"..."} (chmod 600).'
        )
    data = json.loads(CREDS_PATH.read_text())
    return data["host"], data["user"], data["password"]


def start_robot() -> None:
    """One-shot: log in, take control (force), unlock brakes, enable FCI."""
    host, user, password = load_creds()
    print(f"[desk] connecting to {host} as {user}")
    with DeskSession(host, user, password) as s:
        print("[desk] taking control (press the FR3 circle button if prompted)...")
        s.take_control(force=True)
        print("[desk] unlocking brakes...")
        s.unlock_brakes()
        print("[desk] enabling FCI...")
        s.enable_fci()
        print("[desk] robot is ready.")


if __name__ == "__main__":
    start_robot()
