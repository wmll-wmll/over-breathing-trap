"""V11 ↔ Godot 的 TCP 桥。

协议沿用 v10.7（新行分隔的 JSON），Godot 是服务端，Python 连过去。

    Godot → Python（观测）:
        {"t":..., "pos":[x,y,z], "light":..., "light_dir":[dx,dy,dz],
         "light_var":..., "danger":..., "energy_pool":...}

    Python → Godot（动作）:
        {"action":"move", "velocity":[vx,vy,vz], "t":..., "light":..., ...}

★ 桥本身不含任何智能 —— 它只负责收发。所有因果都在场和 Transformer 里。
"""

from __future__ import annotations

import json
import socket


class GodotBridge:
    """极简 TCP 客户端：连 Godot、收观测、发动作。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 9002,
                 connect_timeout: float = 15.0):
        self.host = host
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self._sock: socket.socket | None = None
        self._buf: str = ""
        self._connected = False

    # ── 连接 ────────────────────────────────────────────────────────
    def connect(self, retries: int = 20, retry_delay: float = 1.0) -> bool:
        """连 Godot。Godot 启动需要时间，所以默认会重试。"""
        import time

        for attempt in range(1, retries + 1):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.connect_timeout)
            try:
                s.connect((self.host, self.port))
                s.settimeout(None)
                self._sock = s
                self._connected = True
                print(f"[Godot] 已连接 {self.host}:{self.port}")
                return True
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                s.close()
                if attempt == retries:
                    print(f"[Godot] 连接失败（试了 {retries} 次）: {e}")
                    return False
                print(f"[Godot] 等 Godot 起来… ({attempt}/{retries})")
                time.sleep(retry_delay)
        return False

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ── 收观测 ──────────────────────────────────────────────────────
    def recv(self, timeout: float = 10.0) -> dict | None:
        """阻塞收一条观测。超时或断连返回 None。"""
        if not self._connected or not self._sock:
            return None
        self._sock.settimeout(timeout)
        while "\n" not in self._buf:
            try:
                chunk = self._sock.recv(65536)
            except (socket.timeout, OSError):
                return None
            if not chunk:
                self._connected = False
                return None
            self._buf += chunk.decode("utf-8", errors="replace")
        line, self._buf = self._buf.split("\n", 1)
        line = line.strip()
        if not line:
            return self.recv(timeout)
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    # ── 发动作 ──────────────────────────────────────────────────────
    def send_action(self, velocity, energy_pool: float | None = None,
                    t: float = 0.0, light: float = 0.0, dE: float = 0.0) -> bool:
        if not self._connected or not self._sock:
            return False
        msg = {
            "action": "move",
            "velocity": [float(velocity[0]), float(velocity[1]), float(velocity[2])],
            "t": round(float(t), 1),
            "light": round(float(light), 4),
            "dE": round(float(dE), 1),
        }
        if energy_pool is not None:
            msg["energy_pool"] = round(float(energy_pool), 1)
        try:
            self._sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            return True
        except OSError:
            self._connected = False
            return False
