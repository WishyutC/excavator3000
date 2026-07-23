"""Standalone real-time HUD for telemetry sent by the Webots controller."""

import argparse
import json
import math
import queue
import socket
import threading
import tkinter as tk

from config import CONFIG


BACKGROUND = "#080d14"
PANEL = "#111a25"
TEXT = "#e8f0f7"
MUTED = "#8d9baa"
CYAN = "#27d7ff"
OBSERVER_CONFIG = CONFIG["observer"]

SENSOR_LAYOUT = [
    ("FRONT", 0, 350, 100),
    ("LEFT FRONT", 4, 145, 180),
    ("RIGHT FRONT", 5, 555, 180),
    ("LEFT", 2, 85, 325),
    ("RIGHT", 3, 615, 325),
    ("LEFT BACK", 6, 145, 470),
    ("RIGHT BACK", 7, 555, 470),
    ("BACK", 1, 350, 550)
]


def proximity_color(distance_m, raw_value):
    limits = OBSERVER_CONFIG["proximity_m"]

    if distance_m is not None:
        if distance_m <= limits["danger"]:
            return "#ff3b3b"
        if distance_m <= limits["warning"]:
            return "#ff941f"
        if distance_m <= limits["caution"]:
            return "#ffe04a"
        return "#35df76"

    if raw_value >= 3900:
        return "#ff3b3b"
    if raw_value >= 2800:
        return "#ff941f"
    if raw_value >= 1400:
        return "#ffe04a"
    return "#35df76"


class TelemetryReceiver(threading.Thread):
    def __init__(self, host, port, messages):
        super().__init__(daemon=True)
        self.messages = messages
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))

    def run(self):
        while True:
            try:
                payload, _ = self.socket.recvfrom(65535)
                message = json.loads(payload.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return

            try:
                self.messages.put_nowait(message)
            except queue.Full:
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    pass
                self.messages.put_nowait(message)

            if message.get("type") == "shutdown":
                return

    def close(self):
        self.socket.close()


class AgentHUD:
    def __init__(self, host, port):
        self.root = tk.Tk()
        self.root.title(OBSERVER_CONFIG["window_title"])
        self.root.geometry("1120x700")
        self.root.minsize(960, 620)
        self.root.configure(bg=BACKGROUND)

        self.canvas = tk.Canvas(
            self.root,
            bg=BACKGROUND,
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.messages = queue.Queue(maxsize=1)
        self.receiver = TelemetryReceiver(host, port, self.messages)
        self.receiver.start()
        self.latest = None

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self.poll)
        self.draw()

    def poll(self):
        try:
            while True:
                message = self.messages.get_nowait()
                if message.get("type") == "shutdown":
                    self.close()
                    return
                self.latest = message
        except queue.Empty:
            pass

        self.draw()
        self.root.after(40, self.poll)

    def draw_sensor(self, name, sensor_index, center_x, center_y):
        sensor = self.latest["sensors"][sensor_index]
        raw_value = float(sensor["raw"])
        distance = sensor.get("distance_m")
        color = proximity_color(distance, raw_value)

        x1, y1 = center_x - 82, center_y - 33
        x2, y2 = center_x + 82, center_y + 33
        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill=PANEL,
            outline=color,
            width=3
        )
        self.canvas.create_text(
            center_x,
            center_y - 10,
            text=name,
            fill=color,
            font=("Segoe UI", 11, "bold")
        )

        distance_text = (
            f"{distance * 100:.1f} cm"
            if distance is not None
            else "NO LOOKUP TABLE"
        )
        self.canvas.create_text(
            center_x,
            center_y + 12,
            text=f"{distance_text}  |  raw {raw_value:.0f}",
            fill=TEXT,
            font=("Consolas", 9)
        )

    def draw_speedometer(self, center_x, center_y, radius):
        speed = max(0.0, float(self.latest.get("linear_speed_m_s", 0.0)))
        max_speed = max(
            0.01,
            float(self.latest.get("max_linear_speed_m_s", speed))
        )
        ratio = min(1.0, speed / max_speed)

        bounds = (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        )
        self.canvas.create_arc(
            *bounds,
            start=0,
            extent=180,
            style="arc",
            outline="#26384a",
            width=13
        )
        self.canvas.create_arc(
            *bounds,
            start=180 - ratio * 180,
            extent=ratio * 180,
            style="arc",
            outline=CYAN,
            width=13
        )

        for tick in range(6):
            tick_ratio = tick / 5
            angle = math.pi - tick_ratio * math.pi
            inner = radius - 17
            outer = radius - 5
            x1 = center_x + math.cos(angle) * inner
            y1 = center_y - math.sin(angle) * inner
            x2 = center_x + math.cos(angle) * outer
            y2 = center_y - math.sin(angle) * outer
            self.canvas.create_line(x1, y1, x2, y2, fill=MUTED, width=2)

        needle_angle = math.pi - ratio * math.pi
        needle_x = center_x + math.cos(needle_angle) * (radius - 25)
        needle_y = center_y - math.sin(needle_angle) * (radius - 25)
        self.canvas.create_line(
            center_x,
            center_y,
            needle_x,
            needle_y,
            fill="#ff5964",
            width=4
        )
        self.canvas.create_oval(
            center_x - 7,
            center_y - 7,
            center_x + 7,
            center_y + 7,
            fill="#ff5964",
            outline=""
        )

        self.canvas.create_text(
            center_x,
            center_y - 45,
            text=f"{speed * 100:.1f}",
            fill=TEXT,
            font=("Consolas", 25, "bold")
        )
        self.canvas.create_text(
            center_x,
            center_y - 18,
            text="cm/s",
            fill=MUTED,
            font=("Segoe UI", 10, "bold")
        )
        self.canvas.create_text(
            center_x,
            center_y + 22,
            text=f"MAX {max_speed * 100:.1f} cm/s",
            fill=MUTED,
            font=("Consolas", 9)
        )

    def draw(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            32,
            25,
            anchor="w",
            text="EXCAVATOR 3000  /  PROXIMITY HUD",
            fill=CYAN,
            font=("Segoe UI", 18, "bold")
        )
        self.canvas.create_line(32, 52, 1088, 52, fill="#233244", width=2)

        if self.latest is None:
            self.canvas.create_text(
                560,
                350,
                text="WAITING FOR WEBOTS TELEMETRY...",
                fill=MUTED,
                font=("Segoe UI", 16, "bold")
            )
            return

        for sensor_data in SENSOR_LAYOUT:
            self.draw_sensor(*sensor_data)

        self.canvas.create_oval(
            292,
            267,
            408,
            383,
            fill="#152b3b",
            outline=CYAN,
            width=4
        )
        self.canvas.create_polygon(
            350,
            280,
            332,
            315,
            368,
            315,
            fill=CYAN,
            outline=""
        )
        self.canvas.create_text(
            350,
            342,
            text="ROBOT",
            fill=TEXT,
            font=("Segoe UI", 13, "bold")
        )

        self.canvas.create_rectangle(
            720,
            70,
            1085,
            585,
            fill=PANEL,
            outline="#233244",
            width=2
        )
        self.canvas.create_text(
            748,
            92,
            anchor="w",
            text="AGENT STATE",
            fill=MUTED,
            font=("Segoe UI", 11, "bold")
        )

        self.draw_speedometer(902, 235, 115)

        details = [
            (
                "EPISODE / STEP",
                f'{self.latest["episode"]} / {self.latest["step"]}',
                TEXT
            ),
            (
                "DECISION",
                f'{self.latest["action_name"]}  [{self.latest["action"]}]',
                CYAN
            ),
            (
                "WHEEL SPEED",
                (
                    f'L {self.latest["left_speed"]:+.1f}   '
                    f'R {self.latest["right_speed"]:+.1f}'
                ),
                TEXT
            ),
            (
                "TARGET / TURN RATE",
                (
                    f'{self.latest.get("target_linear_speed_m_s", 0.0) * 100:+.1f} cm/s   '
                    f'{self.latest.get("angular_speed_rad_s", 0.0):+.2f} rad/s'
                ),
                TEXT
            ),
            ("REWARD", f'{self.latest["reward"]:+.3f}', "#ffd166"),
            ("TOTAL REWARD", f'{self.latest["total_reward"]:+.3f}', "#ffd166")
        ]

        for index, (label, value, color) in enumerate(details):
            y = 356 + index * 38
            self.canvas.create_text(
                748,
                y,
                anchor="w",
                text=label,
                fill=MUTED,
                font=("Segoe UI", 9, "bold")
            )
            self.canvas.create_text(
                748,
                y + 19,
                anchor="w",
                text=value,
                fill=color,
                font=("Consolas", 12, "bold")
            )

        limits = OBSERVER_CONFIG["proximity_m"]
        legend = (
            f'RED <{limits["danger"] * 100:.0f} cm   '
            f'ORANGE <{limits["warning"] * 100:.0f} cm   '
            f'YELLOW <{limits["caution"] * 100:.0f} cm   GREEN clear'
        )
        self.canvas.create_text(
            720,
            610,
            anchor="w",
            text=legend,
            fill=MUTED,
            font=("Segoe UI", 9)
        )
        self.canvas.create_text(
            720,
            640,
            anchor="w",
            text="LIVE  UDP 127.0.0.1",
            fill="#35df76",
            font=("Consolas", 10, "bold")
        )

    def close(self):
        self.receiver.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Webots RL telemetry HUD")
    parser.add_argument("--host", default=OBSERVER_CONFIG["host"])
    parser.add_argument("--port", type=int, default=OBSERVER_CONFIG["port"])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    AgentHUD(arguments.host, arguments.port).run()
