"""Own, discover, and control the local headless Webots training process."""

import json
import os
from pathlib import Path
import subprocess
import threading
import time

import psutil


class TrainingProcessManager:
    def __init__(self, project_root, world_path=None, webots_path=None):
        self.project_root = Path(project_root).resolve()
        self.world_path = Path(
            world_path or self.project_root.parent.parent / "worlds" / "car_env_test_v2.wbt"
        ).resolve()
        self.webots_path = Path(
            webots_path
            or os.environ.get("WEBOTS_EXE", "")
            or r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
        )
        self.runtime_path = self.project_root / ".dashboard-runtime.json"
        self.log_path = self.project_root / "runs" / "dashboard-webots.log"
        self._lock = threading.Lock()

    def _read_pid(self):
        try:
            return int(json.loads(self.runtime_path.read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _write_pid(self, pid):
        self.runtime_path.write_text(json.dumps({"pid": int(pid)}), encoding="utf-8")

    @staticmethod
    def _is_webots(process):
        try:
            return process.name().lower() in {"webots.exe", "webots"}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _matching_process(self):
        saved_pid = self._read_pid()
        if saved_pid:
            try:
                process = psutil.Process(saved_pid)
                if process.is_running() and self._is_webots(process):
                    return process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        world_name = self.world_path.name.lower()
        candidates = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if (process.info["name"] or "").lower() not in {"webots.exe", "webots"}:
                    continue
                command = " ".join(process.info["cmdline"] or []).lower()
                if world_name in command and "--mode=fast" in command:
                    candidates.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if len(candidates) == 1:
            self._write_pid(candidates[0].pid)
            return candidates[0]
        return None

    @staticmethod
    def _tree(process):
        try:
            return [process, *process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return [process]

    def status(self):
        process = self._matching_process()
        if process is None:
            return {"state": "stopped", "pid": None, "started_at": None}
        try:
            state = "paused" if process.status() == psutil.STATUS_STOPPED else "running"
            return {
                "state": state,
                "pid": process.pid,
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(process.create_time())
                )
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"state": "stopped", "pid": None, "started_at": None}

    def start(self):
        with self._lock:
            current = self.status()
            if current["state"] != "stopped":
                raise RuntimeError("Webots training is already running.")
            if not self.webots_path.is_file():
                raise RuntimeError(f"Webots executable was not found: {self.webots_path}")
            if not self.world_path.is_file():
                raise RuntimeError(f"Webots world was not found: {self.world_path}")

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_path.open("a", encoding="utf-8")
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            process = subprocess.Popen(
                [
                    str(self.webots_path), "--batch", "--mode=fast",
                    "--no-rendering", "--stdout", "--stderr",
                    str(self.world_path)
                ],
                cwd=self.project_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags
            )
            log_handle.close()
            self._write_pid(process.pid)
            time.sleep(0.4)
            if process.poll() is not None:
                raise RuntimeError(
                    f"Webots exited immediately. Check {self.log_path.name}."
                )
            return self.status()

    def pause(self):
        with self._lock:
            process = self._matching_process()
            if process is None:
                raise RuntimeError("No matching Webots training process is running.")
            processes = self._tree(process)
            for item in reversed(processes):
                try:
                    item.suspend()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return self.status()

    def resume(self):
        with self._lock:
            process = self._matching_process()
            if process is None:
                raise RuntimeError("No paused Webots training process was found.")
            for item in self._tree(process):
                try:
                    item.resume()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return self.status()

    def stop(self):
        with self._lock:
            process = self._matching_process()
            if process is None:
                return self.status()
            processes = self._tree(process)
            for item in reversed(processes):
                try:
                    if item.status() == psutil.STATUS_STOPPED:
                        item.resume()
                    item.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            _, alive = psutil.wait_procs(processes, timeout=5)
            for item in alive:
                try:
                    item.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.runtime_path.unlink(missing_ok=True)
            return self.status()
