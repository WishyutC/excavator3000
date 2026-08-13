"""Local training control and analytics dashboard for excavator3000."""

from functools import wraps
import os
from pathlib import Path
import secrets

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from waitress import serve

from dashboard_config import public_fields, update_config
from dashboard_data import TrainingDataReader
from dashboard_process import TrainingProcessManager


PROJECT_ROOT = Path(__file__).resolve().parent


def _secret(path, environment_name, length=32):
    from_environment = os.environ.get(environment_name)
    if from_environment:
        return from_environment
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    value = secrets.token_urlsafe(length)
    path.write_text(value, encoding="utf-8")
    return value


def create_app(project_root=PROJECT_ROOT):
    root = Path(project_root).resolve()
    app = Flask(
        __name__,
        template_folder=str(root / "templates"),
        static_folder=str(root / "static")
    )
    app.secret_key = _secret(root / ".dashboard-secret", "DASHBOARD_SECRET")
    dashboard_token = _secret(root / ".dashboard-token", "DASHBOARD_TOKEN", 18)
    csrf_token = secrets.token_urlsafe(24)
    data_reader = TrainingDataReader(root / "runs" / "training.csv")
    process_manager = TrainingProcessManager(root)
    config_path = root / "config.py"

    def authenticated(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if not session.get("authenticated"):
                if request.path.startswith("/api/"):
                    return jsonify({"ok": False, "error": "Authentication required."}), 401
                return redirect(url_for("login"))
            return function(*args, **kwargs)
        return wrapper

    def valid_csrf():
        return secrets.compare_digest(
            request.headers.get("X-CSRF-Token", ""), session.get("csrf_token", "")
        )

    @app.after_request
    def security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            supplied = request.form.get("token", "")
            if secrets.compare_digest(supplied, dashboard_token):
                session.clear()
                session["authenticated"] = True
                session["csrf_token"] = csrf_token
                return redirect(url_for("index"))
            error = "Incorrect dashboard token."
        return render_template("login.html", error=error)

    @app.post("/logout")
    @authenticated
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    @authenticated
    def index():
        return render_template("dashboard.html", csrf_token=session["csrf_token"])

    @app.get("/api/dashboard")
    @authenticated
    def dashboard_data():
        return jsonify({
            "ok": True,
            "training": data_reader.snapshot(),
            "process": process_manager.status()
        })

    @app.get("/api/config")
    @authenticated
    def get_config():
        return jsonify({"ok": True, "fields": public_fields(config_path)})

    @app.post("/api/config")
    @authenticated
    def save_config():
        if not valid_csrf():
            return jsonify({"ok": False, "error": "Invalid request token."}), 403
        payload = request.get_json(silent=True) or {}
        try:
            update_config(config_path, payload.get("values", {}))
            return jsonify({
                "ok": True,
                "fields": public_fields(config_path),
                "message": "Configuration saved. It applies the next time Webots starts."
            })
        except (ValueError, KeyError, SyntaxError, OSError) as error:
            return jsonify({"ok": False, "error": str(error)}), 400

    @app.post("/api/process/<action>")
    @authenticated
    def process_action(action):
        if not valid_csrf():
            return jsonify({"ok": False, "error": "Invalid request token."}), 403
        actions = {
            "start": process_manager.start,
            "pause": process_manager.pause,
            "resume": process_manager.resume,
            "stop": process_manager.stop
        }
        if action not in actions:
            return jsonify({"ok": False, "error": "Unsupported process action."}), 404
        try:
            return jsonify({"ok": True, "process": actions[action]()})
        except (RuntimeError, OSError, ValueError) as error:
            return jsonify({"ok": False, "error": str(error)}), 409

    app.config["DASHBOARD_TOKEN"] = dashboard_token
    return app


def main():
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    app = create_app()
    print("\nExcavator3000 training dashboard")
    print(f"Local URL: http://127.0.0.1:{port}")
    print(f"Login token: {app.config['DASHBOARD_TOKEN']}")
    print("Cloudflare Tunnel origin: http://localhost:8080\n")
    serve(app, host="127.0.0.1", port=port, threads=8)


if __name__ == "__main__":
    main()
