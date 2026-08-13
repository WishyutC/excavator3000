# Excavator3000 training dashboard

The local dashboard reads `runs/training.csv`, monitors the headless Webots
process, and edits an approved set of values in `config.py`.

## Start it

Double-click `Start_dashboard.bat`, or run:

```powershell
python dashboard.py
```

Open `http://127.0.0.1:8080` and enter the login token printed by the server.
The same token is stored locally in `.dashboard-token`. This file is ignored by
Git and must not be shared publicly.

Install dashboard dependencies on a new computer with:

```powershell
python -m pip install -r requirements-dashboard.txt
```

## Cloudflare Tunnel

Point the tunnel service at this origin:

```text
http://localhost:8080
```

For a temporary Quick Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8080
```

Keep the token login enabled even when Cloudflare Access is also configured.

## Controls

- **Start training** launches Webots with `--batch --mode=fast --no-rendering`.
- **Pause** suspends the managed Webots process tree without closing it.
- **Resume** continues a paused process.
- **Stop** terminates Webots. Work since the last checkpoint can be lost.
- Configuration changes are saved safely to `config.py` and apply only after
  Webots is started again.

The dashboard binds to `127.0.0.1`, so it is not directly exposed on the local
network. Cloudflare Tunnel is the external access layer.
