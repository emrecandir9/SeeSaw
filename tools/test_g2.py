"""Run isolated Godot G2 lifecycle checks against the existing mock sidecar.

Use a Python environment with sidecar/requirements.txt installed.
No Godot scenes or project settings are created in the repository.
"""
import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True, help="Godot 4.7.2 executable")
    args = parser.parse_args()
    processes = []
    with tempfile.TemporaryDirectory(prefix="seesaw-g2-") as folder:
        project = Path(folder)
        shutil.copytree(ROOT / "godot/scripts/network", project / "scripts/network")
        shutil.copy(ROOT / "sidecar/fixtures/contract/paired_valid.json", project)
        (project / "project.godot").write_text('config_version=5\n[application]\nconfig/name="SeeSaw G2 tests"\n', encoding="utf-8")
        env = os.environ.copy()
        ports = [free_port(), free_port(), free_port()]
        env["SEESAW_TEST_ENDPOINT"] = f"http://127.0.0.1:{ports[0]}/probe"
        env["SEESAW_SLOW_ENDPOINT"] = f"http://127.0.0.1:{ports[1]}/probe"
        env["SEESAW_OFFLINE_ENDPOINT"] = f"http://127.0.0.1:{ports[2]}/probe"
        try:
            with (project / "server.log").open("w+") as log:
                for port, delay in zip(ports[:2], [0, 500]):
                    processes.append(subprocess.Popen(
                        [sys.executable, "server.py", "--mock", "--host", "127.0.0.1", "--port", str(port), "--delay-ms", str(delay)],
                        cwd=ROOT / "sidecar", stdout=log, stderr=log,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
                for port in ports[:2]:
                    deadline = time.monotonic() + 15
                    while True:
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                                break
                        except OSError:
                            if time.monotonic() > deadline:
                                log.seek(0)
                                raise RuntimeError("Mock startup failed:\n" + log.read())
                            time.sleep(0.1)
                run = subprocess.run([args.godot, "--headless", "--path", str(project), "--script", "res://scripts/network/test_probe_client.gd"], env=env, timeout=60)
                return run.returncode
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
