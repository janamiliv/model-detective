"""Start and stop a pinned llama-server as a subprocess."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_BIN = Path("tools/llama.cpp/bin/llama-server.exe")


@dataclass
class LlamaServer:
    """Context manager around one llama-server process serving one GGUF.

    Device policy is explicit: ``n_gpu_layers=0`` with ``device="none"`` runs on CPU,
    which is what this machine needs until the NVIDIA driver is updated (docs/hardware.md).
    """

    model_path: Path
    port: int = 8080
    binary: Path = DEFAULT_BIN
    n_gpu_layers: int = 0
    device: str | None = "none"
    ctx_size: int = 2048
    threads: int | None = None
    parallel: int = 1
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_s: float = 120.0
    log_path: Path | None = None
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        cmd = [
            str(self.binary),
            "-m",
            str(self.model_path),
            "--port",
            str(self.port),
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.n_gpu_layers),
            "--parallel",
            str(self.parallel),
            "--no-webui",
        ]
        if self.device is not None:
            cmd += ["--device", self.device]
        if self.threads is not None:
            cmd += ["-t", str(self.threads)]
        return cmd + list(self.extra_args)

    def start(self) -> LlamaServer:
        if not self.binary.exists():
            raise FileNotFoundError(f"llama-server binary not found: {self.binary}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL  # noqa: SIM115
        self._proc = subprocess.Popen(
            self.command(), stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        self._wait_healthy()
        return self

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        with httpx.Client(timeout=2.0) as http:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited early with code {self._proc.returncode}"
                    )
                try:
                    r = http.get(f"{self.base_url}/health")
                    if r.status_code == 200 and r.json().get("status") == "ok":
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server not healthy after {self.startup_timeout_s}s")

    def build_info(self) -> str:
        with httpx.Client(timeout=5.0) as http:
            return str(http.get(f"{self.base_url}/props").json().get("build_info", ""))

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._proc = None

    def __enter__(self) -> LlamaServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
