"""Subprocess client for the patched dflash daemon (with park/unpark commands)."""
from __future__ import annotations
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
from typing import Optional

from . import config

log = logging.getLogger(__name__)


class DflashClient:
    def __init__(self, bin_path: str, target_path: str, draft_path: str,
                 max_ctx: int = 16384, ddtree_budget: int = 16,
                 *,
                 ddtree_temp: Optional[float] = None,
                 chain_seed: bool = True,
                 fa_window: Optional[int] = None,
                 kv_tq3: Optional[bool] = None,
                 lm_head_fix: Optional[bool] = None,
                 boot_timeout_s: float = 180.0,
                 boot_vram_mib: int = 18000):
        """Spawn the patched dflash daemon as a subprocess.

        Defaults for fa_window / kv_tq3 / lm_head_fix come from
        ``config.DFLASH_REQUIRED_ENV`` and are the only flags pflash relies on.
        Override per-call only when you know what you're doing.
        """
        self.bin_path = bin_path
        self.target_path = target_path
        self.draft_path = draft_path
        self.max_ctx = max_ctx
        env_overrides = {
            "DFLASH27B_FA_WINDOW": str(0 if fa_window is None else fa_window),
            "DFLASH27B_KV_TQ3": "1" if (kv_tq3 if kv_tq3 is not None else True) else "0",
            "DFLASH27B_LM_HEAD_FIX": "1" if lm_head_fix else "0",
        }
        env = {**os.environ, **config.DFLASH_REQUIRED_ENV, **env_overrides}
        bin_dir = os.path.dirname(os.path.abspath(bin_path))
        if sys.platform == "win32":
            # Windows uses PATH for DLL search rather than LD_LIBRARY_PATH.
            path_extra = [os.path.join(bin_dir, "bin"), bin_dir]
            env["PATH"] = os.pathsep.join(
                path_extra + ([env["PATH"]] if env.get("PATH") else []))
        else:
            ld_extra = [bin_dir, os.path.join(bin_dir, "bin")]
            env["LD_LIBRARY_PATH"] = ":".join(
                ld_extra + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        self.r_pipe, self.w_pipe = os.pipe()
        # subprocess.Popen pass_fds is POSIX-only. On Windows, mark the pipe
        # write-end inheritable, convert it to a Win32 HANDLE, and pass that
        # handle value as --stream-fd; the child inherits via close_fds=False.
        if sys.platform == "win32":
            import msvcrt
            os.set_inheritable(self.w_pipe, True)
            stream_fd_val = int(msvcrt.get_osfhandle(self.w_pipe))
        else:
            stream_fd_val = self.w_pipe
        cmd = [bin_path, target_path, draft_path,
               "--daemon", "--fast-rollback", "--ddtree",
               f"--ddtree-budget={ddtree_budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
        if ddtree_temp is not None:
            cmd.append(f"--ddtree-temp={ddtree_temp}")
        if not chain_seed:
            cmd.append("--ddtree-no-chain-seed")
        log.info("spawning dflash daemon: %s", " ".join(cmd))
        if sys.platform == "win32":
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         close_fds=False, env=env)
        else:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         pass_fds=(self.w_pipe,), env=env)
        os.close(self.w_pipe)
        self._wait_until_loaded(timeout=boot_timeout_s, vram_mib=boot_vram_mib)
        # Park draft by default; user calls unpark when needed
        self._send("park draft\n")

    def _wait_until_loaded(self, timeout: float = 180.0, vram_mib: int = 18000):
        boot = time.time()
        daemon_pid = self.proc.pid if self.proc else None
        while time.time() - boot < timeout:
            time.sleep(1)
            vram = self._read_gpu_vram_mib()
            if vram is not None and vram > vram_mib:
                return
            # On UMA architectures (Strix Halo iGPU gfx1151 etc.), the
            # static "VRAM" buffer is small (~8 GiB BIOS-set) and most of
            # the daemon's 17 GiB target allocation lands in HMM-mapped
            # host memory, invisible to rocm-smi. Fall back to checking
            # the daemon process RSS — UMA allocations show up there.
            rss = self._read_daemon_rss_mib(daemon_pid) if daemon_pid else None
            if rss is not None and rss > vram_mib:
                return
        raise RuntimeError(
            f"dflash daemon failed to load target weights within {timeout:.0f}s "
            f"(expected VRAM or daemon RSS > {vram_mib} MiB). "
            f"Check the daemon's stderr.")

    @staticmethod
    def _read_daemon_rss_mib(pid: int) -> Optional[int]:
        """Return resident-set size (MiB) of the daemon process.

        Used as a UMA fallback: on Strix Halo gfx1151 (and similar HMM
        systems) the daemon's weight allocations live in host memory,
        not in rocm-smi's static VRAM buffer."""
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        # Format: "VmRSS:  17430964 kB"
                        return int(parts[1]) // 1024
        except (FileNotFoundError, OSError, ValueError, IndexError):
            return None
        return None

    @staticmethod
    def _read_gpu_vram_mib() -> Optional[int]:
        """Return max used VRAM (MiB) across visible GPUs, CUDA or HIP.

        Tries ``nvidia-smi`` first; on HIP/ROCm hosts falls back to
        ``rocm-smi --showmeminfo vram --json``. Returns the max across
        all enumerated cards so the boot probe works regardless of which
        card the daemon bound to. Returns ``None`` if neither tool is
        available or both produced unparseable output."""
        import json as _json
        # CUDA path
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode()
            return max(int(line.strip()) for line in out.splitlines() if line.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        # ROCm path. `rocm-smi` may not be on $PATH on ROCm installs that
        # don't source /etc/profile.d/rocm.sh in non-login shells — fall back
        # to the canonical install location.
        import shutil as _shutil
        rocm_smi = _shutil.which("rocm-smi") or (
            "/opt/rocm/bin/rocm-smi" if os.path.exists("/opt/rocm/bin/rocm-smi") else None)
        if rocm_smi is None:
            return None
        try:
            out = subprocess.check_output(
                [rocm_smi, "--showmeminfo", "vram", "--json"],
                stderr=subprocess.DEVNULL).decode()
            data = _json.loads(out)
            used_bytes_per_card = []
            for card in data.values():
                if not isinstance(card, dict):
                    continue
                used = card.get("VRAM Total Used Memory (B)")
                if used is not None:
                    used_bytes_per_card.append(int(used))
            if used_bytes_per_card:
                return max(used_bytes_per_card) // (1024 * 1024)
            return None
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError,
                KeyError, _json.JSONDecodeError):
            return None

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        # Read until -1 sentinel
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            if struct.unpack("<i", b)[0] == -1:
                break

    def free_drafter(self): self._send("free drafter\n")
    def park_draft(self):    self._send("park draft\n")
    def unpark_draft(self):  self._send("unpark draft\n")
    def park_target(self):   self._send("park target\n")
    def unpark_target(self): self._send("unpark target\n")

    def compress(self, prompt_ids: list[int], keep_ratio: float, drafter_gguf: str) -> list[int]:
        """C++ drafter score+compress via daemon. Returns compressed token ids.

        Daemon command: compress <bin> <keep_x1000> <drafter_gguf>
        """
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        keep_x1000 = int(round(keep_ratio * 1000))
        self.proc.stdin.write(f"compress {path} {keep_x1000} {drafter_gguf}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def generate(self, prompt_ids: list[int], n_gen: int) -> list[int]:
        """Send prompt + n_gen request, return generated token ids."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        self.proc.stdin.write(f"{path} {n_gen}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def close(self, timeout: float = 5.0):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
