"""
llama_manager.py
Quan ly vong doi tien trinh llama-server.exe:
  - start()  : khoi dong neu chua chay, cho /health tra ve "ok"
  - stop()   : gui taskkill de dung tien trinh
  - healthy(): kiem tra nhanh /health
"""
import json
import os
import subprocess
import time
import urllib.request

import config as cfg


def healthy() -> bool:
    """Tra ve True neu llama-server dang chay va san sang."""
    try:
        with urllib.request.urlopen(cfg.LLAMA_HEALTH_URL, timeout=1.0) as r:
            status = json.loads(r.read().decode())
            return status.get("status") == "ok"
    except Exception:
        return False


def start(timeout: int = 60) -> None:
    """
    Khoi dong llama-server.exe voi cac tham so GPU toi uu.
    Cho toi da `timeout` giay cho den khi server san sang.
    Raises RuntimeError neu qua thoi gian.
    """
    if healthy():
        print("[llama-manager] llama-server dang chay san, bo qua khoi dong.")
        return

    exe = cfg.LLAMA_SERVER_EXE
    if not os.path.exists(exe):
        raise FileNotFoundError(f"Khong tim thay llama-server.exe: {exe}")

    model = cfg.MODEL_PATH
    if not model or not os.path.exists(model):
        raise FileNotFoundError(
            f"Khong tim thay model '{cfg.MODEL_NAME}'. "
            "Dat model cung thu muc voi script hoac thiet lap MODEL_PATH trong .env"
        )

    cmd = [
        exe,
        "-m", model,
        "--port", str(cfg.LLAMA_PORT),
        "-c",  str(cfg.LLAMA_CONTEXT),
        "--threads", str(cfg.LLAMA_THREADS),
        "-ngl", str(cfg.LLAMA_NGL),
        "--parallel", str(cfg.LLAMA_PARALLEL),
        "-b",  str(cfg.LLAMA_BATCH),
        "-ub", str(cfg.LLAMA_UBATCH),
        "-fa", "on",
        "--no-warmup",
    ]

    # Tren Windows: chay ngam, tach khoi cua so console hien tai
    creationflags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED + NEW_PROCESS_GROUP + CREATE_NO_WINDOW

    # Thiet lap PATH cho CUDA & Vulkan DLLs
    env = os.environ.copy()
    ollama_lib_dir = os.path.dirname(exe)
    cuda_v13_path = os.path.join(ollama_lib_dir, "cuda_v13")
    vulkan_path = os.path.join(ollama_lib_dir, "vulkan")
    env["PATH"] = f"{cuda_v13_path};{vulkan_path};{env.get('PATH', '')}"

    print(f"[llama-manager] Khoi dong llama-server tren port {cfg.LLAMA_PORT} ...")
    print(f"[llama-manager] Model: {model}")
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
        env=env,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1)
        if healthy():
            print("[llama-manager] llama-server san sang.")
            return

    raise RuntimeError(
        f"llama-server khong khoi dong duoc sau {timeout} giay. "
        "Kiem tra lai duong dan exe / model va log thu cong."
    )


def stop() -> None:
    """Dung tien trinh llama-server.exe (bang taskkill tren Windows)."""
    print("[llama-manager] Dang dung llama-server.exe ...")
    subprocess.run(
        ["taskkill", "/f", "/im", "llama-server.exe"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[llama-manager] Da gui lenh dung llama-server.exe.")
