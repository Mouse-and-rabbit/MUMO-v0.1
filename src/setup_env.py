"""
MUMO — environment helper
Makes MUMO run identically on Windows (local) and Linux (Streamlit Cloud).

ensure_vina() returns a usable AutoDock Vina executable path:
  - On Windows: uses the bundled bin/vina.exe
  - On Linux (cloud): uses 'vina' if already on PATH, otherwise downloads the
    official static Linux binary once and caches it in bin/vina.
"""

import os
import sys
import stat
import shutil
import platform
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(BASE, "bin")

# Official AutoDock Vina 1.2.5 static Linux binary
VINA_LINUX_URL = ("https://github.com/ccsb-scripps/AutoDock-Vina/releases/"
                  "download/v1.2.5/vina_1.2.5_linux_x86_64")


def ensure_vina():
    """Return a path to a working vina executable for this operating system."""
    os.makedirs(BIN, exist_ok=True)

    # Windows: use the bundled .exe we already have locally.
    if platform.system() == "Windows":
        return os.path.join(BIN, "vina.exe")

    # Linux/Mac: prefer an already-installed vina (PATH, or the conda/venv bin).
    on_path = shutil.which("vina")
    if on_path:
        return on_path
    conda_vina = os.path.join(os.path.dirname(sys.executable), "vina")
    if os.path.exists(conda_vina):
        return conda_vina

    # Otherwise download the static Linux binary once and cache it.
    local = os.path.join(BIN, "vina")
    if not os.path.exists(local):
        urllib.request.urlretrieve(VINA_LINUX_URL, local)
        # make it executable (chmod +x)
        st = os.stat(local)
        os.chmod(local, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return local
