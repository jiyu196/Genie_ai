import os
import threading
from datetime import datetime

def trace(tag: str, request_id: str = ""):
    pid = os.getpid()
    tid = threading.get_ident()
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    prefix = f"[{ts}] [PID:{pid}] [TID:{tid}]"
    if request_id:
        prefix += f" [RID:{request_id}]"

    print(f"{prefix} {tag}")

