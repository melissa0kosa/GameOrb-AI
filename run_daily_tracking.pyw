"""
Windowless launcher for the daily tracker (used by Windows Task Scheduler via pythonw.exe).

The old .bat opened a console window at 10:00; closing that window killed the run.
This runs with no window at all and appends everything to data/tracking.log.
"""
import os
import sys
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data", exist_ok=True)
log = open(os.path.join("data", "tracking.log"), "a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = log
print(f"===== {datetime.now():%a %m/%d/%Y %H:%M:%S} (windowless) =====")

# Idle sleep was cutting runs short (Windows logged them as terminated). Ask Windows to
# stay awake while this process runs; it reverts automatically when the process exits.
# This does not block a manual shutdown, sleep from the Start menu or closing a laptop lid.
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
try:
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
except Exception:
    pass

sys.argv = ["track_upcoming.py"] + sys.argv[1:]
try:
    import track_upcoming
    track_upcoming.main()
except SystemExit:
    pass
except Exception:
    import traceback
    traceback.print_exc()
finally:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass
