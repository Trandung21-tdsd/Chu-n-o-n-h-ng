"""
TATO Unified Training & Telemetry Runner.
Automatically launches hardware telemetry logging in the background,
executes your deep learning training command, and cleanly stops/saves
the telemetry CSV as soon as training finishes (no manual Ctrl+C required).
Works seamlessly on Linux and Windows.
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# Import standalone telemetry helper if present in same folder
try:
    from telemetry_logger_standalone import main as run_standalone_logger
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="TATO Unified Train & Telemetry Runner")
    parser.add_argument("--cmd", type=str, required=True, help="Lenh train cua ban (vd: 'python3 train.py --epochs 10')")
    parser.add_argument("--model", type=str, default="ResNet-18", help="Ten kien truc model (vd: ResNet-18, MobileNetV3)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--precision", type=str, default="AMP", help="Precision (FP32 hoac AMP)")
    parser.add_argument("--tag", type=str, default="", help="Nhan cho lan chay nay (vd: RUN_001)")
    args = parser.parse_args()

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger_script = os.path.join(script_dir, "telemetry_logger_standalone.py")

    # 1. Start Telemetry Logger as a background subprocess
    logger_cmd = [
        sys.executable, logger_script,
        "--model", args.model,
        "--batch_size", str(args.batch_size),
        "--precision", args.precision,
        "--tag", tag
    ]

    print("=" * 65)
    print("   TATO Unified Runner: Training AI + Telemetry Tu Dong")
    print("=" * 65)
    print(f"[*] Model: {args.model} | Batch: {args.batch_size} | Precision: {args.precision}")
    print(f"[*] Lenh train se thuc thi: {args.cmd}")
    print("[*] Dang khoi dong Telemetry Logger 1Hz chay ngam...")

    logger_proc = subprocess.Popen(logger_cmd)
    time.sleep(1.5)  # allow logger to initialize NVML

    # 2. Run the actual AI training command
    print("\n" + "=" * 65)
    print(f"   [BAT DAU HUAN LUYEN] -> {args.cmd}")
    print("=" * 65 + "\n")

    t0 = time.time()
    try:
        train_proc = subprocess.run(args.cmd, shell=True)
        train_code = train_proc.returncode
    except KeyboardInterrupt:
        print("\n[!] Nguoi dung huy tien trinh train.")
        train_code = 130

    total_time_s = round(time.time() - t0, 1)

    # 3. Cleanly stop Telemetry Logger
    print("\n" + "=" * 65)
    print(f"[*] Training hoan tat sau {total_time_s} giay! Dang dung Telemetry...")
    print("=" * 65)

    try:
        # Send SIGINT (Ctrl+C) to logger process so it flushes CSV cleanly
        import signal
        if sys.platform != "win32":
            logger_proc.send_signal(signal.SIGINT)
        else:
            logger_proc.terminate()
        logger_proc.wait(timeout=5)
    except Exception:
        logger_proc.kill()

    print(f"\n[+] HOAN TAT 100%! Da sinh ra file telemetry dong bo voi buoi train.")
    sys.exit(train_code)


if __name__ == "__main__":
    main()
