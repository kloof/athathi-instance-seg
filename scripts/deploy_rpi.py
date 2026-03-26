"""Deploy inference to Raspberry Pi and run benchmark."""

import paramiko
import os
import time

HOST = "192.168.8.111"
USER = "talal"
PASS = "talal"
REMOTE_DIR = "/home/talal/Desktop/test_inference"

LOCAL_BASE = r"C:\Users\klof\Desktop\New folder (4)\wall-seg"

# Files to copy
FILES = {
    "src/infer.py": "infer.py",
    "src/preprocess.py": "preprocess.py",
    "checkpoints/randla_3m/model.onnx": "model.onnx",
}


def ssh_exec(ssh, cmd, print_output=True):
    """Execute command and print output."""
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if print_output:
        if out.strip():
            print(out.strip())
        if err.strip():
            print(f"STDERR: {err.strip()}")
    return out, err


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {USER}@{HOST}...")
    ssh.connect(HOST, username=USER, password=PASS, timeout=10)
    print("Connected!")

    # Create remote directory
    ssh_exec(ssh, f"mkdir -p {REMOTE_DIR}")

    # Upload files via SFTP
    sftp = ssh.open_sftp()
    for local_rel, remote_name in FILES.items():
        local_path = os.path.join(LOCAL_BASE, local_rel)
        remote_path = f"{REMOTE_DIR}/{remote_name}"
        print(f"Uploading {local_rel} -> {remote_path}")
        sftp.put(local_path, remote_path)

    # Create a test benchmark script on the Pi
    test_script = '''#!/usr/bin/env python3
"""Benchmark wall segmentation inference on Raspberry Pi."""
import numpy as np
import time
import subprocess
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check temp
def get_temp():
    try:
        t = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
        return float(t) / 1000.0
    except:
        return -1

# Check if onnxruntime is installed
try:
    import onnxruntime as ort
except ImportError:
    print("Installing onnxruntime...")
    subprocess.run(["pip3", "install", "onnxruntime"], check=True)
    import onnxruntime as ort

print(f"ONNX Runtime version: {ort.__version__}")
print(f"CPU temp before: {get_temp():.1f}°C")

# Load model
print("\\nLoading model...")
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
print(f"Model loaded. Input: {sess.get_inputs()[0].shape}")

# Generate test data (simulate a 3m block with 16384 points)
print("\\n--- Single block benchmark ---")
np.random.seed(42)
x = np.random.randn(1, 10, 16384).astype(np.float32)

# Warmup
_ = sess.run(None, {"input": x})

# Benchmark single block
times = []
for i in range(5):
    t0 = time.time()
    out = sess.run(None, {"input": x})
    t = time.time() - t0
    times.append(t)
    print(f"  Run {i+1}: {t*1000:.0f}ms")

avg = np.mean(times)
print(f"  Average: {avg*1000:.0f}ms per block")
print(f"  CPU temp after single: {get_temp():.1f}°C")

# Simulate full room inference (46 blocks like room_merged_leveled)
print("\\n--- Full room simulation (46 blocks) ---")
t0 = time.time()
for i in range(46):
    _ = sess.run(None, {"input": x})
    if (i+1) % 10 == 0:
        print(f"  Block {i+1}/46, temp={get_temp():.1f}°C")
total = time.time() - t0
print(f"  Total: {total:.1f}s for 46 blocks")
print(f"  Average: {total/46*1000:.0f}ms per block")
print(f"  CPU temp after full room: {get_temp():.1f}°C")

# Simulate living room (84 blocks)
print("\\n--- Living room simulation (84 blocks) ---")
t0 = time.time()
for i in range(84):
    _ = sess.run(None, {"input": x})
    if (i+1) % 20 == 0:
        print(f"  Block {i+1}/84, temp={get_temp():.1f}°C")
total = time.time() - t0
print(f"  Total: {total:.1f}s for 84 blocks")
print(f"  Average: {total/84*1000:.0f}ms per block")
print(f"  CPU temp after living room: {get_temp():.1f}°C")

print("\\n--- Summary ---")
print(f"  Model size: {os.path.getsize('model.onnx')/1e6:.1f} MB")
print(f"  Per block: {avg*1000:.0f}ms")
print(f"  Room (46 blocks): ~{avg*46:.0f}s")
print(f"  Living room (84 blocks): ~{avg*84:.0f}s")
'''

    # Write test script
    with sftp.open(f"{REMOTE_DIR}/benchmark.py", "w") as f:
        f.write(test_script)

    sftp.close()
    print("\nAll files uploaded!")

    # Check Python and deps
    print("\n--- Checking Pi environment ---")
    ssh_exec(ssh, "python3 --version")
    ssh_exec(ssh, "uname -a")
    ssh_exec(ssh, "cat /proc/cpuinfo | grep 'Model' | head -1")
    ssh_exec(ssh, "free -h | head -2")

    # Run benchmark
    print("\n--- Running benchmark ---")
    stdin, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && python3 benchmark.py 2>&1",
        timeout=600,
    )
    # Stream output
    for line in iter(stdout.readline, ""):
        print(line.rstrip())

    err = stderr.read().decode()
    if err.strip():
        print(f"STDERR: {err.strip()}")

    ssh.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
