"""Rent a vast.ai GPU, upload code, and print SSH command.

Usage:
    python scripts/vastai_launch.py
    python scripts/vastai_launch.py --gpu RTX_4090
    python scripts/vastai_launch.py --offer-id 31126094  # specific machine
"""

import subprocess
import json
import argparse
import time
import sys


def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="RTX_3090", help="GPU model (RTX_3090, RTX_4090)")
    parser.add_argument("--offer-id", type=int, default=None, help="Specific offer ID")
    parser.add_argument("--disk", type=int, default=300, help="Disk space in GB")
    args = parser.parse_args()

    if args.offer_id:
        offer_id = args.offer_id
        print(f"Using specific offer: {offer_id}")
    else:
        # Find cheapest offer
        print(f"Searching for cheapest {args.gpu}...")
        query = f"gpu_name={args.gpu} num_gpus=1 reliability>0.95 inet_down>500 disk_space>{args.disk}"
        result = run(f'vastai search offers "{query}" -o dph --limit 1 --raw')
        offers = json.loads(result)
        if not offers:
            print("No offers found!")
            sys.exit(1)
        offer = offers[0]
        offer_id = offer["id"]
        price = offer["dph_total"]
        net_down = offer.get("inet_down", 0)
        location = offer.get("geolocation", "Unknown")
        print(f"Best offer: #{offer_id} at ${price:.4f}/hr, {net_down:.0f} Mbps down, {location}")

    # Create instance
    print("Creating instance...")
    result = run(
        f'vastai create instance {offer_id} '
        f'--image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime '
        f'--disk {args.disk} '
        f'--raw'
    )
    instance = json.loads(result)
    instance_id = instance.get("new_contract")
    if not instance_id:
        print(f"Failed to create instance: {result}")
        sys.exit(1)
    print(f"Instance created: #{instance_id}")

    # Wait for it to be ready
    print("Waiting for instance to start...", end="", flush=True)
    for _ in range(60):
        time.sleep(5)
        print(".", end="", flush=True)
        result = run(f"vastai show instance {instance_id} --raw")
        info = json.loads(result)
        status = info.get("actual_status", "")
        if status == "running":
            break
    print()

    if status != "running":
        print(f"Instance not ready after 5 min. Status: {status}")
        sys.exit(1)

    # Get SSH info
    ssh_host = info.get("ssh_host", "")
    ssh_port = info.get("ssh_port", "")
    print(f"\n{'='*60}")
    print(f"Instance #{instance_id} is RUNNING")
    print(f"{'='*60}")
    print(f"\n1. Copy your code to the instance:")
    print(f"   scp -P {ssh_port} -r . root@{ssh_host}:/workspace/wall-seg/")
    print(f"\n2. SSH into the instance:")
    print(f"   ssh -p {ssh_port} root@{ssh_host}")
    print(f"\n3. Run setup:")
    print(f"   cd /workspace/wall-seg && bash scripts/vastai_setup.sh")
    print(f"\n4. When done, destroy the instance:")
    print(f"   vastai destroy instance {instance_id}")
    print(f"\nPrice: ${info.get('dph_total', 0):.4f}/hr")


if __name__ == "__main__":
    main()
