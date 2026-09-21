import os
import subprocess
import sys
import time

POLL_INTERVAL_SECONDS = 5
BRANCH = "main"


def run_git(cmd):
    """Run a git command in the current directory and return trimmed stdout."""
    res = subprocess.run(
        ["git"] + cmd, capture_output=True, text=True, check=False
    )
    return res.stdout.strip()


def main():
    print("=" * 65)
    print("STRAIT by Penguin - Git-Native Real-Time Watcher")
    print(f"Monitoring branch: origin/{BRANCH} (Forced Sync Mode)")
    print("=" * 65)

    # Initial sync
    run_git(["fetch", "origin", BRANCH])
    last_commit = run_git(["rev-parse", f"origin/{BRANCH}"])
    run_git(["reset", "--hard", f"origin/{BRANCH}"])
    print(f"[*] Initial commit synced: {last_commit[:7]}")
    print("[*] Waiting for Device A pushes...")

    while True:
        try:
            # 1. Fetch latest metadata from GitHub
            run_git(["fetch", "origin", BRANCH])
            remote_commit = run_git(["rev-parse", f"origin/{BRANCH}"])

            # 2. Check for new commit
            if remote_commit and remote_commit != last_commit:
                print(
                    f"\n\n[+] New commit detected! {last_commit[:7]} -> {remote_commit[:7]}"
                )
                print("[*] Syncing workspace with remote...")

                # Force working directory to exact remote state
                sync_out = run_git(["reset", "--hard", f"origin/{BRANCH}"])
                print(f"[*] Git: {sync_out}")

                # 3. Read latest.txt
                if not os.path.exists("latest.txt"):
                    print("[!] latest.txt still not found on disk.")
                    last_commit = remote_commit
                    continue

                with open("latest.txt", "r") as f:
                    active_csv = f.read().strip()

                print(f"[*] Payload pointer: '{active_csv}'")

                if not os.path.exists(active_csv):
                    print(f"[!] Error: {active_csv} was not found on disk.")
                    last_commit = remote_commit
                    continue

                # 4. Trigger navigation
                print(
                    f"[*] Launching antarctic_nav.py with {active_csv}...\n"
                    + "=" * 50
                )
                subprocess.run(
                    [sys.executable, "antarctic_nav.py", active_csv], check=True
                )
                print("=" * 50)
                print(f"[*] Route generation complete for {active_csv}!")

                last_commit = remote_commit
                print("[*] Resuming monitor loop...")
            else:
                print(".", end="", flush=True)

        except subprocess.CalledProcessError as err:
            print(f"\n[!] antarctic_nav.py exited with error code {err.returncode}")
            last_commit = remote_commit
        except KeyboardInterrupt:
            print("\n[!] Watcher stopped by user.")
            break
        except Exception as e:
            print(f"\n[!] Watcher loop error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()