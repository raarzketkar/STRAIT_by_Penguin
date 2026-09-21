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
    print(f"Monitoring branch: origin/{BRANCH} (Zero-Cache Mode)")
    print("=" * 65)

    # Initial sync to ensure local is up to date
    run_git(["fetch", "origin", BRANCH])
    last_commit = run_git(["rev-parse", "HEAD"])
    print(f"[*] Initial HEAD commit: {last_commit[:7]}")
    print("[*] Waiting for Device A pushes...")

    while True:
        try:
            # 1. Fetch latest metadata directly from GitHub (bypasses all CDN caches)
            run_git(["fetch", "origin", BRANCH])
            remote_commit = run_git(["rev-parse", f"origin/{BRANCH}"])

            # 2. Check if a new commit landed
            if remote_commit and remote_commit != last_commit:
                print(
                    f"\n\n[+] New commit detected! {last_commit[:7]} -> {remote_commit[:7]}"
                )
                print("[*] Pulling changes from origin...")

                pull_res = run_git(["pull", "--rebase", "origin", BRANCH])
                print(pull_res)

                # 3. Read the freshly pulled local latest.txt
                if not os.path.exists("latest.txt"):
                    print("[!] latest.txt not found after pull.")
                    last_commit = remote_commit
                    continue

                with open("latest.txt", "r") as f:
                    active_csv = f.read().strip()

                print(f"[*] Payload pointer: '{active_csv}'")

                if not os.path.exists(active_csv):
                    print(
                        f"[!] Error: {active_csv} was not downloaded by git pull."
                    )
                    last_commit = remote_commit
                    continue

                # 4. Trigger antarctic_nav.py with the real file
                print(
                    f"[*] Launching antarctic_nav.py with {active_csv}...\n"
                    + "=" * 50
                )
                subprocess.run(
                    [sys.executable, "antarctic_nav.py", active_csv], check=True
                )
                print("=" * 50)
                print(f"[*] Route generation complete for {active_csv}!")

                # Update pointer so we don't repeat this commit
                last_commit = remote_commit
                print("[*] Resuming monitor loop...")
            else:
                print(".", end="", flush=True)

        except subprocess.CalledProcessError as err:
            print(f"\n[!] antarctic_nav.py exited with error code {err.returncode}")
            last_commit = remote_commit  # Prevent infinite crash loops
        except KeyboardInterrupt:
            print("\n[!] Watcher stopped by user.")
            break
        except Exception as e:
            print(f"\n[!] Watcher loop error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
