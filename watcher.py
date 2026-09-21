import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
REPO_OWNER = "raarzketkar"
REPO_NAME = "STRAIT_by_Penguin"
BRANCH = "main"  # Set to "master" if your default branch is master
POLL_INTERVAL_SECONDS = 10

# GitHub raw content base endpoints
RAW_BASE = (
    f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"
)
LATEST_URL = f"{RAW_BASE}/latest.txt"

# Track the last file that was processed to avoid running duplicates
last_processed_file = None


def fetch_text_content(url: str) -> str:
    """Fetch text from a URL with a timestamp query to bypass any intermediate caching."""
    cache_busting_url = f"{url}?t={int(time.time())}"
    req = urllib.request.Request(
        cache_busting_url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read().decode("utf-8").strip()


def download_file(url: str, output_destination: str) -> None:
    """Download a remote binary/text file directly to disk."""
    cache_busting_url = f"{url}?t={int(time.time())}"
    req = urllib.request.Request(
        cache_busting_url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        with open(output_destination, "wb") as f:
            f.write(response.read())


def main():
    global last_processed_file

    print("=" * 65)
    print("STRAIT by Penguin - Continuous Watcher")
    print(f"Monitoring: https://github.com/{REPO_OWNER}/{REPO_NAME} [{BRANCH}]")
    print("Waiting for new CSV submissions from Device A...")
    print("=" * 65)

    while True:
        try:
            # 1. Read latest.txt from GitHub to see what Device A submitted
            target_csv_name = fetch_text_content(LATEST_URL)

            # 2. Process only if a new file pointer is discovered
            if target_csv_name and target_csv_name != last_processed_file:
                print(f"\n[+] New payload detected: '{target_csv_name}'")
                print(f"[*] Downloading '{target_csv_name}' from GitHub...")

                csv_url = f"{RAW_BASE}/{target_csv_name}"
                download_file(csv_url, target_csv_name)

                print(f"[*] File saved locally as '{target_csv_name}'.")
                print(
                    f"[*] Launching antarctic_nav.py with argument: {target_csv_name}\n"
                )

                # 3. Execute antarctic_nav.py using the active Python interpreter
                subprocess.run(
                    [sys.executable, "antarctic_nav.py", target_csv_name],
                    check=True,
                )

                last_processed_file = target_csv_name
                print(
                    f"\n[*] Run finished successfully for '{target_csv_name}'."
                )
                print("[*] Resuming monitor loop...")

        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                # latest.txt does not exist on GitHub yet
                pass
            else:
                print(f"[!] HTTP error ({http_err.code}): {http_err.reason}")
        except subprocess.CalledProcessError as proc_err:
            print(
                f"[!] Execution failed inside antarctic_nav.py (Exit Code: {proc_err.returncode})"
            )
        except KeyboardInterrupt:
            print("\nWatcher stopped by user. Exiting.")
            sys.exit(0)
        except Exception as err:
            print(f"[!] Warning: {err}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
