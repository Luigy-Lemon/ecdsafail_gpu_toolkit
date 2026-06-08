#!/usr/bin/env python3
# Pure Python Curses TUI Dashboard for ecdsafail island search.
# Safe, reliable, and portable across Linux/GPU environments.

import os
import sys
import re
import time
import signal
import subprocess
import threading
import queue
import locale
import curses

# Set locale to enable UTF-8 character support in curses
locale.setlocale(locale.LC_ALL, '')

# Global state
MY_PID = os.getpid()
LOCKFILE = "/tmp/ecdsafail_dashboard.lock"
has_lock = False

# Read CLI arguments
if len(sys.argv) < 5:
    print("usage: dashboard.py STATE START N CHUNK [GPUS]", file=sys.stderr)
    sys.exit(1)

STATE = sys.argv[1]
START = int(sys.argv[2])
N = int(sys.argv[3])
CHUNK = int(sys.argv[4])
GPUS = sys.argv[5] if len(sys.argv) > 5 else "auto"

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(os.path.dirname(HERE), "island.sh")

# Acquire Concurrency Lock
try:
    if os.path.exists(LOCKFILE):
        with open(LOCKFILE, "r") as lf:
            old_pid = lf.read().strip()
        if old_pid:
            # Check if old process is still alive
            try:
                os.kill(int(old_pid), 0)
                print(f"ERROR: Another instance of dashboard.py is already running (PID: {old_pid}).", file=sys.stderr)
                sys.exit(1)
            except OSError:
                pass # Stale lock
    with open(LOCKFILE, "w") as lf:
        lf.write(str(MY_PID))
    has_lock = True
except Exception as e:
    print(f"ERROR: Lockfile error: {e}", file=sys.stderr)
    sys.exit(1)

# Extract Target Score cleanly from the state file
target_score = None
try:
    if os.path.exists(STATE):
        with open(STATE, "rb") as f:
            content = f.read()
            idx = content.find(b"SCORE_TARGET:")
            if idx != -1:
                end_idx = content.find(b"\n", idx)
                if end_idx != -1:
                    target_score = content[idx + 13:end_idx].decode("utf-8", errors="ignore").strip()
                else:
                    target_score = content[idx + 13:].decode("utf-8", errors="ignore").strip()
except Exception:
    pass

# Determine GPUS count
num_gpus = 1
try:
    if GPUS == "auto" or not GPUS:
        res = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if res.returncode == 0:
            num_gpus = len([line for line in res.stdout.strip().split("\n") if line.strip()])
    else:
        if GPUS.isdigit():
            num_gpus = int(GPUS)
        else:
            num_gpus = len([g for g in GPUS.split(",") if g.strip()])
except Exception:
    num_gpus = 1
if num_gpus < 1:
    num_gpus = 1

# Launch the search subprocess in a new process group
proc = subprocess.Popen(
    ["bash", BIN, "search", STATE, str(START), str(N), str(CHUNK), GPUS],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    preexec_fn=os.setsid
)

# Shared thread-safe output queue
output_queue = queue.Queue()

# Thread to read search stdout/stderr
def read_output_stream(stream, q):
    try:
        for line in iter(stream.readline, ""):
            q.put(line)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass

stream_thread = threading.Thread(target=read_output_stream, args=(proc.stdout, output_queue))
stream_thread.daemon = True
stream_thread.start()

# Shared GPU utilization stats
gpu_info = {}
gpu_info_lock = threading.Lock()

def update_gpu_stats():
    while proc.poll() is None:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu", "--format=csv,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
            )
            if res.returncode == 0:
                new_info = {}
                for line in res.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        g_id, name, util = parts
                        name = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
                        util = util.replace("%", "").strip()
                        new_info[g_id] = {"name": name[:16], "util": util}
                with gpu_info_lock:
                    gpu_info.update(new_info)
        except Exception:
            pass
        time.sleep(1.5)

gpu_thread = threading.Thread(target=update_gpu_stats)
gpu_thread.daemon = True
gpu_thread.start()

# Dynamic search state variables
per = (N + num_gpus - 1) // num_gpus
gpu_done = {str(i): 0 for i in range(num_gpus)}
gpu_total = {str(i): per for i in range(num_gpus)}
gpu_speed = {str(i): 0 for i in range(num_gpus)}
gpu_cands = {str(i): 0 for i in range(num_gpus)}

found_nonces = []
recent_logs = []
search_config = None
start_time = time.time()
started = False
completed = False
final_elapsed = 0

# Regex definitions
progress_re = re.compile(r"GPU\[(\d+)\] OFFSET=(\d+): PROGRESS done=(\d+) total=(\d+) speed=(\d+) candidates=(\d+)")
clean_re = re.compile(r"CLEAN nonce=(\d+)")
config_re = re.compile(r"loaded:\s*(.*)")

def format_time(seconds):
    if seconds < 0:
        return "--:--"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def format_num(num):
    return f"{num:,}"

def cleanup():
    # Terminate process group
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=2)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    # Clean up lockfile
    if has_lock:
        try:
            os.remove(LOCKFILE)
        except Exception:
            pass

    # Print final summary to standard terminal
    print("\nSearch completed.")
    if found_nonces:
        print("Found Candidates:")
        for nonce in found_nonces:
            print(f"  CLEAN nonce={nonce}")
    sys.exit(0)

def main_ui(stdscr):
    global started, search_config, completed, final_elapsed
    
    # Configure curses coloring
    curses.use_default_colors()
    curses.start_color()
    
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # Label / Header
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # Status / Warning
    curses.init_pair(3, curses.COLOR_GREEN, -1)    # Progress / Clean
    curses.init_pair(4, curses.COLOR_MAGENTA, -1)  # Target / Candidate
    curses.init_pair(5, curses.COLOR_RED, -1)      # Errors
    
    stdscr.nodelay(True)
    curses.curs_set(0) # Hide cursor
    
    last_update_time = 0

    while True:
        # Check window size
        try:
            height, width = stdscr.getmaxyx()
        except Exception:
            height, width = 24, 80

        if height < 20 or width < 60:
            stdscr.clear()
            try:
                stdscr.addstr(0, 0, "Window too small. Please resize.")
            except Exception:
                pass
            stdscr.refresh()
            time.sleep(0.1)
            try:
                ch = stdscr.getch()
                if ch in (ord('q'), ord('Q'), 27): # q, Q or ESC
                    break
            except Exception:
                pass
            continue

        # Parse all incoming lines from the search thread
        new_data = False
        while not output_queue.empty():
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            
            line_strip = line.strip()
            if not line_strip:
                continue

            prog_m = progress_re.match(line_strip)
            clean_m = clean_re.match(line_strip)
            config_m = config_re.search(line_strip)

            if prog_m:
                started = True
                new_data = True
                gpu_id = prog_m.group(1)
                offset = int(prog_m.group(2))
                done = int(prog_m.group(3))
                gpu_done[gpu_id] = offset + done
                gpu_total[gpu_id] = int(prog_m.group(4))
                gpu_speed[gpu_id] = int(prog_m.group(5))
                gpu_cands[gpu_id] = int(prog_m.group(6))
            elif clean_m:
                new_data = True
                nonce = clean_m.group(1)
                if nonce not in found_nonces:
                    found_nonces.append(nonce)
            elif config_m:
                new_data = True
                search_config = config_m.group(1)
            else:
                # Skip progress log keyword matches to avoid clutter
                if "PROGRESS" not in line_strip:
                    new_data = True
                    ts = time.strftime("%H:%M:%S")
                    recent_logs.append(f"[{ts}] {line_strip}")
                    if len(recent_logs) > 20:
                        recent_logs.pop(0)

        # Redraw throttled to 5fps or when there is new data
        current_time = time.time()
        if new_data or (current_time - last_update_time >= 0.2):
            last_update_time = current_time
            
            stdscr.clear()
            
            # Title
            title = "⚡ ECDSAFAIL ISLAND SEARCH TUI ⚡"
            title_x = max(0, (width - len(title)) // 2)
            stdscr.addstr(0, title_x, title, curses.color_pair(1) | curses.A_BOLD)
            
            # Metadata block
            stdscr.addstr(2, 2, "Search State: ")
            stdscr.addstr(2, 16, STATE, curses.A_BOLD)
            
            y = 3
            if target_score:
                stdscr.addstr(y, 2, "Target Score: ")
                stdscr.addstr(y, 16, target_score, curses.color_pair(4) | curses.A_BOLD)
                y += 1
                
            stdscr.addstr(y, 2, "Range Start : ")
            stdscr.addstr(y, 16, format_num(START), curses.A_BOLD)
            y += 1
            
            stdscr.addstr(y, 2, "Total Count : ")
            stdscr.addstr(y, 16, format_num(N), curses.A_BOLD)
            y += 1
            
            if search_config:
                stdscr.addstr(y, 2, "Config      : ")
                cfg_w = width - 18
                stdscr.addstr(y, 16, search_config[:cfg_w], curses.color_pair(1))
                y += 1
                
            y += 1 # Empty spacing
            
            # Check if subprocess finished
            if not completed and proc.poll() is not None and not stream_thread.is_alive() and output_queue.empty():
                completed = True
                final_elapsed = int(current_time - start_time)

            # Aggregate stats
            if completed:
                total_done = N
                elapsed = final_elapsed
                total_speed = sum(gpu_speed.values())
            else:
                total_done = sum(gpu_done.values())
                if total_done > N:
                    total_done = N
                total_speed = sum(gpu_speed.values())
                elapsed = int(current_time - start_time)
            
            # Progress Section
            if completed:
                stdscr.addstr(y, 2, "Status: ", curses.color_pair(3) | curses.A_BOLD)
                stdscr.addstr(y, 10, f"COMPLETED (elapsed: {format_time(final_elapsed)}) [Press 'q' to exit]", curses.color_pair(3) | curses.A_BOLD)
                y += 2
                
                bar_w = width - 22
                if bar_w < 10:
                    bar_w = 10
                stdscr.addstr(y, 2, f"Elapsed Time: {format_time(final_elapsed)} | ETA: 00:00 | Speed: ")
                stdscr.addstr(f"{format_num(total_speed)} n/s", curses.color_pair(3) | curses.A_BOLD)
                y += 2
                
                bar = "=" * bar_w
                stdscr.addstr(y, 2, "Progress: [")
                stdscr.addstr(bar, curses.color_pair(3))
                stdscr.addstr(f"] 100%")
                y += 2
            elif not started:
                stdscr.addstr(y, 2, "Status: ", curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(y, 10, f"STARTING UP (elapsed: {format_time(elapsed)})", curses.color_pair(2) | curses.A_BOLD)
                y += 2
                stdscr.addstr(y, 2, "Progress: ", curses.A_DIM)
                stdscr.addstr(y, 12, "[ Waiting for GPU initialization... ]", curses.A_DIM)
                y += 2
            else:
                eta_str = "--:--"
                if total_speed > 0:
                    eta_str = format_time((N - total_done) // total_speed)
                overall_pct = (total_done * 100 // N) if N > 0 else 0
                if overall_pct > 100:
                    overall_pct = 100
                    
                stdscr.addstr(y, 2, f"Elapsed Time: {format_time(elapsed)} | ETA: {eta_str} | Speed: ")
                stdscr.addstr(f"{format_num(total_speed)} n/s", curses.color_pair(3) | curses.A_BOLD)
                y += 2
                
                bar_w = width - 22
                if bar_w < 10:
                    bar_w = 10
                filled = overall_pct * bar_w // 100
                bar = "=" * filled + " " * (bar_w - filled)
                stdscr.addstr(y, 2, "Progress: [")
                stdscr.addstr(bar[:filled], curses.color_pair(3))
                stdscr.addstr(bar[filled:])
                stdscr.addstr(f"] {overall_pct:3d}%")
                y += 2
                
            # GPU Utilization Table
            stdscr.addstr(y, 2, f"  {'GPU':<4} | {'Model':<16} | {'Util':<4} | {'Progress Bar':<20} | {'%':<5} | {'Speed (n/s)':<11} | {'Candidates':<10}"[:width-4], curses.color_pair(1) | curses.A_UNDERLINE | curses.A_BOLD)
            y += 1
            
            with gpu_info_lock:
                for g_id in sorted(gpu_done.keys(), key=int):
                    g_done = gpu_done[g_id] if not completed else gpu_total[g_id]
                    g_tot = gpu_total[g_id]
                    g_sp = gpu_speed[g_id]
                    g_cand = gpu_cands[g_id]
                    
                    g_pct = (g_done * 100 // g_tot) if g_tot > 0 else 0
                    if g_pct > 100 or completed:
                        g_pct = 100
                        
                    g_filled = g_pct * 20 // 100
                    g_bar = "=" * g_filled + " " * (20 - g_filled)
                    
                    g_model = gpu_info.get(g_id, {}).get("name", "Unknown")
                    g_util = gpu_info.get(g_id, {}).get("util", "0") if not completed else "0"
                    
                    if not started:
                        g_pct = 0
                        g_bar = " " * 20
                        g_sp_str = "-"
                        g_cand_str = "-"
                    else:
                        g_sp_str = format_num(g_sp)
                        g_cand_str = str(g_cand)
                        
                    stdscr.addstr(y, 2, f"   #{g_id:<2} | {g_model:<16} | {g_util:>3}% | [")
                    stdscr.addstr(g_bar[:g_filled], curses.color_pair(3))
                    stdscr.addstr(g_bar[g_filled:])
                    stdscr.addstr(f"] | {g_pct:>3}% | {g_sp_str:<11} | {g_cand_str:<10}"[:width - (28 + len(g_bar))])
                    y += 1
                    
            y += 1 # spacing
            
            # Candidates Box
            stdscr.addstr(y, 2, f"🎉 FOUND CANDIDATES ({len(found_nonces)}) 🎉", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            if not found_nonces:
                stdscr.addstr(y, 4, "(No candidates found yet)", curses.A_DIM)
                y += 1
            else:
                for nonce in found_nonces:
                    stdscr.addstr(y, 4, "- Nonce: ")
                    stdscr.addstr(nonce, curses.color_pair(3) | curses.A_BOLD)
                    stdscr.addstr(" (CLEAN)")
                    y += 1
                    
            y += 1 # spacing
            
            # Recent Logs Box
            log_w = width - 6
            if log_w < 10:
                log_w = 10
            stdscr.addstr(y, 2, f"╭─ Recent Logs {'─' * (log_w - 15)}╮"[:width-4])
            y += 1
            
            max_logs = max(3, height - y - 2)
            visible_logs = recent_logs[-max_logs:] if recent_logs else []
            
            log_count = 0
            if not visible_logs:
                stdscr.addstr(y, 2, "│ ")
                stdscr.addstr("(No logs yet)", curses.A_DIM)
                stdscr.addstr(" " * (log_w - 15) + "│")
                y += 1
                log_count = 1
            else:
                for log in visible_logs:
                    disp = log[:log_w - 4]
                    pad = log_w - 4 - len(disp)
                    stdscr.addstr(y, 2, "│ ")
                    
                    if any(w in disp.lower() for w in ["error", "fail", "mismatch"]):
                        stdscr.addstr(disp, curses.color_pair(5))
                    elif any(w in disp.lower() for w in ["warning", "warn"]):
                        stdscr.addstr(disp, curses.color_pair(2))
                    elif any(w in disp.lower() for w in ["clean", "success", "ok"]):
                        stdscr.addstr(disp, curses.color_pair(3))
                    else:
                        stdscr.addstr(disp)
                    stdscr.addstr(" " * pad + " │")
                    y += 1
                    log_count += 1
                    
            while log_count < max_logs:
                stdscr.addstr(y, 2, f"│{' ' * (log_w - 2)}│")
                y += 1
                log_count += 1
                
            stdscr.addstr(y, 2, f"╰{'─' * (log_w - 2)}╯"[:width-4])
            stdscr.refresh()

        # Handle keyboard input
        try:
            ch = stdscr.getch()
            if ch in (ord('q'), ord('Q'), 27): # q, Q or ESC
                break
        except Exception:
            pass

        # Keep running to display TUI even after subprocess finishes

        time.sleep(0.05)

if __name__ == "__main__":
    try:
        curses.wrapper(main_ui)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
