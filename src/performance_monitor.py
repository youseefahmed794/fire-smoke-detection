"""
PERFORMANCE_MONITOR.PY
----------------------
This utility class helps track the computer's health while the AI is running.
It monitors:
1. CPU Usage (percentage)
2. RAM Usage (in Megabytes)
3. GPU Usage and Temperature (if a dedicated GPU is present)
"""
import psutil   # Tool for reading system (CPU/RAM) stats
import GPUtil   # Tool for reading Graphics Card (GPU) stats
import os
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        """
        Initialize the monitor by getting the current process ID (PID).
        We only want to track resources used by THIS specific program, not the whole computer.
        """
        self.process = psutil.Process(os.getpid())

    def log_stats(self):
        """
        Gather and print system statistics.
        Call this periodically (e.g., every 100 frames) to check for overheating or memory leaks.
        """
        # 1. Get CPU and RAM
        # cpu_percent triggers a small sleep to measure utilization, so it might take 0.1s
        cpu = self.process.cpu_percent()

        # rss is 'Resident Set Size' (Physical Memory used)
        ram = self.process.memory_info().rss / 1024**2 # Convert Bytes -> MB

        stats_msg = f"💻 System: CPU {cpu:.1f}% | RAM {ram:.1f}MB"

        # 2. Get GPU stats if available
        # GPUtil finds all NVIDIA GPUs on the system
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                # load is 0.0 to 1.0, so multiply by 100 for percentage
                stats_msg += f" | 🎮 GPU {gpu.id}: {gpu.load*100:.1f}% ({gpu.temperature}°C)"

        # Print to console for immediate feedback
        print(stats_msg)
        # logger.info(stats_msg) # Uncomment to save stats to your log file
