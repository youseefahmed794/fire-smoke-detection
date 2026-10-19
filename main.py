"""
MAIN.PY
-------
This is the central entry point for the Fire Detection System.
It allows the user to choose between:
1. Video File Mode (Testing on pre-recorded footage) - Uses video_module.py
2. Live Camera Mode (RTSP Streaming) - Uses cam_module.py (Threaded)
"""
import sys
import logging
import os

# Add src to the system path to allow imports from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import setup_logging
import src.services.video_service as video_module
import src.services.camera_service as cam_module

import argparse

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Fire and Smoke Detection System")
    parser.add_argument('--mode', choices=['video', 'camera'], help="Operation mode")
    parser.add_argument('--source', help="Path to video file (optional, for video mode)")
    parser.add_argument('--logs', action='store_true', help="View recent detection logs")

    args = parser.parse_args()

    if args.logs:
        from src.services.database_service import DatabaseService
        from tabulate import tabulate

        db = DatabaseService()
        logs = db.get_recent_logs(limit=10)

        if logs:
            headers = ["ID", "Time", "Type", "Channels", "Image Path"]
            print("\n" + "="*50)
            print("📜 Recent Detection Logs")
            print("="*50)
            print(tabulate(logs, headers=headers, tablefmt="grid"))
            print("\n")
        else:
            print("\n📭 No logs found in the database.\n")

        sys.exit(0)

    if args.mode:
        # CLI Mode
        if args.mode == 'video':
            logger.info("CLI selection: Video File Mode")
            video_path = args.source if args.source else None
            print("\n🚀 Starting Video File Mode...")
            video_module.main(video_source=video_path)

        elif args.mode == 'camera':
            logger.info("CLI selection: Live Camera Mode")
            print("\n🚀 Starting Live Camera Mode...")
            cam_module.main()

    else:
        # Interactive Mode
        print("\n" + "="*50)
        print("🔥 FIRE AND SMOKE DETECTION SYSTEM 🔥")
        print("="*50)
        print("\nPlease select a mode:")
        print("1. 📁 Video File Testing")
        print("2. 📹 Live Camera / RTSP (Uses configuration from .env)")
        print("q. ❌ Quit")

        choice = input("\nEnter choice (1/2/q): ").strip().lower()

        if choice == '1':
            logger.info("User selected: Video File Mode")

            # Ask user for video path
            print("\n[OPTIONAL] Enter path to video file (Press Enter to use default from Config):")
            video_path_input = input("Path: ").strip().strip('"').strip("'")

            video_path = video_path_input if video_path_input else None

            print("\n🚀 Starting Video File Mode...")
            # Call the main function of the video module
            video_module.main(video_source=video_path)

        elif choice == '2':
            logger.info("User selected: Live Camera Mode")
            print("\n🚀 Starting Live Camera Mode...")
            # Call the main function of the cam module
            cam_module.main()

        elif choice == 'q':
            print("\nGoodbye! 👋")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please try again.")
            main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 User interrupted. Exiting...")
        sys.exit(0)
