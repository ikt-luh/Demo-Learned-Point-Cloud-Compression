import threading
import os
import argparse
import shutil
import atexit

from src.server.capturer import Capturer
from server.server import Server
from server.monitoring import Monitor


def parse_arguments():
    parser = argparse.ArgumentParser(description="DASH streaming server setup")
    parser.add_argument('--output-dir', type=str, default="./segments", help="Directory to store output segments (default: ./segments)")
    parser.add_argument('--segment-duration', type=int, default=0.5, help="Duration of each segment in seconds (default: 2)")
    parser.add_argument('--port', type=int, default=8000, help="Port for the HTTP server (default: 8000)")
    parser.add_argument('--qualities', type=int, default=4, help="Number of qualities to emulates")
    parser.add_argument('--init-file', type=str, default="init.m4s", help="Initialization file name (default: init.m4s)")

    return parser.parse_args()


def run_capturing(output_dir, init_file, mpd_file, segment_duration, qualities):
    capturer = Capturer(output_dir, segment_duration)
    threading.Thread(target=capturer.capture, daemon=None).start()


def run_mpd_updater(output_dir, init_file, mpd_file, segment_duration):
    manifest_monitor = Monitor(output_dir, init_file, mpd_file, segment_duration)
    threading.Thread(target=manifest_monitor.monitor_segments, daemon=None).start()


def run_http_server(output_dir, port):
    Server.start_http_server(output_dir, port=port)


def cleanup_segments(output_dir):
    """Cleanup the output directory by deleting it and its contents."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned up {output_dir}")


if __name__ == "__main__":
    # Parse command-line arguments and store them in a variable
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Update MPD file location based on output directory
    mpd_file = os.path.join(args.output_dir, "live.mpd")

    # Register cleanup
    atexit.register(cleanup_segments, args.output_dir)

    # Run tasks
    run_capturing(args.output_dir, args.init_file, mpd_file, args.segment_duration, args.qualities)
    #run_mpd_updater(args.output_dir, args.init_file, mpd_file, args.segment_duration)
    run_http_server(args.output_dir, args.port)
