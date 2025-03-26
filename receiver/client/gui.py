from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import random  # Mock data for latencies

# Factory function to create the Flask app
bandwidths = [0] * 20

def create_flask_app(client):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "secret!"
    socketio = SocketIO(app)
    app.thread_started = False

    # Route for the configuration page
    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            new_quality = request.form.get("fixed_quality")
            if new_quality and new_quality.isdigit():
                client.segment_downloader.current_quality = int(new_quality)
        return render_template("index.html", current_quality=client.segment_downloader.current_quality or "")

    # Background thread to emit bandwidth and latency updates
    def background_data_updater():
        while True:
            time.sleep(1)  # Wait for 1 second

            bandwidth = client.get_bandwidth()  # Assume this method fetches the current bandwidth

            latencies = client.get_latencies()
            socketio.emit("update_data", {"bandwidth": bandwidth, "latencies": latencies})

    # Start the background thread
    @app.before_request
    def start_background_thread():
        if not app.thread_started:
            app.thread_started = True
            thread = threading.Thread(target=background_data_updater)
            thread.daemon = True
            thread.start()

    return app, socketio
