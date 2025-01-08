from flask import Flask, request

def create_flask_app(client):
    """Factory function to create the Flask app."""
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            new_quality = request.form.get("fixed_quality")
            if new_quality.isdigit():
                client.segment_downloader.current_quality = int(new_quality)
        return f"""
        <html>
            <body>
                <h1>Streaming Client Configuration</h1>
                <form method="POST">
                    <label for="fixed_quality">Fixed Quality (0, 1, 2, 3, 4):</label>
                    <input type="number" id="fixed_quality" name="fixed_quality" value="{client.segment_downloader.current_quality or ''}" min="0" max="4">
                    <button type="submit">Update</button>
                </form>
            </body>
        </html>
        """

    return app
