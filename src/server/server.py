from http.server import SimpleHTTPRequestHandler, HTTPServer

class Server(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def start_http_server(directory, port=8000):
        server_address = ('', port)
        # Create the server with the correct directory
        httpd = HTTPServer(server_address, lambda *args, **kwargs: Server(*args, directory=directory, **kwargs))
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()
