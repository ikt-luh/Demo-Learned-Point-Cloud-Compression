from http.server import SimpleHTTPRequestHandler, HTTPServer

class HTTPServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        self.output_directory = directory
        super().__init__(*args, **kwargs)

    @staticmethod
    def start(directory, port=8080):
        server_address = ('', port)
        httpd = HTTPServer(
            server_address,
            lambda *args, **kwargs: HTTPServerHandler(*args, directory=directory, **kwargs)
        )
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()
