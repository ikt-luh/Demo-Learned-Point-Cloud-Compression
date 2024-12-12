from http.server import SimpleHTTPRequestHandler, HTTPServer

class HTTPServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        self.output_directory = directory
        super().__init__(*args, **kwargs)

    @staticmethod
    def start(directory, ip_addr, port=8080):
        server_address = (ip_addr, port)
        httpd = HTTPServer(
            server_address,
            lambda *args, **kwargs: HTTPServerHandler(*args, directory=directory, **kwargs)
        )
        print(f"Serving {directory} at http://{ip_addr}:{port}")
        httpd.serve_forever()
