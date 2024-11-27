from client.client import StreamingClient 

mpd_url = 'http://localhost:8000/manifest.mpd'  # Your MPD URL
base_url = 'http://localhost:8000/'  # Your MPD URL
client = StreamingClient(base_url, mpd_url, "downloaded_segments")
client.stream()