import requests
import time

class SegmentDownloader:
    def __init__(self, base_url, segment_template, segment_duration):
        self.base_url = base_url
        self.segment_template = segment_template
        self.segment_duration = segment_duration

    def download_segment(self, quality, segment_number):
        segment_url = self.base_url + "/" + self.segment_template.replace(
            "$Number$", f"{segment_number:015d}"
        ).replace("$RepresentationID$", str(quality))

        for attempt in range(3):
            try:
                response = requests.get(segment_url, stream=True)
                if response.status_code == 200:
                    return response.content
                time.sleep(self.segment_duration / 3)
            except Exception as e:
                print(f"Failed to download segment {segment_number}, attempt {attempt + 1}: {e}")
        return None