import requests
import time

class SegmentDownloader:
    def __init__(self):
        pass

    def download_segment(self, base_url, media_template, quality, segment_number):
        # Prepare URL
        segment_url = base_url + "/" + media_template.replace("$Number$", "{:015d}".format(segment_number))
        segment_url = segment_url.replace("$RepresentationID$", str(quality))

        for attempt in range(3):
            try:
                response = requests.get(segment_url, stream=True)

                if response.status_code == 200:
                    return response.content

                time.sleep(self.segment_duration / 3)
            except Exception as e:
                print(f"Failed to download segment {segment_number}, attempt {attempt + 1}: {e}")

        return None