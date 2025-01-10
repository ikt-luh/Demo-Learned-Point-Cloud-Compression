import requests
import time

class SegmentDownloader:
    def __init__(self, fixed_quality_mode=True, init_quality=0):
        self.estimated_bandwidth = 100000 # Init
        self.fixed_quality_mode = fixed_quality_mode
        self.current_quality = init_quality
        self.segment_duration = 1.0


    def download_segment(self, base_url, media_template, segment_number):
        # Prepare URL
        self.decide_quality()
        segment_url = base_url + "/" + media_template.replace("$Number$", "{:015d}".format(segment_number))
        segment_url = segment_url.replace("$RepresentationID$", str(self.current_quality))

        for attempt in range(3):
            try:
                response = requests.get(segment_url, stream=True)

                if response.status_code == 200:
                    return response.content

                time.sleep(self.segment_duration / 3)
            except Exception as e:
                print(f"Failed to download segment {segment_number}, attempt {attempt + 1}: {e}")

        return None

    def decide_quality(self):
        """Determine quality level based on bandwidth."""
        # Fixed Quality Mode
        if self.fixed_quality_mode is not None:
            return self.current_quality
        
        # Bandwidth estimation #TODO
        if self.bandwidth > 2000:
            return 0
        elif self.bandwidth > 1000:
            return 1
        return 2