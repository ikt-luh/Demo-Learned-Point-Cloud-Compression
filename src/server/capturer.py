import os
import time
import gzip
import json
from lxml import etree

class Capturer:
    def __init__(self, output_dir, segment_duration, qualities=["V100", "V200", "V300"]):
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.qualities = qualities  # Simulating 3 different qualities for the video

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize MPD structure
        self.mpd_root = etree.Element("MPD", 
                                      xmlns="urn:mpeg:dash:schema:mpd:2011",
                                      xmlns_xsi="http://www.w3.org/2001/XMLSchema-instance",
                                      xsi_schemaLocation="urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd",
                                      profiles="urn:mpeg:dash:profile:isoff-live:2011,http://dashif.org/guidelines/dash-if-simple",
                                      type="dynamic", 
                                      availabilityStartTime="1970-01-01T00:00:00Z",
                                      publishTime="1970-01-01T00:00:00Z", 
                                      minimumUpdatePeriod="PT2S", 
                                      minBufferTime="PT2S", 
                                      timeShiftBufferDepth="PT1M", 
                                      maxSegmentDuration="PT2S")
        
        self.period = etree.SubElement(self.mpd_root, "Period", id="P0", start="PT0S")
        self.adaptation_set = None

    def setup_adaptation_set(self):
        """Create or reset the AdaptationSet in the MPD."""
        if self.adaptation_set is None:
            self.adaptation_set = etree.SubElement(self.period, "AdaptationSet", 
                                                   contentType="pointcloud", 
                                                   maxFrameRate="60/2", )
            etree.SubElement(self.adaptation_set, "Role", 
                             schemeIdUri="urn:mpeg:dash:role:2011", value="main")
            etree.SubElement(self.adaptation_set, "SegmentTemplate", 
                             timescale="1000000", 
                             duration="2002000", 
                             availabilityTimeOffset="1.969", 
                             availabilityTimeComplete="false", 
                             initialization="chunk-stream-$RepresentationID$/init-$RepresentationID$.m4s", 
                             media="chunk-stream-$RepresentationID$/$Number%05d$.m4s", 
                             startNumber="1")

            for idx, quality in enumerate(self.qualities):
                etree.SubElement(self.adaptation_set, "Representation", 
                                 id=f"{quality}", 
                                 mimeType="video/mp4", 
                                 codecs="avc1.64001e", 
                                 bandwidth=str(100000 * (idx + 1)), 
                                 width="640", 
                                 height="360", 
                                 sar="1:1")

    def update_mpd(self, segment_index):
        """Update the MPD with the latest segment number."""
        self.mpd_root.set("publishTime", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.adaptation_set.find("./SegmentTemplate").set("startNumber", str(segment_index))

    def create_init_file(self, quality):
        """Create the initialization file for each quality."""
        init_data = {
            "schema": "custom-schema-v1",
            "fields": ["timestamp", "value"],
            "timescale": 1000  # 1ms timescale
        }
        quality_folder = os.path.join(self.output_dir, f"chunk-stream-{quality}")
        os.makedirs(quality_folder, exist_ok=True)
        init_file_path = os.path.join(quality_folder, f"init-{quality}.m4s")

        if not os.path.exists(init_file_path):
            with gzip.open(init_file_path, "wt") as f:
                json.dump(init_data, f)

    def save_mpd(self):
        """Save the current MPD to a file."""
        mpd_path = os.path.join(self.output_dir, "manifest.mpd")
        mpd_content = etree.tostring(self.mpd_root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        with open(mpd_path, "wb") as f:
            f.write(mpd_content)

    def capture(self):
        """Simulate capturing and updating the MPD dynamically."""
        segment_index = 1
        segment_start_time = time.time()
        self.setup_adaptation_set()

        for data in self.generate_live_data():
            for quality in self.qualities:
                self.create_init_file(quality)

                quality_folder = os.path.join(self.output_dir, f"chunk-stream-{quality}")
                os.makedirs(quality_folder, exist_ok=True)

                segment_file = f"{segment_index:05d}.m4s"
                segment_path = os.path.join(quality_folder, segment_file)
                segment_data = self.generate_segment_data(quality)

                with gzip.open(segment_path, "wt") as f:
                    json.dump(segment_data, f)

            if time.time() - segment_start_time >= self.segment_duration:
                segment_index += 1
                segment_start_time = time.time()
                self.update_mpd(segment_index)
                self.save_mpd()

    def generate_live_data(self):
        """Simulate live data generation."""
        while True:
            yield {"timestamp": time.time(), "value": f"example-{int(time.time() * 1000)}"}
            time.sleep(0.5)

    def generate_segment_data(self, quality):
        """Generate dummy segment data."""
        return {"quality": quality, "data": [{"timestamp": time.time(), "value": f"example-{int(time.time() * 1000)}"} for _ in range(10)]}
