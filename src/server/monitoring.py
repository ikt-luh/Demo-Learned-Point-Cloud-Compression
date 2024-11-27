import time
import os
from datetime import datetime, timezone


class Monitor():
    def __init__(self, output_dir, init_file, mpd_file, segment_duration):
        self.output_dir = output_dir
        self.mpd_file = mpd_file
        self.init_file = init_file
        self.segment_duration = segment_duration

    # Generate or update the MPD file
    def update_mpd(self, segment_count):
        now = datetime.now(timezone.utc).isoformat()
        mpd_content = f"""<?xml version="1.0" encoding="utf-8"?>
        <MPD xmlns="urn:mpeg:dash:schema:mpd:2011"
            type="dynamic"
            publishTime="{now}"
            minimumUpdatePeriod="PT2S"
            timeShiftBufferDepth="PT30S"
            maxSegmentDuration="PT{self.segment_duration}S"
            profiles="urn:mpeg:dash:profile:isoff-live:2011">
          <Period start="PT0S">
            <!-- Point Cloud Adaptation Set -->
            <AdaptationSet contentType="pointcloud" res="1024" maxFrameRate="60/2" segmentAlignment="true" mimeType="pointcloud/custom" startWithSAP="1">
              <Role schemeIdUri="urn:mpeg:dash:role:2011" value="main"></Role>
              <SegmentTemplate media="$RepresentationID$/$Number$.m4s" initialization="$RepresentationID$/init.mp4" duration="{self.segment_duration}" startNumber="0"></SegmentTemplate>
      
              <Representation id="V300" bandwidth="300000" sar="1:1" frameRate="60/2" codecs="uni"></Representation>
              <Representation id="V600" bandwidth="600000" sar="1:1" frameRate="60/2" codecs="uni"></Representation>
              <Representation id="V1200" bandwidth="1200000" sar="1:1" frameRate="60/2" codecs="uni"></Representation>
            </AdaptationSet>

            <!-- Other AdaptationSets can go here if needed -->
          </Period>
        </MPD>
        """
        with open(self.mpd_file, "w") as f:
            f.write(mpd_content)
        print(f"MPD file updated: {self.mpd_file}")


    def monitor_segments(self):
        segment_count = 0
        while True:
            # Check for new segments
            current_segments = len([f for f in os.listdir(self.output_dir) if f.endswith(".m4s")])
            if current_segments > segment_count:
                segment_count = current_segments
                self.update_mpd(segment_count)
            time.sleep(0.1) 