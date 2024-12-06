import os
from lxml import etree
import time

class MPDManager:
    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.mpd_root = etree.Element(
            "MPD",
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
            maxSegmentDuration="PT2S",
        )
        self.period = etree.SubElement(self.mpd_root, "Period", id="P0", start="PT0s")
        self.adaptation_set = None

    def setup_adaptation_set(self):
        self.adaptation_set = etree.SubElement(
            self.period, "AdaptationSet", contentType="pointcloud", maxFrameRate="30"
        )
        etree.SubElement(self.adaptation_set, "Role", schemeIDUri="urn:mpeg:dash:role:2011", value="main")
        etree.SubElement(
            self.adaptation_set,
            "SegmentTemplate",
            timescale="1000000",
            duration="2002000",
            availabilityTimeOffset="1.969",
            availabilityTimeComplete="false",
            initialization="chunk-stream-$RepresentationID$/init-$RepresentationID$.m4s",
            media="chunk-stream-$RepresentationID$/$Number%05d$.m4s",
            startNumber="1",
        )
        etree.SubElement(
            self.adaptation_set,
            "Representation",
            id="1",
            mimeType="pointcloud/custom",
            codecs="unified",
            bandwidth="999",
        )

    def update_mpd(self, segment_index, bandwidth):
        self.mpd_root.set(
            "publishTime", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self.adaptation_set.find("./SegmentTemplate").set("startNumber", str(segment_index))
        self.adaptation_set.find("./Representation").set("bandwidth", str(bandwidth))

    def save_mpd(self):
        mpd_path = os.path.join(self.output_directory, "manifest.mpd")
        mpd_content = etree.tostring(self.mpd_root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        with open(mpd_path, "wb") as f:
            f.write(mpd_content)

