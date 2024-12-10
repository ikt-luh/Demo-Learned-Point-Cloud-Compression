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
        self.representations = {}  # Store representations dynamically
        self.initialized = False

    def setup_adaptation_set(self):
        self.adaptation_set = etree.SubElement(
            self.period,
            "AdaptationSet",
            mimeType="pointcloud/custom",
            contentType="pointcloud",
            maxFrameRate="30",
            subsegmentAlignment="true",
            subsegmentStartsWithSAP="1",
        )
        etree.SubElement(
            self.adaptation_set,
            "SegmentTemplate",
            duration="120",
            timescale="30",
            media="ID$RepresentationID$/segment-$Number$.bin",
            startNumber="1",
            initialization="$RepresentationID$/$RepresentationID$_0.m4s",
        )

    def add_representation(self, rep_id, mime_type, codecs, bandwidth):
        if rep_id in self.representations:
            return  # Avoid duplicate representations

        representation = etree.SubElement(
            self.adaptation_set,
            "Representation",
            id=str(rep_id),
            mimeType=mime_type,
            codecs=codecs,
            bandwidth=str(bandwidth),
        )
        self.representations[rep_id] = {"element": representation, "segments": []}

        # Create the correct folder structure for initialization files
        init_path = os.path.join(self.output_directory, f"ID{rep_id}/init_{rep_id}.m4s")
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        with open(init_path, "wb") as init_file:
            init_file.write(b"")  # Placeholder for initialization data

    def update_segment(self, rep_id, segment_number, segment_path, bandwidth):
        if rep_id not in self.representations:
            raise ValueError(f"Representation {rep_id} not found.")

        # Update bandwidth dynamically
        representation_element = self.representations[rep_id]["element"]
        representation_element.set("bandwidth", str(bandwidth))

        # Track segment information
        self.representations[rep_id]["segments"].append((segment_number, segment_path))

    def update_metadata(self):
        # Update MPD publish time
        self.mpd_root.set(
            "publishTime", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    def save_mpd(self):
        mpd_path = os.path.join(self.output_directory, "manifest.mpd")
        mpd_content = etree.tostring(self.mpd_root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        with open(mpd_path, "wb") as f:
            f.write(mpd_content)
