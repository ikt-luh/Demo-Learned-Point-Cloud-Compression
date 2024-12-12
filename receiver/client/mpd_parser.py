import requests
import xml.etree.ElementTree as ET

class MPDParser:
    def __init__(self, mpd_url):
        self.mpd_url = mpd_url
        self.mpd_tree = None
        self.mpd_root = None

    def fetch_mpd(self):
        response = requests.get(self.mpd_url)
        if response.status_code == 200:
            self.mpd_tree = ET.ElementTree(ET.fromstring(response.content))
            self.mpd_root = self.mpd_tree.getroot()
        else:
            raise Exception(f"Failed to fetch MPD: HTTP {response.status_code}")

    def parse_mpd(self):
        if self.mpd_root is None:
            raise Exception("MPD not fetched. Call fetch_mpd() first.")

        mpd_namespace = {'': 'urn:mpeg:dash:schema:mpd:2011'}
        ET.register_namespace('', 'urn:mpeg:dash:schema:mpd:2011')

        mpd_data = {
            "type": self.mpd_root.get("type"),
            "availabilityStartTime": self.mpd_root.get("availabilityStartTime"),
            "publishTime": self.mpd_root.get("publishTime"),
            "minimumUpdatePeriod": self.mpd_root.get("minimumUpdatePeriod"),
            "minBufferTime": self.mpd_root.get("minBufferTime"),
            "timeShiftBufferDepth": self.mpd_root.get("timeShiftBufferDepth"),
            "maxSegmentDuration": self.mpd_root.get("maxSegmentDuration"),
            "periods": []
        }

        for period in self.mpd_root.findall(".//Period", mpd_namespace):
            period_data = {
                "id": period.get("id"),
                "start": period.get("start"),
                "adaptation_sets": []
            }

            for adaptation_set in period.findall(".//AdaptationSet", mpd_namespace):
                adaptation_data = {
                    "mimeType": adaptation_set.get("mimeType"),
                    "contentType": adaptation_set.get("contentType"),
                    "maxFrameRate": adaptation_set.get("maxFrameRate"),
                    "segment_template": {},
                    "representations": []
                }

                segment_template = adaptation_set.find(".//SegmentTemplate", mpd_namespace)
                if segment_template is not None:
                    adaptation_data["segment_template"] = {
                        "duration": int(segment_template.get("duration")),
                        "media": segment_template.get("media"),
                        "startNumber": int(segment_template.get("startNumber")),
                        "initialization": segment_template.get("initialization")
                    }

                for representation in adaptation_set.findall(".//Representation", mpd_namespace):
                    representation_data = {
                        "id": representation.get("id"),
                        "mimeType": representation.get("mimeType"),
                        "codecs": representation.get("codecs"),
                        "bandwidth": int(representation.get("bandwidth"))
                    }
                    adaptation_data["representations"].append(representation_data)

                period_data["adaptation_sets"].append(adaptation_data)

            mpd_data["periods"].append(period_data)

        return mpd_data