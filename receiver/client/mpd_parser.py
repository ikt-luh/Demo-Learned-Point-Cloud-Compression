import sys
import time
import requests
import xml.etree.ElementTree as ET

class MPDParser:
    def __init__(self, mpd_url):
        self.mpd_url = mpd_url
        self.mpd_data = None


    def get_segment_duration(self):
        return float(self.mpd_data.get("maxSegmentDuration"))

    def get_publish_time(self):
        return self.mpd_data.get("publishTime")

    def get_media_template(self):
        return self.mpd_data['periods'][0]['adaptation_sets'][0]['segment_template']['media']

    def get_codec_info(self, quality):
        return self.mpd_data['periods'][0]["adaptation_sets"][0]["representations"][quality]["codecs"]

    def update_mpd(self):
        for attempt in range(3):  # Try 3 times
            try:
                response = requests.get(self.mpd_url)
            except:
                time.sleep(0.1)
                continue

            if response.status_code == 200 and response.content.strip():
                response_data = response.content
                self.parse_mpd(response_data)
                return True
            else:
                time.sleep(0.3)  # Wait before retrying
        return False



    def parse_mpd(self, response):
        mpd_tree = ET.ElementTree(ET.fromstring(response))
        mpd_root = mpd_tree.getroot()
        mpd_namespace = {'': 'urn:mpeg:dash:schema:mpd:2011'}
        ET.register_namespace('', 'urn:mpeg:dash:schema:mpd:2011')

        mpd_data = {
            "type": mpd_root.get("type"),
            "availabilityStartTime": mpd_root.get("availabilityStartTime"),
            "publishTime": mpd_root.get("publishTime"),
            "minimumUpdatePeriod": mpd_root.get("minimumUpdatePeriod"),
            "minBufferTime": mpd_root.get("minBufferTime"),
            "timeShiftBufferDepth": mpd_root.get("timeShiftBufferDepth"),
            "maxSegmentDuration": mpd_root.get("maxSegmentDuration"),
            "periods": []
        }

        for period in mpd_root.findall(".//Period", mpd_namespace):
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
                        "duration": float(segment_template.get("duration")),
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
        self.mpd_data = mpd_data
