import requests
import xml.etree.ElementTree as ET

class MPDParser:
    def __init__(self, mpd_url):
        self.mpd_url = mpd_url
        self.representations = {}
        self.timescale = None

    def fetch_mpd(self):
        response = requests.get(self.mpd_url)
        response.raise_for_status()
        return ET.fromstring(response.content)

    def parse_mpd(self, root):
        namespace = {'mpd': 'urn:mpeg:dash:schema:mpd:2011'}
        period = root.find('mpd:Period', namespace)
        adaptation_set = period.find('mpd:AdaptationSet', namespace)
        
        self.timescale = int(adaptation_set.get('timescale', 1))
        
        for representation in adaptation_set.findall('mpd:Representation', namespace):
            rep_id = representation.get('id')
            bandwidth = int(representation.get('bandwidth'))
            segment_template = representation.find('mpd:SegmentTemplate', namespace)
            media_template = segment_template.get('media')
            start_number = int(segment_template.get('startNumber', 1))
            
            self.representations[rep_id] = {
                'bandwidth': bandwidth,
                'media_template': media_template,
                'next_segment': start_number
            }