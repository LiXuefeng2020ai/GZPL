from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample
from typing import List,Optional, Text
import os
import pdb
import json

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

class SnipsProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = None
    
    
    def get_examples(self, data_dir: str, domain: str) -> List[InputExample]:
        examples = []
        with open(os.path.join(data_dir,f"{domain}.json")) as f:
            data_str = f.read()
            data = json.loads(data_str)
            for i,label_dict in enumerate(data):
                guid = str(i)
                # text_a = label_dict["sentence"]
                # slot_type = label_dict["slot_type"]
                # text_b = f"which is the {slot_type} ?"  # 问题模板demo
                # text_b = f"{slot_type} is the ?"  # 问题模板demo
                slot_type = label_dict["slot_type"]
                text_a = f"what is the {slot_type} ?"
                text_b = label_dict["sentence"]
                label_list = label_dict["label"]
                if label_list == []:
                    label = "none"
                elif len(label_list)==1:
                    label = " ".join(label_list[0])
                else:
                    record_label_list = []
                    for span in label_list:
                        record_label_list.append(" ".join(span))
                    label = ",".join(record_label_list)
                example = InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    tgt_text=label
                )
                examples.append(example)
        return examples
    
    
    def pretrain_get_examples(self, data_dir: str, domain: str, key_words: str) -> List[InputExample]:
            examples = []
            pos_examples = []
            with open(os.path.join(data_dir,f"negetive_{domain}.json")) as f:
                data_str = f.read()
                data = json.loads(data_str)
                for i,label_dict in enumerate(data):
                    guid = str(i)
                    
                    slot_type = label_dict["slot_type"]
                    text_a = f"what is the {slot_type} ?"
                    text_b = label_dict["sentence"]
                    # text_b = f"music_item best_rating facility poi genre service country current_location city object_type object_name geographic_poi timeRange restaurant_name year movie_type track object_location_type spatial_relation cuisine object_part_of_series_type condition_description playlist restaurant_type sort movie_name rating_value condition_temperature party_size_description location_name party_size_number playlist_owner state entity_name served_dish rating_unit artist album object_select, which is the slot type of {slot_type} ?"  # 问题模板demo
                    # text_b = f"{slot_type}"
                    label = label_dict["label"]
                    
                    example = InputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=text_b,
                        tgt_text=label
                    )
                    examples.append(example)

                return examples
    
    def concate_get_examples(self, data_dir: str, domain: str, key_words:str) -> List[InputExample]:
        examples = []
        with open(os.path.join(data_dir,f"universal_{domain}.json")) as f:
            lines = json.load(f)
            for i,line in enumerate(lines):
                guid = str(i)
                domain_slots = " ".join(domain2slot[domain])
                text_a = f"what is the entities ? {domain_slots}"
                text_b = line['sentence']
                # label = line[label]
                assert len(line['entities'])==len(domain2slot[domain])
                label = ';'.join(line['entities'])
                example = InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    tgt_text=label
                )
                # print(1)
                # pdb.set_trace()
                examples.append(example)
        return examples
            
    
        
            
    
    
        
