# from data_reader import domain2slot
import json
import os
import pdb
from typing import *


domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

unseen_slot = ["playlist_owner","entity_name","party_size_number","served_dish","restaurant_type","party_size_description","facility","restaurant_name","poi","cuisine","condition_description","geographic_poi","condition_temperature","current_location","year","album","genre","service","track","object_part_of_series_type","rating_value","object_select","best_rating","rating_unit","object_location_type","location_name","movie_name","movie_type"]

seen_slot = ["artist","playlist","music_item","country","state","timeRange","sort","spatial_relation","city","object_name","object_type"]

def snips_process(data_dir: str, domain: str) -> None:
    file = open(os.path.join(data_dir,f"{domain}.txt"))
    json_dict = []
    with open(os.path.join(data_dir,f"{domain}.json"),"w") as output_file:
        for i,line in enumerate(file):
            text_split = line.strip().split("\t")
            label_dict = label_process(text_split,domain)
            for slot_type in domain2slot[domain]:
                record_dict = {}
                record_dict["sentence"] = text_split[0]
                record_dict["slot_type"] = slot_type
                record_dict["label"] = label_dict[slot_type]
                json_dict.append(record_dict)
        output_file.write(json.dumps(json_dict,indent=0))
    file.close()
    output_file.close()                
                
def label_process(text_split:List[str], domain:str) -> Dict:
    label_dict = {}
    word_list = text_split[0].strip().split(" ")
    label_list = text_split[1].strip().split(" ")
    for slot_type in domain2slot[domain]:
        label_dict[slot_type] = []
    span_start = -1
    record_slot = "O"
    for i,label in enumerate(label_list):
        label_span = label.split("-")
        if len(label_span)==1:
            if span_start!=-1:
                label_dict[record_slot].append(word_list[span_start:i])
                span_start = -1
                record_slot = "O"
            else:
                continue
        
        else:
            if label_span[0]=="B":
                if span_start!=-1:
                    label_dict[record_slot].append(word_list[span_start:i])
                span_start = i
                record_slot = label_span[1]
            else:
                continue
    # 防止在结尾的实体没被统计
    if span_start!=-1:
        label_dict[record_slot].append(word_list[span_start:])
    return label_dict
    
if __name__ == '__main__':

    # print(label_process(text.strip().split("\t"),domain="AddToPlaylist"))
    for domain in domain2slot.keys():
        snips_process(data_dir=f"../data/snips/{domain}/",domain=domain)
        print(f"{domain} finished!!!")
    # with open(os.path.join("../data/snips/AddToPlaylist/","AddToPlaylist.json")) as f:
    #     string = f.read()
    #     data = json.loads(string)
    #     print(data[5])
    
            
            
            
    