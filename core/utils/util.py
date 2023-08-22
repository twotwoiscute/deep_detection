import json
import os
from typing import List

def load_node_config(node_config_file):
    with open(node_config_file, "r") as file:
        config_data = json.load(file)
    config_data["config_path"] = os.path.abspath(node_config_file)
    return config_data

def _load_category_file(filename: str) -> List[str]:
    # read each line as a class
    with open(filename, "r") as f:
        category_list = [line.strip() for line in f.readlines()]
    return category_list