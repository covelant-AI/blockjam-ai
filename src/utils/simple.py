from typing import Any, Dict, List, Optional
import os
import json

def find_object_by_key_value(
    obj: List[Dict[str, Any]], 
    key: str, 
    value: Any
) -> Dict[str, Any]:
    result = next((x for x in obj if x[key] == value), None)
    return result


def load_json_file(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            json_file = json.load(f)
    else:
        json_file = {}
    return json_file