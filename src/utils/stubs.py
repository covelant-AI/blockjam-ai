import pickle
import os

def save_detections_to_stub(detections, stub_path):
    if stub_path is not None:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(detections, f)

def load_detections_from_stub(stub_path):
    if stub_path is not None:
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    return None