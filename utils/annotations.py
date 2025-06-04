import json

def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_annotations(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)
