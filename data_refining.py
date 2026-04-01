import json



def remove_duplicates(file):
    """Remove duplicate entries from the jsonl data"""
    with open(file, 'r') as f:
        data = json.load(f)