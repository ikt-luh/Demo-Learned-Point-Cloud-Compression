import os
import csv



def process_logs_and_save(data, file_path):
    flat_data = flatten_dict(data)

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=flat_data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(flat_data) 

        
def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)