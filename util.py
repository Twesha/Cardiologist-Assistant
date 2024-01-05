import os
import random

def image_full_path():
    folder_path = r"utils\random_img"
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if not files:
        print("No files found in the specified folder.")
        return None

    random_file = random.choice(files)
    return os.path.join(folder_path, random_file)