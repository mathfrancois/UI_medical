import os
import time
import shutil

FOLDERS_TO_CLEAN = {
    'uploads': 3600,         # 1h
    'models': 86400,         # 24h
    'decompression': 3600,   # 1h
    'prediction': 3600       # 1h
}

def cleanup_old_files():
    now = time.time()
    for folder, max_age in FOLDERS_TO_CLEAN.items():
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isdir(file_path):
                    if now - os.path.getmtime(file_path) > max_age:
                        shutil.rmtree(file_path)
                else:
                    if now - os.path.getmtime(file_path) > max_age:
                        os.remove(file_path)
            except Exception as e:
                print(f"Erreur lors de la suppression de {file_path} : {e}")

if __name__ == "__main__":
    cleanup_old_files()
