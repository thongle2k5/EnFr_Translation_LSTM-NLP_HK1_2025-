import gzip
import shutil
import os
RAW_DIR = "data/raw"
files_map = {
    "train.en.gz": "train.en",
    "train.fr.gz": "train.fr",
    "val.en.gz":   "val.en",
    "val.fr.gz":   "val.fr",
  
    "test_2016_flickr.en.gz": "test.en",
    "test_2016_flickr.fr.gz": "test.fr"
}
for gz_name, new_name in files_map.items():
    
    
    path_to_zip = os.path.join(RAW_DIR, gz_name)
    path_to_new = os.path.join(RAW_DIR, new_name)
  
    if os.path.exists(path_to_zip):
     
        with gzip.open(path_to_zip, 'rb') as f_in:
       
            with open(path_to_new, 'wb') as f_out:
        
                shutil.copyfileobj(f_in, f_out)
        print(f"Đã giải nén xong: {new_name}")
    else:
        print(f"Không tìm thấy file: {gz_name}")