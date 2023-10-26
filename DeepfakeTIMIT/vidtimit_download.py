import os
import requests
from zipfile import ZipFile
import shutil

files = os.listdir("./Data/higher_quality")
files.remove(".dircksum")

# https://zenodo.org/record/158963/files/fadg0.zip

if "zip_files" not in os.listdir():
    os.mkdir("zip_files")

for file in files:
    if f"{file}.zip" not in os.listdir("zip_files"):
        url = f"https://zenodo.org/record/158963/files/{file}.zip"
        r = requests.get(url,allow_redirects=True)
        open(f"zip_files/{file}.zip","wb").write(r.content)

if "real_data" not in os.listdir():
    os.mkdir("real_data")

for file in files:
    if file not in os.listdir("real_data"):
        with ZipFile(f"zip_files/{file}.zip") as f:
            f.extractall(f"real_data")

if "real_images" not in os.listdir():
    os.mkdir("real_images")

img_count = 0

for file in os.listdir("real_data"):

    for i in os.listdir(f"real_data/{file}/video"):
        print(i)
        count = 0
        for j in os.listdir(f"real_data/{file}/video/{i}"):
            if count<8:
                shutil.copy(f"real_data/{file}/video/{i}/{j}",f"real_images/{img_count}")
                img_count+=1
                count+=1
            else:
                count=0
                break

for file in os.listdir("real_images"):
    os.rename(f"real_images/{file}",f"real_images/{file}.png")
