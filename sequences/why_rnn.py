"""Script to examine why RNNs are useful for sequence data."""

import torch
import torch.nn as nn
import requests, zipfile, io, unicodedata, string

from torch.utils.data import DataLoader, Dataset

zip_file_url = "https://download.pytorch.org/tutorial/data.zip"

r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

namge_language_data = {}
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
alphabet = {letter: index for index, letter in enumerate(all_letters)}


def unicode_to_ascii(s):
    """Convert unicode string to ascii."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


for zip_path in z.namelist():
    if "data/names/" in zip_path and zip_path.endswith(".txt"):
        lang = zip_path[len("data/names/") : -len(".txt")]
        with z.open(zip_path) as myfile:
            lang_names = [
                unicode_to_ascii(line).lower()
                for line in str(myfile.read(), encoding="utf-8").strip().split("\n")
            ]
            namge_language_data[lang] = lang_names
        print(lang, ": ", len(lang_names))
