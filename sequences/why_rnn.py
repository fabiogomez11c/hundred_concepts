"""Script to examine why RNNs are useful for sequence data."""

import torch
import torch.nn as nn
import requests, zipfile, io, unicodedata, string
from typing import Dict

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


class LanguageNameDataset(Dataset):
    def __init__(self, language_dict: Dict, vocabulary: Dict):
        self.label_names = [x for x in language_dict.keys()]
        self.data = []
        self.labels = []
        self.vocabulary = vocabulary
        for y, language in enumerate(self.label_names):
            for name in language_dict[language]:
                self.data.append(name)
                self.labels.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        name = self.data[idx]
        label = self.labels[idx]
        label_vec = torch.tensor([label], dtype=torch.long)

        return self.string2integervec(name), label_vec

    def string2integervec(self, input_string: str):
        """
        Convert string (name) to an tensor of integers according to the vocab.
        In this case, each word is the token.
        """
        T = len(input_string)
        name_vec = torch.zeros(T, dtype=torch.long)

        for pos, character in enumerate(input_string):
            name_vec[pos] = self.vocabulary[character]

        return name_vec
