import os
import re
from horley_encoding import convert_to_horley


all_texts = {}

filenames = os.listdir("data/raw_texts/")
for filename in filenames:
    with open(f"data/raw_texts/{filename}", "r") as file:
        lines = file.readlines()
        raw_text = [line.split(',')[1][:-1] for line in lines]
        labels = [line.split(',')[0] for line in lines]
        clean_lines = [line.split('-') for line in raw_text]
        encoded_lines = [[convert_to_horley(glyph) for glyph in line] for line in clean_lines]
        for label, encoded_line in zip(labels, encoded_lines):
            all_texts[label] = ' '.join(encoded_line)


def search_glyphs(glyph_list):
    results = {}
    for glyphs in glyph_list:
        results[glyphs] = []
        for line in all_texts:
            pattern = fr"\b{'(.)'.join(glyphs)}\b"
            matches = re.findall(pattern, all_texts[line])
            if matches:
                results[glyphs].append(line)
    return {' '.join(k): v for k, v in results.items() if v}

