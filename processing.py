from horley_encoding import convert_to_horley


def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = [line.split(',')[1][:-1] for line in file.readlines()]
    return raw_data

def clean_line(line):
    replacements = {
        "(": "",
        ")": "",
        "128": "001V.076",
        "-999": "",
        ".076.": ".076-",
        "-022h-": "-"
    }
    for old, new in replacements.items():
        line = line.replace(old, new)
    return line.split('-')


def clean_lines(lines):
    return [clean_line(line) for line in lines]


def encode_lines(lines):
    return [[convert_to_horley(glyph) for glyph in line] for line in lines]


def split_sequences(lines):
    sequences = []
    for line in lines:
        i = 0
        j = 1
        while j < len(line):
            if line[j][-3:] == '.76':
                sequences.append(line[i:j])
                i = j
            j += 1
    for sequence in sequences:
        sequence[0] = sequence[0][:-3]
        sequence.insert(1, '<76>')
    filtered_sequences = [seq for seq in sequences if seq[0] and len(seq) >= 4]
    return sequences, filtered_sequences

