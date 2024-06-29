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
        "999": "",
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

    # We will add a special token to represent glyph 076
    # when appended to the first glyph of the triad

    for i, sequence in enumerate(sequences):
        sequence[0] = sequence[0][:-3]
        sequence.insert(1, '<76>')

    sequences = [sequence for sequence in sequences if len(sequence) >= 4 and sequence[0] and '?' not in sequence]

    return sequences


def main():
    raw_data_I = load_file('data/I.csv')
    clean_data_I = clean_lines(raw_data_I)
    encoded_data_I = encode_lines(clean_data_I)
    sequences_I = split_sequences(encoded_data_I)
    print(sequences_I)


if __name__ == '__main__':
    main()