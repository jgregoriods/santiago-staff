import re


DELIMITERS = [' a ', ' o ', ' ote ', ' kote ', ' ate ']
DELIMITER_PATTERN = re.compile('|'.join(map(re.escape, DELIMITERS)))
PREFIX_PATTERN = re.compile(r"^(ko te |ko |kote|a |te )")


def clean_name(name):
    return PREFIX_PATTERN.sub('', name)


def split_and_extract_name(cleaned_name):
    parts = re.split(DELIMITER_PATTERN, cleaned_name)
    first_word = parts[0].split()[0] if parts and parts[0] else ''
    return first_word


def count_names(names):
    name_counts = {name: names.count(name) for name in set(names) if name}
    sorted_counts = sorted(name_counts.items(), key=lambda x: -x[1])
    return sorted_counts
