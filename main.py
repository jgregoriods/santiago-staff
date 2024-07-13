import os
import json
from segmentation import vectorize, get_distinctive_glyphs, plot_breakpoints, segment_text
from collocations import get_bigram_collocations, get_trigram_collocations, get_similar_glyphs
from processing import load_file, clean_lines, encode_lines, split_sequences
from nearest_neighbor import analyze_glyphs, glyph_bound
from discourse import plot_discourse
from search import search_glyphs


RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(data, file_name):
    with open(os.path.join(RESULTS_DIR, file_name), 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def main():
    # Santiago Staff
    os.makedirs(os.path.join(RESULTS_DIR, "I"), exist_ok=True)
    raw_data = load_file(f'data/I.csv')
    clean_data = clean_lines(raw_data)
    encoded_data = encode_lines(clean_data)
    _, sequences = split_sequences(encoded_data)

    bigrams = get_bigram_collocations(sequences)
    save_json(bigrams, f'I/bigrams.json')
    bigrams_by_frequency = get_bigram_collocations(sequences, measure='frequency')
    save_json(bigrams_by_frequency, f'I/bigrams_freq.json')

    trigrams = get_trigram_collocations(sequences)
    save_json(trigrams, f'I/trigrams.json')
    trigrams_by_frequency = get_trigram_collocations(sequences, measure='frequency')
    save_json(trigrams_by_frequency, f'I/trigrams_freq.json')

    _, percentages = get_similar_glyphs(sequences)
    save_json(percentages, f'I/percentages.json')

    vectorized_text, vectorizer = vectorize(encoded_data)
    bkpts = plot_breakpoints(vectorized_text, [1, 2], os.path.join(RESULTS_DIR, f'I/breakpoints.png'))

    segmented_text = segment_text(encoded_data, bkpts[0])
    glyphs = get_distinctive_glyphs(segmented_text)
    save_json(glyphs, f'I/glyphs.json')

    clustered_glyphs, _ = analyze_glyphs(encoded_data)
    plot_discourse(clustered_glyphs, encoded_data, bkpt=bkpts[0][0], save_path=os.path.join(RESULTS_DIR, f'I/discourse.png'))

    trigrams_formatted = [[f'{trigram[0][0]}.76', trigram[0][2]] for trigram in trigrams]
    trigrams_sorted = sorted(trigrams_formatted, key=lambda x: glyph_bound(x, encoded_data))
    plot_discourse(trigrams_sorted, encoded_data, bkpt=bkpts[0][0], save_path=os.path.join(RESULTS_DIR, f'I/trigram_discourse.png'))

    XY = [(trigram[0][0], trigram[0][2]) for trigram in trigrams + trigrams_by_frequency]
    search_results = search_glyphs(XY)
    save_json(search_results, f'I/search_results.json')

    # Gv
    os.makedirs(os.path.join(RESULTS_DIR, "Gv"), exist_ok=True)
    raw_data = load_file(f'data/Gv.csv')
    clean_data = clean_lines(raw_data)
    encoded_data = encode_lines(clean_data)
    _, sequences = split_sequences(encoded_data)

    bigrams = get_bigram_collocations(sequences)
    save_json(bigrams, f'Gv/bigrams.json')

    _, percentages = get_similar_glyphs(sequences)
    save_json(percentages, f'Gv/percentages.json')

    # T
    os.makedirs(os.path.join(RESULTS_DIR, "T"), exist_ok=True)
    raw_data = load_file(f'data/T.csv')
    clean_data = clean_lines(raw_data)
    encoded_data = encode_lines(clean_data)
    _, sequences = split_sequences(encoded_data)

    bigrams = get_bigram_collocations(sequences)
    save_json(bigrams, f'T/bigrams.json')


if __name__ == '__main__':
    main()
