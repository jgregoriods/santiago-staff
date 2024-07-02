import os
import json
from segmentation import vectorize, save_glyphs, plot_breakpoints
from collocations import get_bigram_collocations, get_trigram_collocations, get_similar_glyphs
from processing import load_file, clean_lines, encode_lines, split_sequences
from nearest_neighbor import analyze_glyphs
from discourse import plot_discourse
from search import search_glyphs


RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(data, file_name):
    with open(os.path.join(RESULTS_DIR, file_name), 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def main():
    for text in ['I', 'Gv', 'T']:
        os.makedirs(os.path.join(RESULTS_DIR, text), exist_ok=True)
        raw_data = load_file(f'data/{text}.csv')
        clean_data = clean_lines(raw_data)
        encoded_data = encode_lines(clean_data)
        _, sequences = split_sequences(encoded_data)

        bigrams = get_bigram_collocations(sequences)
        save_json(bigrams, f'{text}/bigrams.json')

        trigrams = get_trigram_collocations(sequences)
        save_json(trigrams, f'{text}/trigrams.json')

        _, percentages = get_similar_glyphs(sequences)
        save_json(percentages, f'{text}/percentages.json')

        vectorized_text, vectorizer = vectorize(encoded_data)
        bkpts = plot_breakpoints(vectorized_text, [1, 2], os.path.join(RESULTS_DIR, f'{text}/breakpoints.png'))

        glyphs = save_glyphs(vectorized_text, vectorizer, bkpts)
        save_json(glyphs, f'{text}/glyphs.json')

        clustered_glyphs, _ = analyze_glyphs(encoded_data)
        plot_discourse(clustered_glyphs, encoded_data, bkpt=bkpts[0][0], save_path=os.path.join(RESULTS_DIR, f'{text}/discourse.png'))

        trigrams_formatted = [[f'{trigram[0][0]}.76', trigram[0][2]] for trigram in trigrams]
        clustered_trigrams, _ = analyze_glyphs(trigrams_formatted)
        plot_discourse(clustered_trigrams, trigrams_formatted, bkpt=bkpts[0][0], save_path=os.path.join(RESULTS_DIR, f'{text}/trigram_discourse.png'))

        XY = [(trigram[0][0], trigram[0][2]) for trigram in trigrams]
        search_results = search_glyphs(XY)
        save_json(search_results, f'{text}/search_results.json')


if __name__ == '__main__':
    main()
