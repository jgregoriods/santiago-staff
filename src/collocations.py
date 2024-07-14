from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder
from nltk.lm.preprocessing import pad_both_ends


def get_collocations(sequences, ngram_finder_class, assoc_measures_class, ngram_filters, measure='likelihood_ratio', top_n=10):
    padded = [list(pad_both_ends(sequence, 2)) for sequence in sequences]
    collocation_measures = assoc_measures_class()
    results = []
    for ngram_filter in ngram_filters:
        finder = ngram_finder_class.from_documents(padded)
        finder.apply_freq_filter(2)
        finder.apply_ngram_filter(ngram_filter)
        if measure == 'likelihood_ratio':
            results.extend(finder.score_ngrams(collocation_measures.likelihood_ratio)[:top_n])
        elif measure == 'frequency':
            results.extend(finder.ngram_fd.most_common(top_n))
    return results


def get_bigram_collocations(sequences, measure='likelihood_ratio', top_n=10):
    bigram_filters = [
        lambda *w: w[1] != '<76>' or '?' in w[0],
        lambda *w: w[0] != '<76>' or '?' in w[1],
        lambda *w: w[-1] != '</s>' or '?' in w[0]
    ]
    return get_collocations(sequences, BigramCollocationFinder, BigramAssocMeasures, bigram_filters, measure, top_n)


def get_trigram_collocations(sequences, measure='likelihood_ratio', top_n=10):
    trigram_filters = [
        lambda *w: w[1] != '<76>' or '?' in w[0] or '?' in w[2],
    ]
    return get_collocations(sequences, TrigramCollocationFinder, TrigramAssocMeasures, trigram_filters, measure, top_n)


def is_similar(glyph1, glyph2):
    glyph1_set = set(glyph1.split('.'))
    glyph2_set = set(glyph2.split('.'))
    return len(glyph1_set.intersection(glyph2_set)) > 0


def get_similar_glyphs(sequences):
    repeated_sequences = {
        'XYX': [seq for seq in sequences if is_similar(seq[0], seq[-1])],
        'XXZ': [seq for seq in sequences if is_similar(seq[0], seq[2])],
        'XYY': [seq for seq in sequences if is_similar(seq[2], seq[-1])],
    }

    percentages = {key: len(value) / len(sequences) for key, value in repeated_sequences.items()}

    return repeated_sequences, percentages
