import glob
import argparse
import csv
import re
import string
import nltk
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

CLUSTERS = ['./docs/doc1-*.txt', './docs/doc2-*.txt', './docs/doc3-*.txt', './docs/doc4-*.txt']

def argparser():
    """Training settings"""

    parser = argparse.ArgumentParser(description='SumBasic implementation.')

    parser.add_argument('method_name', type=str, default='leading', 
                        choices=['orig', 'best-avg', 'simplified', 'leading'],
                        help='Name of the summarizing method.')
    
    parser.add_argument('docs', type=str, default='./docs/doc1-*.txt', 
                        choices=CLUSTERS,
                        help='Cluster to summarize')

    return parser.parse_args()

def count_words(text):
    if type(text) is list:
        return len(text)
    return len(text.split())

class Preprocessing:

    STOPWORDS = set(stopwords.words('english'))
    PUNCTUATION = string.punctuation # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    def __init__(self, punctuation=PUNCTUATION, stopwords=STOPWORDS):
        self.stopwords = stopwords
        self.punctuation = punctuation
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, sentence):
        for punc in self.punctuation:
            sentence = sentence.replace(punc, ' ')
        sentence = sentence.lower()
        sentence = sentence.split() 
        sentence = [w for w in sentence if w not in self.stopwords]
        sentence = [self.lemmatizer.lemmatize(w) for w in sentence]
        return sentence


class Cluster(Preprocessing):

    def __init__(self, args):
        super(Cluster, self).__init__()
        self.method_name = args.method_name
        self.cluster_id = args.docs[10]
        self.output = self._get_output_path()
        self.cluster = self._load_cluster(args.docs)

    def _find_files(self, path):
        return glob.glob(path)

    def _split_paragraphe(self, paragraphe):
        paragraphe = re.split(r"[.]", paragraphe)
        paragraphe = ['{}.'.format(sentence).lstrip() for sentence in paragraphe if not sentence == '']
        return paragraphe

    def _load_cluster(self, docs):
        paths = self._find_files(docs)
        cluster = []
        for path in paths:
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    cluster.append(self._split_paragraphe(','.join(row)))
        return cluster

    def _compute_probabilities(self):
        frequencies, probabilities = {}, {}
        word_count = 0
        for sentence in self.cluster:
            sentence = self.preprocess(sentence)
            word_count += count_words(sentence)
            while sentence:
                w = sentence.pop()
                try:
                    frequencies[w] += 1
                except KeyError:
                    frequencies[w] = 1
        for word, freq in frequencies.items():
            probabilities[word] = freq / word_count
        return probabilities

    def _get_output_path(self):
        return '{}-{}.txt'.format(self.method_name, self.cluster_id)

class Leading(Cluster):

    def __init__(self, args):
        super(Leading, self).__init__(args)

    def _get_summary(self):
        cluster = self.cluster.copy()
        prev_summary = ''
        while cluster:
            new_summary = prev_summary + ' ' + cluster.pop(0)[0]
            if count_words(new_summary) > 100:
                return prev_summary
            prev_summary = new_summary
        return new_summary
    
    def summarize(self):
        summary = self._get_summary()
        output = open(self.output, "w")
        output.write(summary)
        output.close()

class SumBasic(Cluster):
    def __init__(self, args, method):
        super(SumBasic, self).__init__(args)
        self.method = method
        self.cluster = [s for p in self.cluster for s in p]
        self.probabilities = self._compute_probabilities()

    def _score(self, sentence):
        assert type(sentence) is str
        score = 0.
        sentence = self.preprocess(sentence)
        if not self.method == 'best-avg':
            max_word = self._get_max_proba_word()
            if max_word not in sentence:
                return -1
        for w in sentence:
            score += self.probabilities[w] / len(sentence)
        return score
    
    def _update_probabilities(self, sentence):
        assert type(sentence) is str
        sentence = self.preprocess(sentence) 
        for w in sentence:
            self.probabilities[w] *= self.probabilities[w]

    def _get_max_proba_word(self):
        max_prob = 0
        for word, prob in self.probabilities.items():
            if prob > max_prob:
                max_prob = prob
                max_word = word
        return max_word

    def _add_next_sentence(self, prev_summary):
        scores = []
        for sentence in self.cluster:
            scores.append(self._score(sentence))
        best_idx = np.argmax(scores)
        next_sentence = self.cluster.pop(best_idx)
        new_summary = prev_summary + ' ' + next_sentence
        if count_words(new_summary) > 100:
            return prev_summary
        if not self.method == 'simplified':
            self._update_probabilities(next_sentence)
        return new_summary

    def _get_summary(self):
        summary = ''
        while self.cluster:
            summary = self._add_next_sentence(summary)
        return summary

    def summarize(self):
        summary = self._get_summary()
        output = open(self.output, "w")
        output.write(summary)
        output.close()

if __name__ == '__main__':
    args = argparser()

    if args.method_name == 'leading':
        summarizer = Leading(args)
    else:
        summarizer = SumBasic(args, method=args.method_name)
    
    summarizer.summarize()

