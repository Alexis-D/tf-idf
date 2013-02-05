from __future__ import print_function

import collections
import math
import os

import nltk
import redis


# TODO(alexis): better cli.
# TODO(alexis): search not only files.
# TODO(alexis): use celery.


def tokenize(text):
    text = text.lower()
    words = nltk.tokenize.regexp_tokenize(text, r'\w+')
    stemmer = nltk.stem.lancaster.LancasterStemmer()
    return [stemmer.stem(word) for word in words]


class TfIdfException(Exception):
    pass


class TfIdf(object):
    _prefix = 'tf-idf'

    # :documents: set of all documents
    # :documents:count: number of documents
    # :tf:<word>:<docid>: float
    # :inverted:<word>: set of docids
    # :idfd:<word>: denominator of idf

    def __init__(self):
        self._redis = redis.StrictRedis()

    def add(self, docid):
        docid = os.path.abspath(docid)
        key = '{}:documents'.format(self._prefix)

        if self._redis.sismember(key, docid):
            self.update(docid)

        else:
            self._score(docid)

    def remove(self, docid):
        raise NotImplemented('Not implemented yet.')

    def update(self, docid):
        self.remove(docid)
        self.add(docid)

    def search(self, query):
        # TODO(alexis): cosine distance & so on. So far search for a single
        # word.
        key = '{}:inverted:{}'.format(self._prefix, query)
        words = tokenize(query)
        docids = self._get_docids(words)
        key = '{}:documents:count'.format(self._prefix)  # number of docs
        documents = float(self._redis.get(key))

        try:
            idfds = self._get_idfds(words)
        except TfIdfException:
            return []

        tfs = self._get_tfs(words, docids)

        vectors = {}

        for docid in docids:
            vectors[docid] = [tfs[(word, docid)] *
                    math.log(documents / idfds[word]) for word in words]

        query_vector = [math.log(documents / idfds[word]) for word in words]

        results = []

        for docid, vector in vectors.items():
            results.append((docid, sum(vector)))

        return sorted(results, key=lambda k: k[1], reverse=True)

    def _get_docids(self, words):
        key = '{}:inverted:{}'
        keys = [key.format(self._prefix, word) for word in words]
        return self._redis.sinter(*keys)

    def _get_idfds(self, words):
        pipe = self._redis.pipeline()

        for word in words:
            key = '{}:idfd:{}'.format(self._prefix, word)
            pipe.get(key)

        result = pipe.execute()

        try:
            return {word: float(idfd) for word, idfd in zip(words, result)}
        except TypeError:
            raise TfIdfException("A word doesn't exist in db.")

    def _get_tfs(self, words, docids):
        pipe = self._redis.pipeline()

        for word in words:
            for docid in docids:
                key = '{}:tf:{}:{}'.format(self._prefix, word, docid)
                pipe.get(key)

        result = iter(pipe.execute())
        tfs = {}

        for word in words:
            for docid in docids:
                tf = next(result) or 0
                tfs[(word, docid)] = float(tf)

        return tfs

    def _score(self, docid):
        pipe = self._redis.pipeline()

        with open(docid) as f:
            text = f.read()

        words = tokenize(text)

        tfs = self._tfs(words)

        self._save_tfs(pipe, docid, tfs)
        self._save_inverted_index(pipe, docid, tfs.keys())
        self._save_idfds(pipe, docid, tfs.keys())

        pipe.execute()

    def _tfs(self, words):
        wc = len(words)
        tfs = {}

        for word, occurences in collections.Counter(words).items():
            tfs[word] = float(occurences) / wc

        return tfs

    def _save_tfs(self, pipe, docid, tfs):
        for word, tf in tfs.items():
            key = '{}:tf:{}:{}'.format(self._prefix, word, docid)
            pipe.set(key, tf)

    def _save_inverted_index(self, pipe, docid, words):
        for word in words:
            key = '{}:inverted:{}'.format(self._prefix, word)
            pipe.sadd(key, docid)

    def _save_idfds(self, pipe, docid, words):
        key = '{}:documents'.format(self._prefix)
        pipe.sadd(key, docid)
        key = '{}:documents:count'.format(self._prefix)
        pipe.incr(key)

        for word in words:
            key = '{}:idfd:{}'.format(self._prefix, word)
            pipe.incr(key)


if __name__ == '__main__':
    import sys

    tfidf = TfIdf()

    if 2 > len(sys.argv) > 3:
        print("tfidf.py takes two arguments.", file=sys.stderr)

    if sys.argv[1] == 'add':
        tfidf.add(sys.argv[2])

    elif sys.argv[1] == 'del':
        pass

    elif sys.argv[1] == 'search':
        results = tfidf.search(sys.argv[2])
        print('\n'.join(map(str, results)))

