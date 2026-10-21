"""Inspectable lexical retrieval and evaluation without external dependencies."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys


def tokenize(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class Index:
    def __init__(self, documents):
        if not isinstance(documents, list) or not documents:
            raise ValueError('Corpus must be a non-empty array')
        self.docs = {}
        self.terms = {}
        self.df = Counter()
        self.postings = {}
        for doc in documents:
            if not isinstance(doc, dict) or not isinstance(doc.get('id'), str) or not doc['id'] or not isinstance(doc.get('text'), str):
                raise ValueError('Every document needs a non-empty string id and string text')
            key = doc['id']
            if key in self.docs:
                raise ValueError(f'Duplicate document id: {key}')
            self.docs[key] = doc
            counts = Counter(tokenize(doc['text']))
            self.terms[key] = counts
            self.df.update(counts.keys())
            for term in counts:
                self.postings.setdefault(term, set()).add(key)
        self.n = len(self.docs)
        self.lengths = {key: sum(terms.values()) for key, terms in self.terms.items()}
        self.average_length = sum(self.lengths.values()) / self.n or 1
        self.idf = {term: math.log((1 + self.n) / (1 + freq)) + 1 for term, freq in self.df.items()}
        self.norms = {key: math.sqrt(sum(((1 + math.log(freq)) * self.idf[term]) ** 2 for term, freq in counts.items())) or 1 for key, counts in self.terms.items()}

    def search(self, query, method='bm25', k=5):
        if method not in ('bm25', 'tfidf'):
            raise ValueError('method must be bm25 or tfidf')
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []
        candidates = set().union(*(self.postings.get(term, set()) for term in query_terms))
        qnorm = math.sqrt(sum(((1 + math.log(freq)) * self.idf[term]) ** 2 for term, freq in query_terms.items() if term in self.idf)) or 1
        results = []
        for key in candidates:
            counts = self.terms[key]
            contributions = {}
            for term, qfreq in query_terms.items():
                freq = counts.get(term, 0)
                if not freq:
                    continue
                if method == 'bm25':
                    idf = math.log(1 + (self.n - self.df[term] + .5) / (self.df[term] + .5))
                    denom = freq + 1.2 * (1 - .75 + .75 * self.lengths[key] / self.average_length)
                    contribution = idf * freq * 2.2 / denom * qfreq
                else:
                    contribution = ((1 + math.log(freq)) * (1 + math.log(qfreq)) * self.idf[term] ** 2) / (self.norms[key] * qnorm)
                contributions[term] = contribution
            score = sum(contributions.values())
            if score > 0:
                doc = self.docs[key]
                results.append({'id': key, 'score': score, 'text': doc['text'], 'contributions': contributions})
        return sorted(results, key=lambda row: (-row['score'], row['id']))[:k]


def evaluate(index, cases, method='bm25', k=5):
    if not isinstance(cases, list) or not cases:
        raise ValueError('Evaluation set must be a non-empty array')
    details = []
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id'] or case['id'] in seen:
            raise ValueError('Evaluation cases require unique string ids')
        seen.add(case['id'])
        if not isinstance(case.get('query'), str) or not isinstance(case.get('relevant'), dict) or not case['relevant']:
            raise ValueError('Every case needs query text and non-empty relevance judgments')
        judgments = case['relevant']
        if any(key not in index.docs for key in judgments):
            raise ValueError('Judgment references a document outside the corpus')
        if any(isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3 for grade in judgments.values()):
            raise ValueError('Relevance grades must be integers from 0 to 3')
        relevant = {key for key, grade in judgments.items() if grade > 0}
        if not relevant:
            raise ValueError('Every query needs at least one positive judgment')
        ranked = index.search(case['query'], method, k)
        ids = [row['id'] for row in ranked]
        grades = [judgments.get(key, 0) for key in ids]
        def dcg(values):
            return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(values))
        ideal = dcg(sorted(judgments.values(), reverse=True)[:k])
        details.append({'id': case['id'], 'query': case['query'], 'ranked_ids': ids,
                        'recall': len(set(ids) & relevant) / len(relevant),
                        'reciprocal_rank': next((1 / rank for rank, key in enumerate(ids, 1) if key in relevant), 0),
                        'ndcg': dcg(grades) / ideal})
    return {'method': method, 'k': k, 'queries': len(details),
            'recall_at_k': sum(row['recall'] for row in details) / len(details),
            'mrr_at_k': sum(row['reciprocal_rank'] for row in details) / len(details),
            'ndcg_at_k': sum(row['ndcg'] for row in details) / len(details), 'per_query': details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['query', 'evaluate'])
    parser.add_argument('--corpus', type=Path, default=Path('data/documents.json'))
    parser.add_argument('--queries', type=Path, default=Path('data/queries.json'))
    parser.add_argument('--text', default='')
    parser.add_argument('--method', choices=['bm25', 'tfidf'], default='bm25')
    parser.add_argument('-k', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if args.k < 1:
            raise ValueError('k must be positive')
        index = Index(json.loads(args.corpus.read_text(encoding='utf-8')))
        result = index.search(args.text, args.method, args.k) if args.command == 'query' else evaluate(index, json.loads(args.queries.read_text(encoding='utf-8')), args.method, args.k)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, TypeError, AttributeError) as error:
        print(f'evidence-search: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
