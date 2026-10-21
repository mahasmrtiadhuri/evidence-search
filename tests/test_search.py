import math
import unittest
from search import Index, evaluate

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index=Index([{'id':'a','text':'worker lease crash recovery'},{'id':'b','text':'worker retry delay'},{'id':'c','text':'cloud secrets token'}])
    def test_both_methods_rank_exact_match_first(self):
        for method in ('bm25','tfidf'):
            results=self.index.search('lease recovery',method)
            self.assertEqual(results[0]['id'],'a')
            self.assertAlmostEqual(sum(results[0]['contributions'].values()),results[0]['score'])
    def test_empty_and_unknown_query(self):
        self.assertEqual(self.index.search(''),[])
        self.assertEqual(self.index.search('xyzunknown'),[])
    def test_case_and_punctuation(self):
        self.assertEqual(self.index.search('WORKER, Lease!')[0]['id'],'a')
    def test_ties_are_deterministic(self):
        index=Index([{'id':'z','text':'same'},{'id':'a','text':'same'}])
        self.assertEqual([r['id'] for r in index.search('same')],['a','z'])
    def test_cosine_bounded(self):
        for result in self.index.search('worker lease cloud','tfidf'):
            self.assertTrue(0 <= result['score'] <= 1 + 1e-12)
    def test_metrics_perfect_result(self):
        result=evaluate(self.index,[{'id':'q','query':'lease','relevant':{'a':3}}],k=1)
        self.assertEqual(result['recall_at_k'],1)
        self.assertEqual(result['mrr_at_k'],1)
        self.assertEqual(result['ndcg_at_k'],1)
    def test_metrics_partial_result(self):
        result=evaluate(self.index,[{'id':'q','query':'lease','relevant':{'a':1,'b':3}}],k=1)
        self.assertEqual(result['recall_at_k'],.5)
        self.assertAlmostEqual(result['ndcg_at_k'],1/7)
    def test_zero_result_metrics(self):
        result=evaluate(self.index,[{'id':'q','query':'unmatched','relevant':{'a':1}}])
        self.assertEqual(result['mrr_at_k'],0)
        self.assertEqual(result['ndcg_at_k'],0)
    def test_bad_corpus(self):
        for docs in ([],[{'id':'a','text':'x'},{'id':'a','text':'y'}],[{'id':None,'text':'x'}]):
            with self.assertRaises(ValueError): Index(docs)
    def test_invalid_judgments(self):
        for relevant in ({'missing':1},{'a':0},{'a':4},{'a':True}):
            with self.assertRaises(ValueError): evaluate(self.index,[{'id':'q','query':'a','relevant':relevant}])
    def test_empty_document(self):
        self.assertEqual(Index([{'id':'empty','text':''}]).search('x'),[])
