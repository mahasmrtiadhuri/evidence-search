# Contributing

Run `python -m unittest discover -s tests -v`. Ranking changes need tests and a comparison on a fixed dataset. Metric changes need a hand-calculable expected value. Never tune a retriever on the held-out set used to report its performance.

Keep datasets free of private source text. Document dataset provenance, relevance-label instructions, and known limitations. Explain new dependencies and preserve deterministic ranking.
