# Metadata Retrieval Ablation

This development-only ablation tests whether selected spaCy entities improve frozen `retrieval-v1` rankings without retuning retrieval.

`prepare_metadata_matches.py` extracts query entities and resolves exact same-label normalized matches against independently persisted document mentions. It does not read retrieval rankings or relevance judgments. `evaluate_metadata_reranking.py` then stable-partitions each frozen hybrid top ten so document/page candidates with a match appear first while preserving original order within both groups.

The 15-document corpus contained 190 mentions: 163 organisations, 23 locations, and four people. spaCy found supported entity types in eight of 30 queries, six had an exact corpus match, and only two rankings changed. Hit@1, Evidence-Hit@1, MRR@10, and all top-ten coverage metrics were unchanged. nDCG@10 increased from 0.624721 to 0.627442 (`+0.002722`).

Because activation was sparse and no first-evidence metric improved, metadata remains stored and inspectable but is not part of the default retrieval score. This negative result prevents adding an unvalidated signal merely to demonstrate model use.
