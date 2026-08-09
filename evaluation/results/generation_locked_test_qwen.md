# Locked-Test Grounded Generation

Selected Qwen 2.5 7B was evaluated once on all 20 locked TAT-DQA questions plus six fixed corpus-unanswerable questions. Frozen `retrieval-v1` supplied judged page evidence in the first five contexts for all 20 answerable questions.

| Measure | Result |
|---|---:|
| Valid structured output | 1.000 |
| Answerable reference coverage | 0.350 |
| Span reference coverage | 0.714 |
| Multi-span reference coverage | 0.667 |
| Arithmetic reference coverage | 0.000 |
| Unanswerable refusal accuracy | 1.000 |
| Page-relevant citation rate | 1.000 |
| Mean generation latency | 1,335 ms |

The selected generator covered seven of 20 reference answers. Five answerable questions were refused despite retrieved judged evidence, and every one of the ten arithmetic answers failed automatic reference coverage. Span and multi-span extraction remained substantially stronger. This is the most consequential final-system weakness: grounded structure and valid citations prevent invented provenance but do not provide reliable numerical reasoning.

The page-relevant citation measure checks source document/page alignment, not semantic entailment. The blinded 36-item development review remains awaiting human support and completeness assessment.
