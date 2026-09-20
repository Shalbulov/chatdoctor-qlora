# Evaluation, 20 held-out examples

| | base | base + adapter |
|---|---|---|
| ROUGE-L F1 | 0.119 | 0.168 |
| BERTScore F1 (rescaled) | -0.037 | 0.125 |
| wins vs the other column | 3 | 17 |
| mean words | 303 | 75 |
| mean characters | 2049 | 436 |
| answers with a bullet list | 95% | 0% |
| answers with bold text | 95% | 0% |
| answers with a heading | 0% | 0% |
| answers with a disclaimer | 60% | 20% |

Reference answers for comparison: 102 words, 0% with a bullet list, 15% with a disclaimer.
