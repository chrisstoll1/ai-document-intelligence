# Grounded Generation Evaluation

This benchmark compares local instruction models after frozen `retrieval-v1` search. It does not alter retrieval order, weights, chunking, embeddings, or the default five-context workflow.

## Preparation

Run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_generation_benchmark.py
```

The fixed seed `docintel-generation-development-v1` selects five span, five arithmetic, and two multi-span questions from the existing TAT-DQA development manifest. Official development answers and derivations are joined by query UID from the ignored raw development file. Six fixed candidate-blind questions with no answer in the corpus test insufficient-evidence behavior. Development preparation does not read locked-test gold. After `generation-v1` was frozen, a separate locked manifest joined all 20 previously selected test questions to released gold exactly once.

## Contract

The model receives the question and the five contexts returned by production hybrid search. Context aliases `C1` to `C5` preserve retrieval order for that request; stable chunk, document, and page provenance remains server-owned.

Answered output contains one or more concise claims, each with at least one exact supplied context ID. Insufficient evidence contains no claims. Malformed output, uncited claims, and invented citation IDs are generation failures rather than correct refusals. The server never repairs or silently drops unsupported citations.

## Evaluation

Candidate comparison records answer correctness, valid structured-output rate, valid-citation rate, evidence-page citation relevance, refusal accuracy, unsupported-answer rate, invalid-generation rate, latency, model revisions, runtime configuration, and all raw outputs. End-to-end status uses corpus answerability; context status requires refusal whenever frozen retrieval did not return judged evidence. Citation-ID validity is not treated as proof that a passage supports a claim; claim support and citation completeness require review under the declared human-authored rubric.

## Selection

Pinned Qwen 2.5 7B and Mistral 7B Instruct v0.3 were evaluated with identical cached inputs, deterministic bfloat16 inference, and JSON Schema-constrained decoding. Qwen was selected as `generation-v1` for higher end-to-end reference coverage (0.667 versus 0.500), retrieval-conditioned coverage (0.727 versus 0.545), and context-status accuracy (0.944 versus 0.778). Mistral was faster and more conservative, including correct refusal on the retrieval miss. Complete outputs and the frozen decision are stored under `evaluation/results/` and `evaluation/config/generation_v1.json`.

After freezing, selected Qwen was run once on all 20 locked answerable questions and the six unanswerable controls. Output validity, refusal accuracy, and page-relevant citation rate were 1.000, but reference coverage was 0.350 and all ten arithmetic questions failed coverage. The 36-item candidate-blind support/completeness review is prepared in `development_review.json` and remains explicitly awaiting project-owner review.

Resource profiling on the RTX 4090 recorded 19.2 seconds cold initialization, 1.60 seconds mean answer latency, and 16.84 GB peak reserved GPU memory.

## Limitations

This is a small development selection from financial reports. Frozen retrieval finds judged evidence within the first five results for fewer than all development questions, so end-to-end answer failure can originate in retrieval or generation. Candidate selection results must separate retrieval-conditioned generation from aggregate system performance.
