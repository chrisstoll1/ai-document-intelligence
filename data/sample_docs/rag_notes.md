# RAG and Source Evidence Notes

Retrieval-augmented generation retrieves source passages before asking a language model to write an answer. This can reduce unsupported answers because the model is given evidence from the document collection.

RAG is still not automatically trustworthy. If the retrieval step finds weak passages, the generated answer may also be weak. A good interface should show the source passages beside the summary so the user can inspect the evidence.

For the final system, generated summaries should be short and should cite the retrieved passages used to create them. The user should be able to distinguish between original source text and generated explanation.
