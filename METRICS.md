# 📊 RAG Pipeline Metrics (M1-M24)

This system implements a comprehensive suite of 24 metrics to evaluate Retrieval Performance, Answer Quality, System Efficiency, and Legal-Specific Accuracy.

| ID      | Category              | Metric                        | Description                                                                        | Formula                                         |
| :------ | :-------------------- | :---------------------------- | :--------------------------------------------------------------------------------- | :---------------------------------------------- |
| **M1**  | Retrieval Performance | **Embedding Time**            | Avg time to embed 10 PDFs (sec). High latency slows down real-time pipelines.      | `t_end - t_start`                               |
| **M2**  | Retrieval Performance | **Index Size**                | Total embedding count. Larger indexes offer better coverage but cost more storage. | `Count(vectors)`                                |
| **M3**  | Retrieval Performance | **Retrieval Latency**         | Time to fetch top-k chunks. Direct impact on responsiveness.                       | `t_end - t_start`                               |
| **M4**  | Retrieval Performance | **Cosine Similarity**         | Semantic closeness of the retrieved document to the query (-1 to 1).               | `(A . B) / (\|\|A\|\| \|\|B\|\|)`               |
| **M5**  | Retrieval Performance | **Top-k Accuracy**            | % of queries where the relevant ground truth passage is in the top-k results.      | `(matches / total_queries) * 100`               |
| **M6**  | Answer Quality        | **ROUGE-1**                   | Word-level overlap (unigrams) with reference. Focuses on content coverage.         | `\|gt ∩ ans\| / \|gt\|`                         |
| **M7**  | Answer Quality        | **ROUGE-2**                   | Bigram overlap. Captures phrase-level similarity.                                  | `\|bigrams_gt ∩ bigrams_ans\| / \|bigrams_gt\|` |
| **M8**  | Answer Quality        | **ROUGE-L**                   | Longest Common Subsequence. Measures structural sentence usage.                    | `LCS(gt, ans) / len(gt)`                        |
| **M9**  | Answer Quality        | **Context Length**            | Number of tokens in the retrieved context passed to the LLM.                       | `tokenizer.encode(context)`                     |
| **M10** | Answer Quality        | **BLEU**                      | Strict exact phrase overlap. Precision-oriented "hallucination check".             | `BP * exp(Σ w log p)`                           |
| **M11** | Answer Quality        | **METEOR**                    | Flexible matching (synonyms/stemming). Better for semantic correctness.            | `F_mean * (1 - Penalty)`                        |
| **M12** | Answer Quality        | **BERTScore (F1)**            | Deep semantic similarity using contextual embeddings.                              | `2 * (P * R) / (P + R)`                         |
| **M13** | Answer Quality        | **Factual Consistency (FCD)** | Measures how grounded the answer is in the Context. Lower is better/closer.        | `(1 - BERTScore(Ans, Ctx)) * 100`               |
| **M14** | Answer Quality        | **Faithfulness**              | % of retrieved evidence correctly incorporated into the answer.                    | `(chunks_used / total_chunks) * 100`            |
| **M15** | Answer Quality        | **GT Coverage**               | % of ground truth words found in the answer.                                       | `(\|gt ∩ ans\| / \|gt\|) * 100`                 |
| **M16** | System Efficiency     | **E2E Latency**               | Total time from Query -> Answer.                                                   | `t_retrieval + t_gen`                           |
| **M17** | System Efficiency     | **Throughput**                | Queries handled per second.                                                        | `1 / (t_retrieval + t_gen)`                     |
| **M18** | System Efficiency     | **CPU Usage**                 | Average processor load during execution.                                           | `(used / total) * 100`                          |
| **M19** | System Efficiency     | **RAM Usage**                 | Memory consumption in GB.                                                          | `used_bytes / 1024^3`                           |
| **M20** | Legal-Specific        | **Citation Accuracy**         | % of correct case citations (e.g., "vs.", "AIR") retrieved.                        | `(correct_citations / total) * 100`             |
| **M21** | Legal-Specific        | **Terminology Precision**     | Correct usage of specifically legal terms (e.g., "plaintiff", "writ").             | `(correct_terms / total_terms) * 100`           |
| **M22** | Legal-Specific        | **Precedent Coverage**        | Frequency of retrieving multiple relevant case laws when needed.                   | `(multi_case_retrieval / total) * 100`          |
| **M23** | Answer Quality        | **FCD (Duplicate)**           | Same as M13. Factual alignment with context.                                       | `(1 - BERTScore(Ans, Ctx)) * 100`               |
| **M24** | Legal-Specific        | **Bias Score**                | Fairness metric quantifying presence of protected terms (caste/religion).          | `(protected_terms / total_terms) * 100`         |
