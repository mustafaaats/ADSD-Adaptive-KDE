# Auxiliary LLM-as-judge evaluation

Two user-supplied ChatGPT responses, both reported as GPT-6 Instant, were scored against 40 synthetic held-out records. The task was auxiliary numerical anomaly assessment, not direct judging of ADSD predictions or expert validation. Separate fresh chats were requested; their use has not been explicitly confirmed.

## Archived materials

- `judge_prompt_plain.txt`: prompt used for the two scored responses.
- `response_1.txt`, `response_2.txt`: responses supplied by the user, preserved as plain text.
- `answer_key_private.json`: labels and anomaly types withheld during collection; released here after both responses were collected. Do not supply this file to a judge in a replication.
- `protocol.json`, `judge_metadata.json`: sampling, thresholds, format change, and known collection metadata.
- `score_judge.py`, `evaluation_summary.json`: scoring script and computed results.
- `scored_run_1.json`, `scored_run_2.json`: judgments joined to hidden labels after collection.
- `prepare_judge.py`: reconstruction script for prompts and labels.
- `judge_prompt.txt`: initial JSON-output prompt; an attempt displayed only an opening square bracket and produced no usable judgments. The display/generation cause was not established. Output format was then changed before scoring valid responses.

## Recompute scores

From the repository root, using the project environment:

```bash
python llm_judge/score_judge.py llm_judge/response_1.txt llm_judge/response_2.txt
```

This overwrites the derived evaluation summary and joined result files, not the response files.

Reconstruct the same records and prompt in a separate directory:

```bash
python llm_judge/prepare_judge.py --output /tmp/comp8240_judge_reconstruction
```

## Results

| Metric | Response 1 | Response 2 |
|---|---:|---:|
| Accuracy | 0.7750 | 0.8000 |
| Precision | 0.7895 | 0.8750 |
| Recall | 0.7500 | 0.7000 |
| F1 | 0.7692 | 0.7778 |
| ROC-AUC | 0.8363 | 0.8413 |
| Average precision | 0.8829 | 0.8913 |

Binary decisions agreed on 35/40 records (87.5%); Cohen's kappa was 0.7475. At the fixed score threshold of 50, both responses detected 5/5 global, 4/5 variance-inflated local, and 5/5 clustered anomalies. Dependency recall was 1/5 and 0/5, respectively.

## Interpretation limits

The diagnostic sample was deliberately balanced: 20 normal observations and five from each of four anomaly groups. It does not represent the population's 5% anomaly prevalence. Precision and AP cannot be compared directly with ADSD metrics on the full test set. Reference summaries use 6,983 training observations, excluding 17 revealed anomalies under the seed-1, 5% setting; hidden anomalies can remain in the reference pool. The two responses are repeated model outputs, not independent human annotations. With five examples per subtype, subtype findings are descriptive.
