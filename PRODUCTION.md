# Production engineering

Aggregate pass rates can look strong with tiny sample sizes. `llm_eval.reliability` adds Wilson-score confidence intervals and a confidence-aware release gate.

A release can now require both a minimum evaluation sample size and a minimum lower confidence bound. This makes the gate conservative when evidence is weak and provides an explicit reason for a blocked release.

## Operational practice

- Keep deterministic regression gates and confidence gates separate.
- Version evaluation datasets and prompt/model configuration with every run.
- Use larger samples for high-impact changes.
- Track latency/cost gates alongside quality confidence.
