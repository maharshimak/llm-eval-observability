# Advanced engineering: operational SLO gates

`llm_eval.slo` turns trace windows into explicit release/operations decisions.

A policy can require a minimum sample count, cap p95 latency, cap average request cost and require a
minimum success rate. Evaluation returns all failed reasons rather than stopping at the first one,
which makes CI and release reports easier to diagnose.

The module complements quality/evaluation gates: a model can be semantically good and still be
unacceptable to deploy if it is too slow, too expensive or unreliable.
