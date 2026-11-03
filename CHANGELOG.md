# Changelog

All notable engineering changes to this repository are documented here.

The format follows the intent of [Keep a Changelog](https://keepachangelog.com/) and uses semantic versioning principles where releases are created.

## [Unreleased]

### Added
- `makma.v1` contract (`llm_eval.contracts`, `contracts/makma.v1.schema.json`) with `evaluate`, `compare`, `validate_citations`, `score_retrieval` and `ingest_traces`, plus `GET /v1/capabilities` and `POST /v1/ops/{operation}`.
- Per-metric release gate with critical budgets, paired bootstrap intervals and exact McNemar test (`llm_eval.release_gate`).
- Failure taxonomy, citation validation, retrieval metrics and trace ingestion with nearest-rank p50/p95.

### Fixed
- `regression_gate` accepted NaN/out-of-range thresholds and then passed any summary; invalid thresholds now raise `ValueError`.
- `run_versioned_experiment` recorded the provider/model of the second output when the first output reported none, instead of `"mixed"`.
- `dataset_fingerprint` and `build_versioned_dataset` produced different SHA-256 values for non-ASCII datasets; both now use one canonical encoding (ASCII fingerprints are unchanged).
- Multi-word expected terms could never match in lexical relevance.

### Engineering
- Added repository ownership and structured contribution workflows.
- Added Dependabot coverage for Python and GitHub Actions dependencies.
- Added issue and pull-request templates focused on reproducibility, validation, risk, and rollback.
- Added consistent editor configuration and engineering standards.

## Release policy

A release should describe user-visible behavior, compatibility notes, validation evidence, and any migration or rollback requirements. Repository history remains the source of truth for individual commits.
