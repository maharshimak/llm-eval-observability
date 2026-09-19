# Security

Relevance is lexical overlap and citation coverage checks identifiers, not factuality or entailment. Candidate outputs supply latency and token counts; these are validated but not independently measured. Cost is calculated only when caller-provided rates are configured; the API defaults to zero rates. JSONL traces, dataset loading and comparison are library utilities, not an integrated dashboard. No provider calls, live telemetry collector or deployed release automation is included.

Use synthetic or explicitly authorized public data. Do not commit API keys, databases, model credentials, patient records or private employer material. Remote model adapters transmit supplied text to the configured endpoint; choose the provider deliberately.

For a suspected vulnerability, use GitHub private vulnerability reporting if enabled. Otherwise contact the maintainer privately through the [portfolio](https://maharshipatel-portfolio.vercel.app/). Do not put secrets or exploit payloads containing private data in a public issue. No response SLA is promised.
