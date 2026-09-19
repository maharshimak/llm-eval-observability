from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    prompt: str
    expected_terms: set[str] = field(default_factory=set)
    expected_citations: set[str] = field(default_factory=set)
    forbidden_phrases: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    text: str
    citations: list[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    case_id: str
    relevance: float
    citation_coverage: float
    forbidden_hit: bool
    latency_ms: float
    estimated_cost_usd: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    name: str
    pass_rate: float
    mean_relevance: float
    mean_citation_coverage: float
    latency_p95_ms: float
    mean_cost_usd: float
    cases: list[CaseMetrics]
