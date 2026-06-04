from dataclasses import dataclass


from src.config import (
    PROMPT_TOKEN_COST_PER_1M,
    COMPLETION_TOKEN_COST_PER_1M
)


@dataclass
class UsageMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


def calculate_estimated_cost(
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    return (
        (prompt_tokens / 1_000_000) * PROMPT_TOKEN_COST_PER_1M
        +
        (completion_tokens / 1_000_000) * COMPLETION_TOKEN_COST_PER_1M
    )
