"""No generation before a token quote and budget reservation; retain uncertain charges."""

import json
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_fps_agent.config import load_config
from embodied_fps_agent.costs import BudgetExceeded, CostLedger
from embodied_fps_agent.policy import ClaudePolicy


def test_zero_and_small_budgets_reject_reservations():
    config = load_config()["api"]
    ledger = CostLedger(config)
    quote = ledger.quote(1000, 1024)
    assert Decimal(quote["maximum_cost_estimate_usd"]) > 0
    with pytest.raises(BudgetExceeded):
        ledger.reserve(quote)
    ledger = CostLedger({**config, "budget_usd": 0.00001})
    with pytest.raises(BudgetExceeded):
        ledger.reserve(ledger.quote(1000, 1024))
    assert ledger.records == []


def test_unknown_charges_keep_the_full_reservation(tmp_path):
    config = {**load_config()["api"], "budget_usd": 0.05}
    path = tmp_path / "costs.json"
    ledger = CostLedger(config, path)
    quote = ledger.quote(1000, 1024)
    index = ledger.reserve(quote)
    ledger.fail(index, "APITimeoutError")
    saved = json.loads(path.read_text())
    assert saved["observed_cost_usd"] == "0"
    assert Decimal(saved["unresolved_reserved_usd"]) > 0
    assert ledger.accounted() == Decimal(quote["maximum_cost_estimate_usd"])
    with pytest.raises(FileExistsError):
        CostLedger(config, path)


def test_actual_usage_cost_and_pricing_tier_are_checked():
    ledger = CostLedger({**load_config()["api"], "budget_usd": 0.05})
    index = ledger.reserve(ledger.quote(1000, 1024))
    row = ledger.settle(index, {"input_tokens": 1000, "output_tokens": 100})
    assert Decimal(row["actual_cost_usd"]) == Decimal("0.00015")
    assert ledger.accounted() == Decimal("0.00015")
    with pytest.raises(BudgetExceeded, match="tier"):
        ledger.quote(100000, 1024)
    mismatched = {**load_config()["api"], "model": "unpriced-model"}
    with pytest.raises(ValueError, match="pricing"):
        CostLedger(mismatched)


def test_counting_and_budget_check_happen_before_generation():
    calls = []

    def count_tokens(**kwargs):
        calls.append("count")
        assert "max_tokens" not in kwargs
        return SimpleNamespace(input_tokens=1000)

    def create(**_):
        calls.append("generate")
        raise AssertionError("Generation must not occur with a zero budget")

    policy = ClaudePolicy(
        load_config()["api"],
        client=SimpleNamespace(messages=SimpleNamespace(count_tokens=count_tokens, create=create)),
    )
    with pytest.raises(BudgetExceeded):
        policy.act(np.zeros((720, 960, 3), dtype=np.uint8))
    assert calls == ["count"] and policy.calls == 0


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"input_tokens": None, "output_tokens": 2},
        {"input_tokens": 2, "output_tokens": False},
    ],
)
def test_invalid_usage_never_becomes_a_zero_charge(usage):
    ledger = CostLedger({**load_config()["api"], "budget_usd": 0.05})
    quote = ledger.quote(1000, 1024)
    index = ledger.reserve(quote)
    with pytest.raises(ValueError):
        ledger.settle(index, usage)
    assert ledger.accounted() == Decimal(quote["maximum_cost_estimate_usd"])
    assert "actual_cost_usd" not in ledger.records[index]
