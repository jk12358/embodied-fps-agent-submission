"""Count before generation, reserve a conservative estimate, and account for actual usage."""

import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    def __init__(self, config: dict, path: Path | None = None):
        self.pricing = config["pricing"]
        if self.pricing["model"] != config["model"]:
            raise ValueError("Verify pricing for the selected model before making API calls")
        self.budget = Decimal(str(config["budget_usd"]))
        self.input_rate = Decimal(str(self.pricing["input_per_million_usd"])) / 1_000_000
        self.output_rate = Decimal(str(self.pricing["output_per_million_usd"])) / 1_000_000
        self.cache_read_multiplier = Decimal(str(self.pricing["cache_read_multiplier"]))
        if any(
            not v.is_finite() or v < 0
            for v in (self.budget, self.input_rate, self.output_rate, self.cache_read_multiplier)
        ):
            raise ValueError("Pricing and budget must be finite nonnegative numbers")
        self.records = []
        self.quotes = []
        self.path = path
        if path is not None and path.exists():
            raise FileExistsError("Do not overwrite a previous API cost ledger")

    def quote(self, input_tokens: int, max_output_tokens: int) -> dict:
        if (
            type(input_tokens) is not int
            or input_tokens < 0
            or type(max_output_tokens) is not int
            or max_output_tokens < 1
        ):
            raise ValueError("Invalid token counts")
        # count_tokens is an estimate, so keep input headroom instead of treating it as exact.
        guarded = (
            int((Decimal(input_tokens) * Decimal("1.1")).to_integral_value(rounding=ROUND_CEILING))
            + 128
        )
        if guarded > self.pricing["max_input_tokens_at_rate"]:
            raise BudgetExceeded("Input exceeds the verified pricing tier")
        maximum = guarded * self.input_rate + max_output_tokens * self.output_rate
        quote = {
            "input_tokens_estimate": input_tokens,
            "guarded_input_tokens": guarded,
            "max_output_tokens": max_output_tokens,
            "maximum_cost_estimate_usd": str(maximum),
            "remaining_budget_usd": str(self.budget - self.accounted()),
        }
        self.quotes.append(quote)
        self.save()
        return quote

    def accounted(self) -> Decimal:
        return sum(
            (Decimal(r.get("actual_cost_usd", r["reserved_usd"])) for r in self.records), Decimal(0)
        )

    def reserve(self, quote: dict) -> int:
        maximum = Decimal(quote["maximum_cost_estimate_usd"])
        if self.budget <= 0 or self.accounted() + maximum > self.budget:
            raise BudgetExceeded("Generation stopped: estimated request exceeds the run budget")
        self.records.append({**quote, "reserved_usd": str(maximum), "status": "pending"})
        self.save()
        return len(self.records) - 1

    def settle(self, index: int, usage: dict) -> dict:
        if any(key not in usage for key in ("input_tokens", "output_tokens")):
            raise ValueError("Missing API usage; keep the reservation as unresolved")
        counts = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = usage.get(key, 0)
            if key.startswith("cache_") and value is None:
                value = 0
            if type(value) is not int or value < 0:
                raise ValueError("Invalid API usage; keep the reservation as unresolved")
            counts[key] = value
        # No caching is requested. If the server reports it anyway, account for it explicitly.
        creation = usage.get("cache_creation") or {}
        if not isinstance(creation, dict):
            raise ValueError("Invalid cache breakdown")
        hour = creation.get("ephemeral_1h_input_tokens", 0) or 0
        if type(hour) is not int or not 0 <= hour <= counts["cache_creation_input_tokens"]:
            raise ValueError("Invalid cache usage")
        short = counts["cache_creation_input_tokens"] - hour
        actual = (
            counts["input_tokens"] * self.input_rate
            + counts["output_tokens"] * self.output_rate
            + short * self.input_rate * Decimal("1.25")
            + hour * self.input_rate * 2
            + counts["cache_read_input_tokens"] * self.input_rate * self.cache_read_multiplier
        )
        record = self.records[index]
        record.update(status="received", usage=counts, actual_cost_usd=str(actual))
        record["estimate_exceeded"] = actual > Decimal(record["reserved_usd"])
        self.save()
        return record

    def fail(self, index: int, error_type: str):
        self.records[index].update(status="cost_unknown", error_type=error_type)
        self.save()  # A failed/ interrupted request keeps its full reservation; do not assume $0.

    def summary(self) -> dict:
        observed = sum(
            (Decimal(r["actual_cost_usd"]) for r in self.records if "actual_cost_usd" in r),
            Decimal(0),
        )
        return {
            "budget_usd": str(self.budget),
            "observed_cost_usd": str(observed),
            "unresolved_reserved_usd": str(self.accounted() - observed),
            "accounted_usd": str(self.accounted()),
            "pricing": self.pricing,
            "records": self.records,
            "preflights": self.quotes,
        }

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary(), indent=2) + "\n")
        temporary.replace(self.path)
