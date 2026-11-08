"""OpenRouter client with a durable, conservative request budget."""

import json
import math
import os
import time

import httpx

from .io import now, write_json


class BudgetExceeded(RuntimeError):
    pass


class RunBudgetExceeded(BudgetExceeded):
    pass


class Budget:
    def __init__(self, limit, path):
        self.limit = float(limit)
        self.path = path
        self.entries = []

    @property
    def spent(self):
        return sum(e["charged_or_reserved_usd"] for e in self.entries)

    def save(self):
        write_json(self.path, {"limit_usd": self.limit, "charged_or_reserved_usd": self.spent,
                               "entries": self.entries})

    def reserve(self, amount, label):
        if amount < 0 or not math.isfinite(amount):
            raise ValueError("Invalid reservation")
        if self.spent + amount > self.limit:
            raise RunBudgetExceeded(f"Run budget ${self.limit:.2f} would be exceeded")
        entry = {"time": now(), "label": label, "charged_or_reserved_usd": amount,
                 "status": "reserved", "cost_source": "conservative_reservation"}
        self.entries.append(entry)
        self.save()
        return entry

    def settle(self, entry, usage):
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0:
            entry["charged_or_reserved_usd"] = float(cost)
            entry["cost_source"] = "provider_reported"
        entry["status"] = "completed"
        entry["usage"] = usage
        self.save()
        if self.spent > self.limit:
            raise RunBudgetExceeded("Provider-reported charges exceed the configured run budget")


class PilotModel:
    def __init__(self, config, budget):
        self.config, self.budget = config, budget
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise ValueError("Set OPENROUTER_API_KEY in .env (see .env.example)")

    def payload(self, messages):
        provider = {"allow_fallbacks": False, "require_parameters": True,
                    "max_price": {
                        "prompt": self.config["prompt_price_ceiling_per_million"],
                        "completion": self.config["completion_price_ceiling_per_million"],
                    }}
        if self.config.get("provider_order"):
            provider["order"] = self.config["provider_order"]
            provider["only"] = self.config["provider_order"]
        return {"model": self.config["model"], "messages": messages,
                "temperature": self.config["temperature"], "tools": self.episode_tools, "tool_choice": "auto",
                "max_tokens": self.config["max_output_tokens"], "provider": provider}

    def query(self, messages, label, phase_remaining):
        payload = self.payload(messages)
        # A conservative byte-based upper bound, including message/template overhead.
        # API max_price also constrains endpoint prices; no silent model fallback.
        input_bound = len(json.dumps(payload).encode()) + 4096
        reserve = (input_bound * self.config["prompt_price_ceiling_per_million"]
                   + self.config["max_output_tokens"]
                   * self.config["completion_price_ceiling_per_million"]) / 1_000_000
        retries = self.config.get("rate_limit_retries", 0)
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError("rate_limit_retries must be an integer from zero to two")
        starting_cost = self.budget.spent
        for attempt in range(retries + 1):
            remaining = phase_remaining - (self.budget.spent - starting_cost)
            if reserve > remaining:
                raise BudgetExceeded("Phase budget would be exceeded by next request")
            entry = self.budget.reserve(reserve, label if not attempt else f"{label}-retry{attempt}")
            minimum_interval = self.config.get("request_interval_seconds", 0)
            since_last = time.monotonic() - getattr(self, "last_request_at", -float("inf"))
            if since_last < minimum_interval:
                time.sleep(minimum_interval - since_last)
            self.last_request_at = time.monotonic()
            # Never replay ambiguous timeouts or completed replies. Only explicit HTTP429
            # rejection is eligible, and each physical request keeps its own reservation.
            with httpx.Client(timeout=180) as client:
                response = client.post("https://openrouter.ai/api/v1/chat/completions", json=payload,
                                       headers={"Authorization": f"Bearer {self.key}",
                                                "X-OpenRouter-Title": "CS329Z Knowing When to Push Back"})
            if response.status_code != 429 or attempt == retries:
                break
            entry["status"] = "http_error_429"
            self.budget.save()
            delay = self.config.get("rate_limit_backoff_seconds", 30) * (attempt + 1)
            time.sleep(delay)
        if response.status_code >= 400:
            # Avoid logging response/request headers or credentials.
            entry["status"] = f"http_error_{response.status_code}"
            self.budget.save()
            raise RuntimeError(f"OpenRouter returned HTTP {response.status_code}; "
                               "check model/provider availability and account credits")
        data = response.json()
        if "error" in data or not data.get("choices"):
            raise RuntimeError("Provider returned an error or no choices (reservation retained)")
        self.budget.settle(entry, data.get("usage", {}))
        return data["choices"][0]["message"], {
            "response_id": data.get("id"), "model": data.get("model"),
            "provider": data.get("provider"), "usage": data.get("usage", {}),
            "cost_usd": entry["charged_or_reserved_usd"], "cost_source": entry["cost_source"],
            "finish_reason": data["choices"][0].get("finish_reason"),
        }
