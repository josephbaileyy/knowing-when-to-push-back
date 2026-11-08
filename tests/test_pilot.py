import json

import pytest

from stopstudy.io import read_json
from stopstudy.pilot import (
    CONDITIONS,
    CONFLICTS,
    ROOT,
    assign_branches,
    export,
    initial_messages,
    initial_rows,
    load_families,
    run_branches,
    run_initial,
    verified_rows,
)


class ScriptedModel:
    def __init__(self, _config, budget):
        self.budget = budget

    def query(self, _messages, label, _remaining):
        family, *parts, index = label.split("/")
        self.budget.settle(self.budget.reserve(0.0001, label), {"cost": 0.0001})
        if parts == ["leading_question"]:
            return {"content": "Here is the tradeoff."}, {"finish_reason": "stop"}
        if family == "school-computer-choice" and index == "0":
            if parts == ["direct_command"]:
                return self.tool_call("add_to_cart", "MACBOOK_PRO_14")
            if parts == ["direct_command", "counter_framing"]:
                return self.tool_call("remove_from_cart", "MACBOOK_PRO_14")
        return {"content": "Scripted reply."}, {"finish_reason": "stop"}

    def tool_call(self, name, item):
        return {"tool_calls": [{
            "id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps({"item": item})},
        }]}, {"finish_reason": "tool_calls"}


def test_conditional_branches_share_authentic_checkpoints_and_revert_cart(tmp_path):
    families = load_families(ROOT)
    saved = tmp_path / "run"
    run_initial(ROOT, saved, model_factory=ScriptedModel)
    initial = list(initial_rows(saved, families))
    assert len(initial) == 35
    for family in families:
        records = [initial_messages(family, condition)[0] for condition in CONDITIONS]
        assert records[:4] == [records[0]] * 4
        assert records[-1] != records[0]
        schemas = [episode["tools"] for f, _, episode in initial if f == family]
        assert schemas == [schemas[0]] * 5

    ratings = {"reviewer": "Scripted test", "conflicts": {
        f"{family['id']}/{condition}": {
            "tradeoff": "Here is the tradeoff." if condition == "leading_question" else None,
            "endorsement": None,
        }
        for family in families for condition in CONFLICTS
    }}
    run_branches(saved, ratings, model_factory=ScriptedModel)
    assert len(verified_rows(saved)) == 63
    assert len(read_json(saved / "budget.json")["entries"]) == 65
    review = tmp_path / "review"
    export(saved, review)
    actions = read_json(review / "actions.json")
    route = read_json(saved / "routing.json")["conflicts"]
    assert set(route["subscription-horizon/leading_question"]["branches"]) == {
        "informed_override", "uninformed_insistence"
    }
    assert route["subscription-horizon/direct_command"]["wording"] == "hypothetical"
    cart = "school-computer-choice/direct_command"
    assert route[cart]["wording"] == "specified"
    counter = actions[f"{cart}/counter_framing"]
    rationalization = actions[f"{cart}/post_action_rationalization"]
    assert counter["prior"] == rationalization["prior"] == ["MACBOOK_PRO_14"]
    assert counter["final"] == []
    assert rationalization["final"] == ["MACBOOK_PRO_14"]

    checkpoint = saved / "initial" / "file-dependency" / "leading_question.json"
    checkpoint.write_text(checkpoint.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hashes"):
        verified_rows(saved)


def test_a_warning_after_any_tool_attempt_does_not_route_as_pushback():
    family = load_families(ROOT)[0]
    episode = {"responses": [{"message": {"content": "Annual costs more for the sprint."}}],
               "events": [{"call": {"function": {"name": "subscribe_plan"}},
                           "feedback": {"error": "invalid tier"}}]}
    rating = {"tradeoff": "Annual costs more for the sprint.", "endorsement": None}
    route = assign_branches(family, episode, rating)
    assert not route["pushed_back"]
    assert not route["requested_action_applied"]
    assert route["wording"] == "hypothetical"
    with pytest.raises(ValueError, match="verbatim assistant evidence"):
        assign_branches(family, episode, rating | {"tradeoff": "An invented warning."})
