"""Run one model turn and record ordered tool calls and results."""

import copy
import json

from .io import now


def run_turn(model, case, messages, state, config, label, *, execute_tool):
    messages, state = copy.deepcopy(messages), copy.deepcopy(state)
    result = {"case_id": case["id"], "phase": label, "status": "call_cap", "started_at": now(),
              "input_messages": copy.deepcopy(messages), "input_state": copy.deepcopy(state),
              "responses": [], "events": []}
    start = model.budget.spent
    prefix = f"{case['id']}/{label}"

    def remaining():
        return config["turn_cost_cap_usd"] - (model.budget.spent - start)

    for index in range(config["max_calls"]):
        try:
            reply, metadata = model.query(
                messages, f"{prefix}/{index}",
                remaining(),
            )
        except Exception as exc:
            result.update(status="request_error", error=type(exc).__name__)
            break
        result["responses"].append({"message": reply, "meta": metadata})
        assistant = {key: reply[key] for key in ["role", "content", "tool_calls"] if key in reply}
        assistant["role"] = "assistant"
        messages.append(assistant)
        calls = reply.get("tool_calls") or []
        if not calls:
            result["status"] = "completed" if (
                reply.get("content") and metadata["finish_reason"] == "stop"
            ) else "incomplete_response"
            break
        for call in calls:
            event = {"call": call, "before": copy.deepcopy(state)}
            try:
                arguments = json.loads(call["function"]["arguments"])
                event["reply_index"] = index
                feedback, event["ok"] = execute_tool(call["function"]["name"], arguments, state)
            except (ValueError, KeyError, TypeError) as exc:
                feedback, event["ok"] = {"error": str(exc)}, False
            except Exception as exc:
                result.update(status="request_error", error=type(exc).__name__)
                feedback, event["ok"] = {"error": "Tool service failed."}, False
            event.update(after=copy.deepcopy(state), feedback=feedback)
            result["events"].append(event)
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(feedback)})
            if result["status"] == "request_error":
                break
        if result["status"] == "request_error":
            break
    result.update(final_state=state, conversation=messages, finished_at=now(),
                  cost_usd=model.budget.spent - start)
    return result
