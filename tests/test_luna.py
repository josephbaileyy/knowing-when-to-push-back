from stopstudy.luna import history_items


def test_replay_preserves_visible_assistant_text_tool_calls_and_receipts():
    messages = [
        {"role": "user", "content": "Can you book the flight?"},
        {"role": "assistant", "content": "I can book it.", "tool_calls": [{
            "type": "function", "id": "call_1", "function": {
                "name": "book_trip", "arguments": '{"mode":"FLIGHT_JFK_IAD"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"applied"}'},
        {"role": "assistant", "content": "The flight is booked."},
    ]
    history = history_items(messages)
    assert [item["type"] for item in history] == [
        "message", "message", "function_call", "function_call_output", "message"
    ]
    assert history[1]["phase"] == "commentary"
    assert history[2]["name"] == "book_trip"
    assert history[2]["call_id"] == history[3]["call_id"] == "call_1"
    assert history[3]["output"][0]["text"] == messages[2]["content"]
    assert history[-1]["phase"] == "final_answer"
