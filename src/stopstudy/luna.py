"""Run the pilot on isolated GPT-6 Luna sessions with Codex dynamic tools."""

import argparse
import json
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace

from .io import now, read_json, write_json
from .pilot import (
    CONDITIONS,
    CONFLICTS,
    DEPS,
    ROOT,
    branch_hashes,
    branch_path,
    canonical_hash,
    checkpoint,
    execute_turn,
    export,
    initial_hashes,
    initial_messages,
    initial_path,
    initial_rows,
    load_families,
    make_routing,
    restore,
    verified_rows,
)

MODEL = "gpt-6-luna"
EFFORT = "low"
SERVER = [
    "codex", "app-server", "--disable", "plugins", "--disable", "shell_tool",
    "--disable", "multi_agent", "--disable", "apps", "--disable", "skill_search",
    "--enable", "skip_host_skill_discovery", "-c", 'web_search="disabled"',
]


def server_command():
    user_config = Path.home() / ".codex" / "config.toml"
    servers = tomllib.loads(user_config.read_text()).get("mcp_servers", {}) if user_config.exists() else {}
    command = list(SERVER)
    # An empty table override does not remove individually configured servers.
    for name in servers:
        command.extend(("-c", f"mcp_servers.{name}.enabled=false"))
    return command


def history_items(messages):
    """Translate the visible checkpoint to Codex's Responses history format."""
    items = []
    for index, message in enumerate(messages):
        role = message["role"]
        content = message.get("content")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": [{"type": "input_text", "text": content}]})
            continue
        if content:
            item = {"type": "message", "role": role, "content": [{
                "type": "output_text" if role == "assistant" else "input_text", "text": content,
            }]}
            if role == "assistant":
                item["phase"] = "final_answer" if index == len(messages) - 1 else "commentary"
            items.append(item)
        for call in message.get("tool_calls", []):
            items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
    return items


class LunaSession:
    def __init__(self, server, cwd):
        self.server, self.cwd = server, cwd
        self.serial = 0
        self.pending = None
        # run_turn measures dollar costs. Codex does not supply per-request dollar costs.
        self.budget = SimpleNamespace(spent=0)
        self.request("initialize", {
            "clientInfo": {"name": "cs329z-luna-pilot", "version": "1.0"},
            "capabilities": {"experimentalApi": True},
        })
        self.send({"method": "initialized"})

    def send(self, message):
        self.server.stdin.write(json.dumps(message) + "\n")
        self.server.stdin.flush()

    def read(self):
        line = self.server.stdout.readline()
        if not line:
            raise RuntimeError("Codex closed the experiment connection")
        return json.loads(line)

    def request(self, method, params, *, wait=True):
        self.serial += 1
        serial = self.serial
        self.send({"id": serial, "method": method, "params": params})
        if not wait:
            return
        while True:
            response = self.read()
            if response.get("id") == serial and "method" not in response:
                if "error" in response:
                    raise RuntimeError(f"{method}: {response['error']['message']}")
                return response["result"]

    def start_turn(self, messages):
        specs = [tool["function"] for tool in self.episode_tools]
        result = self.request("thread/start", {
            "model": MODEL, "allowProviderModelFallback": False, "cwd": self.cwd,
            "ephemeral": True, "environments": [], "baseInstructions": messages[0]["content"],
            "developerInstructions": "", "config": {"model_reasoning_effort": EFFORT},
            "dynamicTools": [{"type": "function", "name": tool["name"],
                              "description": tool["description"], "inputSchema": tool["parameters"]}
                             for tool in specs],
        })
        if result["model"] != MODEL or result["modelProvider"] != "openai":
            raise RuntimeError("Codex did not select the requested Luna model and provider")
        self.thread_id = result["thread"]["id"]
        self.usage = None
        self.pending = None
        self.discovery = []
        if messages[1:-1]:
            self.request("thread/inject_items", {
                "threadId": self.thread_id, "items": history_items(messages[1:-1]),
            })
        self.request("turn/start", {
            "threadId": self.thread_id, "effort": EFFORT,
            "input": [{"type": "text", "text": messages[-1]["content"], "text_elements": []}],
        }, wait=False)

    def query(self, messages, label, _remaining):
        try:
            return self.advance(messages, label)
        except Exception as exc:
            self.last_error = str(exc)
            raise

    def advance(self, messages, label):
        if label.endswith("/0"):
            self.start_turn(messages)
        else:
            receipt = messages[-1]["content"]
            payload = json.loads(receipt)
            self.send({"id": self.pending, "result": {
                "contentItems": [{"type": "inputText", "text": receipt}],
                "success": "error" not in payload and payload.get("status") != "not_in_cart",
            }})
            self.pending = None

        text = []
        while True:
            message = self.read()
            method = message.get("method")
            params = message.get("params", {})
            if "error" in message:
                raise RuntimeError(message["error"]["message"])
            if params.get("threadId", self.thread_id) != self.thread_id:
                continue
            if method == "thread/tokenUsage/updated":
                self.usage = params["tokenUsage"]
            elif method == "model/rerouted":
                raise RuntimeError("Codex rerouted the requested Luna model")
            elif method == "item/tool/call":
                self.pending = message["id"]
                return {"content": "\n\n".join(text) or None, "tool_calls": [{
                    "type": "function", "id": params["callId"], "function": {
                        "name": params["tool"], "arguments": json.dumps(params["arguments"]),
                    },
                }]}, {"finish_reason": "tool_calls", "model": MODEL, "provider": "openai-codex"}
            elif method == "item/completed":
                item = params["item"]
                if item["type"] == "agentMessage":
                    text.append(item["text"])
                elif item["type"] == "mcpToolCall" and item["server"] == "codex" and item["tool"] in {
                    "list_mcp_resources", "list_mcp_resource_templates",
                }:
                    result = item.get("result") or {}
                    contents = [json.loads(part["text"]) for part in result.get("content", [])]
                    if result.get("structuredContent"):
                        contents.append(result["structuredContent"])
                    allowed = ({"resources": []}, {"resourceTemplates": []})
                    if any(content not in allowed for content in contents):
                        raise RuntimeError("A discovery call returned nonempty or unexpected records")
                    self.discovery.append({"tool": item["tool"], "result": contents})
                elif item["type"] not in {"dynamicToolCall", "reasoning", "userMessage",
                                          "functionCallOutput"}:
                    raise RuntimeError(f"Unexpected Codex item: {item['type']} "
                                       f"({item.get('server', '')}/{item.get('tool', '')})")
            elif method == "turn/completed":
                turn = params["turn"]
                if turn["status"] != "completed":
                    raise RuntimeError(f"Codex turn failed: {turn.get('error')}")
                return {"content": "\n\n".join(text)}, {
                    "finish_reason": "stop", "model": MODEL, "provider": "openai-codex",
                    "usage": self.usage, "turn_id": turn["id"],
                }
            elif "id" in message and method:
                self.send({"id": message["id"], "error": {
                    "code": -32601, "message": "Only the fixture's domain tools are available.",
                }})


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    for name in ("luna.py", "pilot.py", *DEPS):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    for name in ("families.json", "config.json"):
        shutil.copy2(ROOT / name, source / name)
    config = read_json(source / "config.json")
    config.update(model=MODEL, reasoning_effort=EFFORT, transport="codex-app-server")
    write_json(source / "config.json", config)
    version = subprocess.check_output(["codex", "--version"], text=True).strip()
    write_json(out / "generation.json", {
        "status": "initial_running", "provenance": "openai-codex-app-server",
        "model": MODEL, "reasoning_effort": EFFORT, "codex_version": version,
        "started_at": now(), "assigned_initial": 35, "assigned_branches": 28,
    })


def run_group(out, selected, *, followups):
    source = out / "source"
    all_families = load_families(source)
    families = [family for family in all_families if family["id"] in selected]
    if len(families) != len(selected):
        raise ValueError("Expected distinct known family IDs")
    config = read_json(source / "config.json")
    routing = read_json(out / "routing.json")["conflicts"] if followups else None

    with tempfile.TemporaryDirectory(prefix="cs329z-luna-") as cwd, subprocess.Popen(
        server_command(), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as server:
        try:
            model = LunaSession(server, cwd)
            for family in families:
                index = all_families.index(family) % len(CONDITIONS)
                for condition in CONDITIONS[index:] + CONDITIONS[:index]:
                    if not followups:
                        path = initial_path(out, family, condition)
                        save_episode(path, model, family, condition,
                                     initial_messages(family, condition), {"actions": []}, config)
                    elif condition in CONFLICTS:
                        parent = read_json(initial_path(out, family, condition))
                        for branch, prompt in routing[f"{family['id']}/{condition}"]["branches"].items():
                            path = branch_path(out, family, condition, branch)
                            messages, state = restore(parent, prompt)
                            save_episode(path, model, family, f"{condition}/{branch}", messages,
                                         state, config, parent["checkpoint_hash"])
        finally:
            server.terminate()


def save_episode(path, model, family, label, messages, state, config, parent_hash=None):
    if path.exists():
        prior = read_json(path)
        if prior["status"] == "completed":
            return
        if prior["responses"] or prior["events"]:
            raise ValueError(f"Refusing to retry a turn with recorded output: {path}")
        archive = path.with_name(f"{path.stem}.failed")
        if archive.exists():
            raise ValueError(f"Turn has already been retried: {path}")
        path.rename(archive)
    model.last_error = None
    episode = execute_turn(model, family, label, messages, state, config)
    episode.pop("cost_usd")
    if model.last_error:
        episode["transport_error"] = model.last_error
    if model.discovery:
        episode["empty_codex_discovery"] = model.discovery
    if parent_hash is None:
        episode["checkpoint_hash"] = canonical_hash(checkpoint(episode))
    else:
        episode["parent_hash"] = parent_hash
    write_json(path, episode)
    print(f"{family['id']}/{label}: {episode['status']}", flush=True)
    if episode["status"] != "completed":
        raise RuntimeError(f"Codex turn incomplete. Partial trace saved to {path}. "
                           f"{model.last_error or episode['status']}")


def route(out, ratings):
    families = load_families(out / "source")
    manifest = read_json(out / "generation.json")
    initial = list(initial_rows(out, families))
    if len(initial) != manifest["assigned_initial"] or any(
        episode["status"] != "completed" for _, _, episode in initial
    ):
        raise ValueError("Expected all completed initial turns")
    write_json(out / "routing.json", make_routing(out, families, read_json(ratings)))
    manifest.update(status="branches_running", initial_hashes=initial_hashes(out))
    write_json(out / "generation.json", manifest)


def finalize(out, review):
    manifest = read_json(out / "generation.json")
    if initial_hashes(out) != manifest["initial_hashes"]:
        raise ValueError("Initial artifacts changed after routing")
    paths = [*out.glob("initial/*/*.json"), *out.glob("branches/*/*/*.json")]
    counts = {"turns": len(paths), "turns_with_token_usage": 0}
    for path in paths:
        episode = read_json(path)
        usage = episode["responses"][-1]["meta"].get("usage")
        if usage is not None:
            counts["turns_with_token_usage"] += 1
    write_json(out / "budget.json", counts | {
        "charged_usd": None, "billing": "Codex account usage; dollar costs not supplied",
    })
    manifest.update(status="complete", finished_at=now(), branch_hashes=branch_hashes(out))
    write_json(out / "generation.json", manifest)
    verified_rows(out)
    export(out, review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "initial", "route", "continue", "finalize"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run)
    elif args.phase == "route":
        route(args.run, args.ratings)
    elif args.phase == "finalize":
        finalize(args.run, args.review)
    else:
        if not args.families:
            parser.error("--families is required for initial and continue")
        run_group(args.run, args.families, followups=args.phase == "continue")


if __name__ == "__main__":
    main()
