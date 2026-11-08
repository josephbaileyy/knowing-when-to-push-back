# Knowing When to Push Back

CS 329Z: Engineering AI Agents · Joseph Bailey · Fall 2026

A tool-use evaluation of when assistants explain a tradeoff, act on a request, or
change the saved state after a follow-up. The pilot uses seven synthetic scenario
families and simulated tools. GPT-4.1 mini and GPT-6 Luna each completed 35 initial
turns and 28 conditional follow-ups.

- [Midpoint video](demo/midpoint-demo.mp4) and [interactive replay](demo/midpoint.html)
- [Captions](demo/midpoint-demo.srt) and [narration](demo/narration.txt)
- [Midway report](report/midway.pdf) and [LaTeX source](report/midway.tex)
- [GPT-4.1 mini traces](results/gpt-4.1-mini/run/) and [GPT-6 Luna traces](results/gpt-6-luna/run/)

## Replay the pilot offline

From the repository root with Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python scripts/build_demo.py --out runs/replay
```

Open `runs/replay/midpoint.html`. Use Next or the arrow keys to move through the
examples. Each chat has a summary of the connected context. Open its **Full context**
entry to inspect the system instructions, exact records, available tools, prior
messages and state before the turn.

The builder re-executes all 126 recorded turns through the simulator and checks the
receipts, dialogue and final state. It uses recorded replies without making API
calls. The public archives retain the dialogue, action trace and usage counts.
Provider response identifiers, Codex turn identifiers and unused reasoning metadata
are omitted. The included manifests verify the public copies.

## Generate fresh turns

### GPT-4.1 mini

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. The runner uses OpenRouter
with a $0.50 total cap and a $0.04 per-turn cap.

```bash
uv run python -m stopstudy.pilot validate
uv run python -m stopstudy.pilot initial --out runs/mini
# Review the initial replies and save ratings.json before continuing.
uv run python -m stopstudy.pilot continue --run runs/mini --ratings ratings.json
uv run python -m stopstudy.pilot export --run runs/mini --out runs/mini-review
```

### GPT-6 Luna

Sign into a local Codex CLI with GPT-6 Luna access. The recorded run used Codex CLI
0.166.0, low reasoning effort and App Server dynamic tools.

```bash
families=(subscription-horizon purchase-sale file-dependency airpods-vs-beats school-computer-choice nyc-to-dc-travel download-sleep-patch)
uv run python -m stopstudy.luna prepare --run runs/luna
uv run python -m stopstudy.luna initial --run runs/luna --families "${families[@]}"
# Review the initial replies and save a separate luna-ratings.json.
uv run python -m stopstudy.luna route --run runs/luna --ratings luna-ratings.json
uv run python -m stopstudy.luna continue --run runs/luna --families "${families[@]}"
uv run python -m stopstudy.luna finalize --run runs/luna --review runs/luna-review
```

### Rating and routing

Review the 14 leading-question and direct-command turns. For each, record an exact
excerpt where the assistant flags the tradeoff or endorses the requested action.
Use JSON `null` for absent evidence. See the saved
[GPT-4.1 mini ratings](results/gpt-4.1-mini/turn1-ratings.json) and
[GPT-6 Luna ratings](results/gpt-6-luna/turn1-ratings.json) for the format. Fresh runs
need ratings from their own responses.

A warning with zero domain-tool calls qualifies for an informed override and an
uninformed insistence branch. Other turns get counter-framing and verification
branches. If the requested action did not happen, the branch asks hypothetically.
Both branches restore the same visible conversation and copy the state independently.

## Build the deliverables

```bash
uv run pytest -q
uv run python scripts/render_video.py --build runs/replay
(cd report && pdflatex -interaction=nonstopmode -halt-on-error midway.tex && pdflatex -interaction=nonstopmode -halt-on-error midway.tex)
```

The video renderer needs macOS `say`, FFmpeg, Node.js and Playwright with Chromium.
If Playwright is installed outside the repository, set `PLAYWRIGHT_MODULE` to its
absolute module path. The video uses synthetic narration and includes a subtitle track.

The report uses the [ICLR 2025 LaTeX template](https://www.overleaf.com/latex/templates/template-for-iclr-2025-conference-submission/gqzkdyycxtvt).
