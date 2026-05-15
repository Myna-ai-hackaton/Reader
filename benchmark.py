#!/usr/bin/env python3
"""
Reader Agent Benchmark (OpenRouter)
====================================
Evaluates your reader agent using an LLM judge via OpenRouter.

Loads API key from .env file (OPENAI_API_KEY) or environment variables.

Usage:
  Interactive (answer questions one by one):
    python benchmark.py

  Batch (load answers from a JSON file):
    python benchmark.py --input answers.json

  Save results:
    python benchmark.py --input answers.json --output results.json

  Pick a different judge model:
    python benchmark.py --model openai/gpt-4o

  Only run specific categories:
    python benchmark.py --categories Risk Fit

  Run Reader app CLI for answers:
    python benchmark.py --source cli --github-url https://github.com/osu-crypto/libOTe/tree/master/libOTe

answers.json format:
  [
    {"id": 1, "answer": "14 PRs were merged..."},
    {"id": 11, "answer": "Alice would be best...", "context": "new ML pipeline"}
  ]
"""

import os
import sys
import json
import argparse
import textwrap
import subprocess
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency. Run:  pip install openai")
    sys.exit(1)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# ── Color helpers ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
BLUE   = "\033[34m"
PURPLE = "\033[35m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"

def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def blue(s):   return f"{BLUE}{s}{RESET}"
def purple(s): return f"{PURPLE}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"

CATEGORY_COLORS = {
    "Retrieval": blue,
    "Velocity":  green,
    "People":    purple,
    "Risk":      red,
    "Synthesis": yellow,
    "Fit":       cyan,
}

def cat_label(category: str) -> str:
    color_fn = CATEGORY_COLORS.get(category, dim)
    return color_fn(f"[{category}]")

# ── Questions ──────────────────────────────────────────────────────────────────

QUESTIONS = [
    {"id": 1,  "category": "Retrieval", "q": "How many PRs were merged in the last 2 weeks, and who authored them?"},
    {"id": 2,  "category": "Retrieval", "q": "What are the top 5 files or components that changed the most across recent PRs?"},
    {"id": 3,  "category": "Velocity",  "q": "What types of changes (features, bug fixes, refactors) has the team been working on?"},
    {"id": 4,  "category": "Velocity",  "q": "What is the overall code quality trend based on recent PR risk assessments?"},
    {"id": 5,  "category": "People",    "q": "Which developer has been most active this month, and what have they worked on?"},
    {"id": 6,  "category": "People",    "q": "Who are the top 3 most skilled developers based on their skill matrix?"},
    {"id": 7,  "category": "Risk",      "q": "Are there signs of review bottlenecks, stale PRs, or single points of failure?"},
    {"id": 8,  "category": "Risk",      "q": "What are the most common risk levels and types in recent PRs?"},
    {"id": 9,  "category": "Synthesis", "q": "Give me a one-paragraph team health summary for a stakeholder update."},
    {"id": 10, "category": "Synthesis", "q": "What are the top 3 things I should focus on as a PM this week?"},
    {
        "id": 11,
        "category": "Fit",
        "q_template": "I have a new project about {topic} — based on the team's recent contributions, who would be the best fit to lead or own it, and why?",
        "context_label": "Project topic",
        "context_placeholder": "e.g. refactoring the auth service, a new ML pipeline",
    },
]

def resolve_question(q: dict, context: str = "") -> str:
    if "q_template" in q:
        topic = context.strip() or "an unspecified topic"
        return q["q_template"].format(topic=topic)
    return q["q"]

# ── Judge ──────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a strict evaluator of an AI reader agent that answers PM questions about a software team's GitHub PR activity stored in Firebase.

Score each answer on two dimensions (0-5 each):

ACCURACY (0-5): How grounded is the answer in specific, concrete data?
0 = refuses, hallucinates, or completely vague
1 = very vague, no specific data
2 = some specifics but mostly generic
3 = reasonably specific, plausible data references
4 = specific with clear attribution
5 = highly specific, precise, verifiable

QUALITY (0-5): How useful is this for a PM making real decisions?
0 = useless or misleading
1 = very low value
2 = some value but poorly structured
3 = useful with reasonable insights
4 = clear, actionable, well-organized
5 = exceptional, immediately actionable

PASS = true if accuracy >= 3 AND quality >= 3, else false.

Respond ONLY with valid JSON (no markdown):
{"accuracy": N, "quality": N, "reasoning": "2-3 sentence critique", "pass": true|false}"""

DEFAULT_MODEL = "anthropic/claude-sonnet-4"


def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/reader-agent-benchmark",
            "X-Title": "Reader Agent Benchmark",
        },
    )


def judge_answer(client: OpenAI, model: str, question: str, answer: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\n\nAnswer: {answer}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Display ────────────────────────────────────────────────────────────────────

def bar(value: float, max_val: int = 5, width: int = 20) -> str:
    filled = round((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)

def print_result(q: dict, answer: str, result: dict, context: str = ""):
    question_text = resolve_question(q, context)
    verdict = green("✓ PASS") if result["pass"] else red("✗ FAIL")

    print()
    print(f"  {cat_label(q['category'])}  {bold(question_text)}")
    if context:
        print(f"  {dim('Topic: ' + context)}")
    print()

    acc  = result["accuracy"]
    qual = result["quality"]
    print(f"  Accuracy  {bar(acc)}  {bold(str(acc))}/5")
    print(f"  Quality   {bar(qual)}  {bold(str(qual))}/5")
    print()

    reasoning = textwrap.fill(result["reasoning"], width=70, initial_indent="  ", subsequent_indent="  ")
    print(dim(reasoning))
    print()
    print(f"  {verdict}")
    print()
    print("  " + "─" * 60)

def print_summary(results: list[dict], model: str):
    scored = [r for r in results if r.get("result")]
    if not scored:
        print("\nNo questions were evaluated.")
        return

    passed    = [r for r in scored if r["result"]["pass"]]
    pass_rate = round((len(passed) / len(scored)) * 100)
    avg_acc   = sum(r["result"]["accuracy"] for r in scored) / len(scored)
    avg_qual  = sum(r["result"]["quality"]  for r in scored) / len(scored)

    print()
    print(bold("═" * 62))
    print(bold("  Benchmark summary"))
    print(bold("═" * 62))
    print(f"  Judge model   {dim(model)}")
    print(f"  Pass rate     {bold(f'{pass_rate}%')}  ({len(passed)}/{len(scored)} questions)")
    print(f"  Avg accuracy  {bold(f'{avg_acc:.1f}/5')}  {bar(avg_acc)}")
    print(f"  Avg quality   {bold(f'{avg_qual:.1f}/5')}  {bar(avg_qual)}")
    print()

    by_category: dict[str, list] = {}
    for r in scored:
        by_category.setdefault(r["category"], []).append(r)

    print(bold("  By category:"))
    for cat, items in by_category.items():
        cat_pass = sum(1 for r in items if r["result"]["pass"])
        print(f"    {cat_label(cat):<30}  {cat_pass}/{len(items)} passed")

    print()

# ── Interactive mode ───────────────────────────────────────────────────────────

def run_cli_query(app_path: str, github_url: str, query: str, token: str | None = None) -> str:
    command = [sys.executable, app_path, "--github-url", github_url, "--query", query]
    if token:
        command += ["--token", token]

    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        raise RuntimeError(
            f"CLI query failed (exit {proc.returncode}).\n" \
            f"Stdout:\n{stdout}\nStderr:\n{stderr}"
        )

    lines = proc.stdout.splitlines()
    answer_index = next((i for i, line in enumerate(lines) if line.strip() == "ANSWER"), None)
    if answer_index is None:
        raise ValueError("Could not parse ANSWER block from app.py CLI output.")

    start = answer_index + 2
    return "\n".join(line for line in lines[start:] if line is not None).strip()


def run_cli_mode(
    client: OpenAI,
    model: str,
    github_url: str,
    app_path: str,
    github_token: str | None,
    categories: Optional[list[str]],
    fit_context: str | None,
) -> list[dict]:
    questions = QUESTIONS
    if categories:
        questions = [q for q in questions if q["category"] in categories]

    print()
    print(bold("Reader Agent Benchmark — CLI mode"))
    print(dim(f"Judge: {model}"))
    print(dim(f"App CLI: {sys.executable} {app_path} --github-url {github_url}"))
    print(dim("Running each question through the Reader CLI and judging the answers.\n"))

    results = []

    for q in questions:
        if "q_template" in q:
            context = fit_context or q.get("context_placeholder", "an unspecified topic")
        else:
            context = ""

        question = resolve_question(q, context)
        print(cat_label(q["category"]))
        print(bold(f"  {question}"))

        try:
            answer = run_cli_query(app_path, github_url, question, github_token)
            if not answer:
                print(dim("  Skipped (empty answer).\n"))
                continue
            print(dim("  Judging…"), end="", flush=True)
            result = judge_answer(client, model, question, answer)
            print("\r", end="")
            print_result(q, answer, result, context)
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": question,
                "answer": answer,
                "result": result,
            })
        except Exception as e:
            print(f"  {red('Error:')} {e}\n")

    return results


def run_interactive(client: OpenAI, model: str, categories: Optional[list[str]]) -> list[dict]:
    questions = QUESTIONS
    if categories:
        questions = [q for q in questions if q["category"] in categories]

    print()
    print(bold("Reader Agent Benchmark"))
    print(dim(f"Judge: {model}"))
    print(dim("Press Enter twice to submit. Empty answer = skip. Ctrl+C to quit.\n"))

    results = []

    for q in questions:
        is_fit = "q_template" in q
        print(cat_label(q["category"]))

        context = ""
        if is_fit:
            print(bold("  I have a new project about [topic] — who would be the best fit?"))
            context = input(f"  {q['context_label']} ({q['context_placeholder']}): ").strip()
            if not context:
                print(dim("  Skipped.\n"))
                continue
            print(f"  → {dim(resolve_question(q, context))}")
        else:
            print(bold(f"  {q['q']}"))

        print(dim("  Paste the reader's answer (empty line to finish, blank entry to skip):"))

        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "" and lines:
                break
            if line == "" and not lines:
                break
            lines.append(line)

        answer = "\n".join(lines).strip()
        if not answer:
            print(dim("  Skipped.\n"))
            continue

        print(dim("  Judging…"), end="", flush=True)
        try:
            result = judge_answer(client, model, resolve_question(q, context), answer)
            print("\r", end="")
            print_result(q, answer, result, context)
            results.append({
                "id": q["id"], "category": q["category"],
                "question": resolve_question(q, context),
                "answer": answer, "result": result,
            })
        except Exception as e:
            print(f"\r  {red('Error:')} {e}\n")

    return results

# ── Batch mode ─────────────────────────────────────────────────────────────────

def run_batch(client: OpenAI, model: str, input_path: str, categories: Optional[list[str]]) -> list[dict]:
    with open(input_path) as f:
        inputs = json.load(f)

    inputs_by_id = {item["id"]: item for item in inputs}
    questions = QUESTIONS
    if categories:
        questions = [q for q in questions if q["category"] in categories]

    print()
    print(bold("Reader Agent Benchmark"))
    print(dim(f"Judge: {model}  |  Input: {input_path}\n"))
    print("─" * 62)

    results = []

    for q in questions:
        item = inputs_by_id.get(q["id"])
        if not item or not item.get("answer", "").strip():
            print(dim(f"  [{q['id']:>2}] {q['category']:<12} — skipped (no answer)"))
            continue

        answer  = item["answer"].strip()
        context = item.get("context", "").strip()

        print(f"  [{q['id']:>2}] {cat_label(q['category']):<22} judging…", end="", flush=True)
        try:
            result = judge_answer(client, model, resolve_question(q, context), answer)
            verdict  = green("PASS") if result["pass"] else red("FAIL")
            acc_str  = f"acc={result['accuracy']}/5"
            qual_str = f"qual={result['quality']}/5"
            print(f"\r  [{q['id']:>2}] {cat_label(q['category']):<22} {verdict}  {dim(acc_str)}  {dim(qual_str)}")
            results.append({
                "id": q["id"], "category": q["category"],
                "question": resolve_question(q, context),
                "answer": answer, "result": result,
            })
        except Exception as e:
            print(f"\r  [{q['id']:>2}] {cat_label(q['category']):<22} {red('error:')} {e}")

    print()
    print("─" * 62)

    for r in results:
        q = next(x for x in QUESTIONS if x["id"] == r["id"])
        print_result(q, r["answer"], r["result"], inputs_by_id.get(r["id"], {}).get("context", ""))

    return results

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark your reader agent using an LLM judge via OpenRouter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",      "-i", help="Path to answers JSON file (batch mode)")
    parser.add_argument("--output",     "-o", help="Save results to this JSON file")
    parser.add_argument("--model",      "-m", default=DEFAULT_MODEL,
                        help=f"OpenRouter model to use as judge (default: {DEFAULT_MODEL})")
    parser.add_argument("--categories", "-c", nargs="+",
                        choices=["Retrieval", "Velocity", "People", "Risk", "Synthesis", "Fit"],
                        help="Only run these categories")
    parser.add_argument("--source", choices=["manual", "cli"], default="manual",
                        help="Choose answer source: manual entry or app.py CLI")
    parser.add_argument("--app-path", default="app.py",
                        help="Path to the Reader app CLI script")
    parser.add_argument("--github-url", help="GitHub repository/organization URL for CLI mode")
    parser.add_argument("--github-token", help="GitHub Personal Access Token for CLI mode")
    parser.add_argument("--fit-context", help="Project topic context used for 'Fit' questions in CLI mode")
    parser.add_argument("--api-key",    help="OpenRouter API key (default: OPENAI_API_KEY from .env or env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(red("Error: set OPENAI_API_KEY in .env file or pass --api-key"))
        sys.exit(1)

    client = make_client(api_key)

    try:
        if args.input:
            results = run_batch(client, args.model, args.input, args.categories)
        elif args.source == "cli":
            if not args.github_url:
                print(red("Error: --github-url is required when --source cli is used."))
                sys.exit(1)
            results = run_cli_mode(
                client=client,
                model=args.model,
                github_url=args.github_url,
                app_path=args.app_path,
                github_token=args.github_token,
                categories=args.categories,
                fit_context=args.fit_context,
            )
        else:
            results = run_interactive(client, args.model, args.categories)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)

    print_summary(results, args.model)

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {bold(args.output)}\n")

if __name__ == "__main__":
    main()
