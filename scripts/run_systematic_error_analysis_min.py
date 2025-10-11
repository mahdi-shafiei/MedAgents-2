#!/usr/bin/env python3
from __future__ import annotations

"""Minimal, self-contained LLM-judge error analysis.

Outputs in --out_dir:
  - counts_by_run.json
  - one_example_per_category.json
  - one_example_per_category.txt
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


CAT_MAP = {
    "ID": "Ignored Correct View",
    "OC": "Overconfident Misleading",
    "ME": "Misleading Evidence",
    "FR": "Faulty Reasoning",
    "PT": "Premature Termination",
}
CAT_CODES = list(CAT_MAP.keys())
SUCCESS_CODE = "OK"


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def find_runs(root: Path, run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for logs_path in root.rglob("logs.json"):
        model_dir = logs_path.parent
        if not (model_dir / "summary.json").exists():
            continue
        rel = model_dir.relative_to(root)
        parts = list(rel.parts)
        # Enforce path filter: only medagents/new/run_*/<model>/
        run_positions = [i for i, p in enumerate(parts) if p.startswith("run_")]
        if not run_positions:
            continue
        run_idx = run_positions[-1]
        if run_idx < 2:
            continue
        if not (parts[run_idx - 2] == "medagents" and parts[run_idx - 1] == "new"):
            continue

        dataset = parts[0] if parts else "unknown_dataset"
        model = parts[-1]
        run = parts[run_idx]
        if run_id is not None and run != f"run_{run_id}":
            continue
        runs.append({"dataset": dataset, "run": run, "model": model, "path": model_dir})
    return runs


def get_client() -> OpenAI:
    return OpenAI(base_url=os.getenv("OPENAI_ENDPOINT"), api_key=os.getenv("OPENAI_API_KEY"))


def clip(s: Any, n: int) -> Any:
    return s[:n] + "…" if isinstance(s, str) and len(s) > n else s


def build_payload(srec: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """Build the JSON payload for the LLM judge.

    Per request, include ALL information from the summary record. We attach the
    full `srec` under `summary` and also include the raw `case` from logs under
    `logs_case` for completeness. This payload is also the exact structure we
    save as the example input for parity between LLM input and saved examples.
    """
    # Provide concise helper signals without dropping any summary info
    rounds = case.get("rounds") or []
    last_fb = (rounds[-1].get("orchestrator_feedback") or {}) if rounds else {}

    # Attempt a light-weight dissent signal from summary experts only
    answers = []
    for e in (srec.get("expert_details") or [])[:8]:
        if isinstance(e, dict) and "response" in e:
            ans = (e.get("response") or {}).get("answer")
        else:
            ans = (e or {}).get("answer")
        if ans is not None:
            answers.append(ans)

    signals = {
        "dissent_present": len(set([str(a) for a in answers])) > 1 if answers else False,
        "rounds_count": len(rounds),
        "should_continue_last": last_fb.get("should_continue"),
    }

    return {
        "summary": srec,          # full summary record (unclipped)
        "logs_case": case,        # full logs record for the case
        "signals": signals,       # minimal helper signals
    }


def judge_case(client: OpenAI, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    sys = (
        "You are an auditor of medical QA debates. Determine correctness and label process errors.\n\n"
        "Input JSON fields:\n"
        "- summary: Full summary record with keys like question, options, answer_idx (ground truth), final_answer, expert_details (per-expert responses/evidence).\n"
        "- logs_case: Full logs for the case, including rounds[*].expert_results and rounds[*].orchestrator_feedback.\n"
        "- signals: Minimal helper metadata (may be sparse).\n\n"
        "Use summary.answer_idx as ground truth and summary.final_answer as the debate outcome.\n"
        "Error Categories (codes; multiple allowed if clearly present):\n"
        "- ID (Ignored Correct View): Some experts' accurate argument is overlooked, leading the discussion astray.\n"
        "- OC (Overconfident Misleading): An incorrect claim is asserted with excessive confidence, influencing the outcome.\n"
        "- ME (Misleading Evidence): Retrieved or cited evidence is inaccurate or distractive, causing reasoning errors.\n"
        "- FR (Faulty Reasoning): Relevant evidence was found, but the agent failed to interpret or utilize it correctly.\n"
        "- PT (Premature Termination): The debate ends while key disagreements or unresolved points remain.\n\n"
        "If the final answer is correct, set correct=true and you may omit the category.\n"
        "Choose exactly ONE primary category when incorrect — the best-fitting cause.\n"
        "Return JSON only: {correct: bool, category: code, rationale: string}."
    )
    # Pass the entire payload so the judge has the full summary and logs
    user = payload
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user)}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content or "{}")
    # Prefer single 'category'; fall back to first of 'categories' if present
    cat = raw.get("category")
    if not cat:
        cats = raw.get("categories") or []
        if isinstance(cats, list) and cats:
            cat = cats[0]
    cat = cat if cat in CAT_CODES else None
    return {"correct": bool(raw.get("correct")), "category": cat, "rationale": (raw.get("rationale") or "")}


def opt_text(options: Any, idx: Any) -> str:
    try:
        if isinstance(options, dict):
            return str(options.get(str(idx), options.get(idx, idx)))
        if isinstance(options, list) and isinstance(idx, int) and 0 <= idx < len(options):
            return str(options[idx])
    except Exception:
        pass
    return str(idx)


def build_trace(srec: Dict[str, Any], case: Dict[str, Any]) -> str:
    parts: List[str] = []
    q = srec.get("question", "")
    options = srec.get("options") or {}
    correct_idx = srec.get("answer_idx")
    final = srec.get("final_answer")
    parts.append(f"Question: {q}")
    if isinstance(options, dict):
        for k, v in options.items():
            parts.append(f"Option {k}: {v}")
    elif isinstance(options, list):
        for i, v in enumerate(options):
            parts.append(f"Option {i}: {v}")
    parts.append(f"Ground truth: {correct_idx} ({opt_text(options, correct_idx)})")
    parts.append(f"Moderator final: {final} ({opt_text(options, final)})")

    rounds = (case or {}).get("rounds") or []
    for r_i, rd in enumerate(rounds, 1):
        parts.append(f"Round {r_i}:")
        for e_i, res in enumerate(rd.get("expert_results") or [], 1):
            r = (res.get("response") or {})
            ans = r.get("answer")
            conf = r.get("confidence")
            just = clip(r.get("justification"), 300)
            if ans is not None:
                parts.append(f"  Expert {e_i}: answer={ans} ({opt_text(options, ans)}), conf={conf}")
            if just:
                parts.append(f"    justification: {just}")
        fb = (rd.get("orchestrator_feedback") or {})
        if fb:
            parts.append(f"  orchestrator: should_continue={fb.get('should_continue')}")
    return "\n".join(parts)


def case_has_dissent(srec: Dict[str, Any], case: Dict[str, Any]) -> bool:
    """Detect if the case exhibits answer diversity across experts/rounds."""
    seen = set()
    # From summary expert_details
    for ed in (srec.get("expert_details") or []):
        if isinstance(ed, dict) and "response" in ed:
            ans = (ed.get("response") or {}).get("answer")
        else:
            ans = (ed or {}).get("answer")
        if ans is not None:
            seen.add(str(ans))
    # From logs rounds expert_results
    for rd in (case.get("rounds") or []):
        for res in (rd.get("expert_results") or []):
            ans = ((res.get("response") or {}).get("answer"))
            if ans is not None:
                seen.add(str(ans))
    return len(seen) >= 2


def _judge_worker(model: str, srec: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """Thread worker: build payload and get LLM judgment.

    Returns a dict with both the `result` from the judge and the `payload`
    used, so that saved examples can exactly mirror the LLM input format.
    """
    try:
        payload = build_payload(srec, case)
        client = get_client()
        result = judge_case(client, model, payload)
        return {"result": result, "payload": payload}
    except Exception as e:
        return {"error": str(e)}


def write_partial_outputs(out: Path, counts_by_run: Dict[str, Counter], picked: Dict[str, Dict[str, Any]]):
    """Persist current run-specific counts and example selections after each case.

    Writes counts_by_run.json, one_example_per_category.json, and .txt atomically.
    """
    # counts per run
    counts_out = {
        run_key: {c: ctr.get(c, 0) for c in CAT_CODES}
        for run_key, ctr in counts_by_run.items()
    }
    tmp = out / "counts_by_run.json.tmp"
    tmp.write_text(json.dumps(counts_out, indent=2) + "\n")
    os.replace(tmp, out / "counts_by_run.json")

    # examples in current state (one per error category) + success last when present
    ordered = [c for c in CAT_CODES if c in picked]
    if SUCCESS_CODE in picked:
        ordered.append(SUCCESS_CODE)
    examples = [picked[c] for c in ordered]

    tmp_json = out / "one_example_per_category.json.tmp"
    tmp_json.write_text(json.dumps(examples, indent=2) + "\n")
    os.replace(tmp_json, out / "one_example_per_category.json")

    lines: List[str] = []
    for ex in examples:
        meta = ex.get("meta", {})
        inp = ex.get("input", {})
        summary = inp.get("summary", {})
        logs_case = inp.get("logs_case", {})

        lines.append(f"== {ex['category']} ==")
        lines.append(
            f"case: {meta.get('dataset')}/{meta.get('run')}/{meta.get('model')} (idx={meta.get('idx')})"
        )
        if summary.get("question"):
            lines.append("Question: " + str(summary.get("question")))
        judge = ex.get("judge") or {}
        if judge.get("rationale"):
            lines.append("Rationale: " + str(judge.get("rationale")))

        # Rebuild trace from current input payload
        try:
            trace = build_trace(summary, logs_case)
        except Exception:
            trace = ""
        lines.append("-- Full Trace --")
        lines.append(trace)

        # Moderator notes from logs_case
        try:
            rounds = (logs_case or {}).get("rounds") or []
            mod_notes = {"round_summaries": [], "areas_of_agreement": [], "areas_of_disagreement": []}
            for rd in rounds:
                ofb = (rd.get("orchestrator_feedback") or {})
                if ofb.get("round_summary"):
                    mod_notes["round_summaries"].append(ofb.get("round_summary"))
                if ofb.get("areas_of_agreement"):
                    mod_notes["areas_of_agreement"].extend(ofb.get("areas_of_agreement") or [])
                if ofb.get("areas_of_disagreement"):
                    mod_notes["areas_of_disagreement"].extend(ofb.get("areas_of_disagreement") or [])
            if any(mod_notes.values()):
                lines.append("-- Moderator Notes --")
                for k, v in mod_notes.items():
                    if isinstance(v, list):
                        for item in v:
                            lines.append(f"{k}: {item}")
                    else:
                        lines.append(f"{k}: {v}")
        except Exception:
            pass

        # Retrieved documents from summary + logs_case evidences
        try:
            docs = []
            for ed in (summary.get("expert_details") or []):
                if isinstance(ed, dict) and "response" in ed:
                    evs = (ed.get("response") or {}).get("evidences")
                else:
                    evs = (ed or {}).get("evidences")
                if isinstance(evs, list):
                    docs.extend(evs)
            for rd in (logs_case.get("rounds") or []):
                for res in (rd.get("expert_results") or []):
                    evs = ((res.get("response") or {}).get("evidences"))
                    if isinstance(evs, list):
                        docs.extend(evs)
            seen = set()
            uniq_docs = []
            for d in docs:
                key = json.dumps(d, sort_keys=True) if isinstance(d, (dict, list)) else str(d)
                if key in seen:
                    continue
                seen.add(key)
                uniq_docs.append(d)
            if uniq_docs:
                lines.append("-- Retrieved Documents (samples) --")
                for d in uniq_docs[:10]:
                    lines.append(f"doc: {str(d)[:500]}")
        except Exception:
            pass

        lines.append("")
    tmp_txt = out / "one_example_per_category.txt.tmp"
    tmp_txt.write_text("\n".join(lines) + "\n")
    os.replace(tmp_txt, out / "one_example_per_category.txt")


def main():
    ap = argparse.ArgumentParser(description="Minimal systematic error analysis")
    ap.add_argument("--root_dir", default="output", help="Root outputs directory")
    ap.add_argument("--out_dir", default="error_analysis_min", help="Where to write outputs")
    ap.add_argument("--use_llm_judge", action="store_true", help="Use LLM judge (required)")
    ap.add_argument("--judge_model", default="gpt-4o-mini", help="Judge model (chat.completions)")
    ap.add_argument("--limit", type=int, default=None, help="Optional max cases per run")
    ap.add_argument("--workers", type=int, default=8, help="Thread workers for judging")
    ap.add_argument("--run_id", type=int, default=None, help="Analyze a specific run id only (e.g., 0 for run_0)")
    args = ap.parse_args()

    if not args.use_llm_judge:
        raise SystemExit("Pass --use_llm_judge to enable LLM-based categorization.")

    load_dotenv()
    root = Path(args.root_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = find_runs(root, run_id=args.run_id)
    if not runs:
        print("No runs found under", root)
        return
    print(f"Found {len(runs)} runs")

    counts_by_run: Dict[str, Counter] = defaultdict(Counter)
    picked: Dict[str, Dict[str, Any]] = {}

    for run in tqdm(runs, desc="Runs", unit="run"):
        logs = load_json(run["path"] / "logs.json")
        summary = load_json(run["path"] / "summary.json")
        idx2sum = {str(rec.get("realidx")): rec for rec in summary}

        # Determine which case indices to process for this run: skip correct ones
        case_indices = []
        for idx in logs.keys():
            if idx not in idx2sum:
                continue
            srec = idx2sum[idx]
            if srec.get("answer_idx") == srec.get("final_answer"):
                # Collect a single representative correct example (no judging)
                if SUCCESS_CODE not in picked:
                    payload = build_payload(srec, logs[idx])
                    picked[SUCCESS_CODE] = {
                        "category_code": SUCCESS_CODE,
                        "category": "Success (Correct Outcome)",
                        "meta": {
                            "dataset": run["dataset"],
                            "run": run["run"],
                            "model": run["model"],
                            "idx": int(idx),
                        },
                        "judge": {"correct": True, "category": None, "rationale": ""},
                        "input": payload,
                    }
                continue
            case_indices.append(idx)
        if args.limit is not None:
            case_indices = case_indices[: args.limit]

        # Dispatch parallel judging for this run
        results_by_idx: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {ex.submit(_judge_worker, args.judge_model, idx2sum[idx], logs[idx]): idx for idx in case_indices}
            with tqdm(total=len(futures), desc=f"{run['dataset']}/{run['model']}", unit="case") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {"error": str(e)}
                    results_by_idx[idx] = res
                    pbar.update(1)

        # Merge results in deterministic order
        run_key = f"{run['dataset']}/{run['run']}/{run['model']}"
        for idx in case_indices:
            srec = idx2sum[idx]
            case = logs[idx]
            result_pack = results_by_idx.get(idx, {})
            result = result_pack.get("result") if isinstance(result_pack, dict) else None
            # If worker failed, keep shape predictable
            if result is None and isinstance(result_pack, dict) and "error" in result_pack:
                result = {"error": result_pack.get("error")}
            payload = result_pack.get("payload") if isinstance(result_pack, dict) else None
            if payload is None:
                # Fallback to building payload here
                payload = build_payload(srec, case)

            # Single primary category per incorrect case
            code = (result or {}).get("category")
            if code and code in CAT_CODES:
                counts_by_run[run_key][code] += 1

                # Build example using the exact LLM input format under `input`
                # and attach judge output plus minimal run metadata.
                new_example = {
                    "category_code": code,
                    "category": CAT_MAP[code],
                    "meta": {
                        "dataset": run["dataset"],
                        "run": run["run"],
                        "model": run["model"],
                        "idx": int(idx),
                    },
                    "judge": result or {},
                    "input": payload,  # exact LLM input format (summary+logs_case+signals)
                }
                # Prefer examples that show dissent (answer diversity)
                has_diversity = case_has_dissent(srec, case)
                prev = picked.get(code)
                if prev is None or (isinstance(prev, dict) and not prev.get("meta", {}).get("has_dissent") and has_diversity):
                    # Store the diversity signal in meta for tie-breaking transparency
                    new_example.setdefault("meta", {})["has_dissent"] = has_diversity
                    picked[code] = new_example
            # Persist incremental outputs after each case
            write_partial_outputs(out, counts_by_run, picked)

    # Final write of all outputs (same as partial writer for consistency)
    write_partial_outputs(out, counts_by_run, picked)

    print("Done. Wrote outputs to", out)


if __name__ == '__main__':
    main()
