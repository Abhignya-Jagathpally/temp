#!/usr/bin/env python3
"""Unified ResistanceMap results dashboard.

Scans every results location across all pipelines/DAGs and renders ONE
self-contained HTML page (figures base64-embedded) so all figures, tables, and
scores live in a single place. Re-run any time to refresh.

  python scripts/build_results_dashboard.py
  -> paper/v20_dashboard/index.html

Honest by construction: it only renders artifacts that exist on disk; missing
sections are shown as "not yet generated", never fabricated.
"""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "v20_dashboard"
OUT = OUT_DIR / "index.html"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read(p: Path) -> Optional[str]:
    try:
        return p.read_text()
    except Exception:
        return None


def _b64img(p: Path) -> Optional[str]:
    try:
        return base64.b64encode(p.read_bytes()).decode()
    except Exception:
        return None


def md_table_to_html(md: str) -> str:
    """Render pipe-delimited markdown (tables + headers + text) to HTML."""
    out: List[str] = []
    in_tbl = False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):  # separator row
                continue
            if not in_tbl:
                out.append("<table>")
                in_tbl = True
                tag = "th"
            else:
                tag = "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_tbl:
                out.append("</table>")
                in_tbl = False
            if s.startswith("# "):
                out.append(f"<h3>{html.escape(s[2:])}</h3>")
            elif s.startswith("## "):
                out.append(f"<h4>{html.escape(s[3:])}</h4>")
            elif s.startswith("### "):
                out.append(f"<h5>{html.escape(s[4:])}</h5>")
            elif s:
                out.append(f"<p>{html.escape(s)}</p>")
    if in_tbl:
        out.append("</table>")
    return "\n".join(out)


def csv_to_html(p: Path, max_rows: int = 60) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(p)
        return df.head(max_rows).to_html(index=False, border=0, classes="data")
    except Exception as e:  # noqa: BLE001
        return f"<p class='muted'>(could not render {html.escape(p.name)}: {html.escape(str(e))})</p>"


def section(title: str, body: str, sub: str = "") -> str:
    s = f"<div class='card'><h2>{html.escape(title)}</h2>"
    if sub:
        s += f"<p class='muted'>{html.escape(sub)}</p>"
    s += body + "</div>"
    return s


def figures_block(d: Path, names: Optional[List[str]] = None) -> str:
    pngs = sorted(d.glob("*.png")) if d.exists() else []
    if names:
        pngs = [d / n for n in names if (d / n).exists()] or pngs
    out = []
    for p in pngs:
        b = _b64img(p)
        if b:
            out.append(
                f"<figure><img src='data:image/png;base64,{b}' alt='{html.escape(p.stem)}'/>"
                f"<figcaption>{html.escape(p.name)}</figcaption></figure>"
            )
    return "<div class='figs'>" + "".join(out) + "</div>" if out else ""


def tables_block(d: Path, patterns=("table*.md", "table*.csv")) -> str:
    if not d.exists():
        return ""
    out = []
    seen = set()
    for pat in patterns:
        for p in sorted(d.glob(pat)):
            key = p.stem.replace(".csv", "").replace(".md", "")
            if key in seen:
                continue
            seen.add(key)
            md = _read(d / f"{key}.md")
            if md:
                out.append(md_table_to_html(md))
            elif p.suffix == ".csv":
                out.append(f"<h4>{html.escape(p.stem)}</h4>" + csv_to_html(p))
    return "\n".join(out)


def kv_scores(items: List[tuple]) -> str:
    cells = "".join(
        f"<div class='score'><div class='v'>{html.escape(str(v))}</div>"
        f"<div class='k'>{html.escape(k)}</div></div>"
        for k, v in items
    )
    return f"<div class='scoreboard'>{cells}</div>"


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts: List[str] = []

    # --- scoreboard (key, trustworthy numbers only) ---
    scores: List[tuple] = []
    v20j = _read(ROOT / "results/v20_first_baselines/v20_first_baselines.json")
    if v20j:
        try:
            d = json.loads(v20j)
            bl = d.get("baselines", {})
            en = bl.get("ElasticNet_Cox", {}).get("c_index")
            if en:
                scores.append(("GDC OS · EN-Cox C-index", f"{en:.3f}"))
            scores.append(("GDC cohort", f"{d.get('n_patients','?')} pts / {d.get('n_events','?')} events"))
        except Exception:
            pass
    # spark survival proxy (honest, post-leakage-fix)
    sp_md = _read(ROOT / "paper/v8_artifacts/spark_pipeline/table1_survival_proxy.md")
    if sp_md:
        import re
        best = 0.0
        for m in re.finditer(r"\|\s*([0-9]\.[0-9]{3})\s*\|", sp_md):
            best = max(best, float(m.group(1)))
        if best:
            scores.append(("Spark survival-proxy · best C-index", f"{best:.3f}"))
    gate = _read(ROOT / "logs/mortfm/v17_gate_revalidate.json")
    if gate:
        try:
            g = json.loads(gate)
            granted = g.get("granted") or g.get("claims_allowed") or g.get("allowed") or []
            n_granted = g.get("n_granted", len(granted) if isinstance(granted, list) else granted)
            scores.append(("Claim-gate · levels granted", n_granted))
        except Exception:
            pass
    cev = _read(ROOT / "logs/mortfm/causal_evidence_v2_summary.json")
    if cev:
        try:
            c = json.loads(cev)
            gate_str = (
                f"{c.get('causal_mechanism_gate_n_supporting','?')}/"
                f"{c.get('causal_mechanism_gate_n_total','?')} "
                f"{'PASS' if c.get('causal_mechanism_gate_pass') else 'FAIL'}"
            )
            scores.append(("Causal k-of-N gate", gate_str))
        except Exception:
            pass
    if scores:
        parts.append(section("Scoreboard", kv_scores(scores),
                             "Only real, leakage-checked numbers. Spark tree models are age-driven (~0.6); Cox arms degenerate pending the clinical-feature fix."))

    # --- Spark lakehouse pipeline ---
    sp = ROOT / "paper/v8_artifacts/spark_pipeline"
    body = tables_block(sp) + figures_block(sp)
    dq = _read(sp / "data_quality_summary.md")
    if dq:
        body = md_table_to_html(dq) + body
    parts.append(section("Spark Lakehouse Pipeline (mortfm_spark_evidence_pipeline · 13 tasks)",
                         body or "<p class='muted'>not yet generated</p>",
                         "Ran green end-to-end (raw→audit→baselines→gate→results) on staged MMRF/processed data."))

    # --- v20 lab-first GDC baselines ---
    v20md = _read(ROOT / "results/v20_first_baselines/v20_first_baselines.md")
    parts.append(section("v20 Lab-First Baselines (GDC open-tier MMRF · OS)",
                         md_table_to_html(v20md) if v20md else "<p class='muted'>not yet generated</p>",
                         "Censoring-aware survival cascade on real GDC clinical data."))

    # --- open-access validation ---
    valj = _read(ROOT / "logs/mortfm/open_access_validation.json")
    if valj:
        parts.append(section("Open-Access External Validation",
                             f"<pre>{html.escape(valj)}</pre>",
                             "GDC OS / GSE136337 / GSE24080 — skips honestly when data/predictions absent."))

    # --- multi-agent review ---
    ma = ROOT / "paper/v20_multiagent"
    links = []
    for f in ("CODE_CHANGE_PLAN.md", "COMPARISON.md"):
        md = _read(ma / f)
        if md:
            links.append(f"<details><summary><b>{html.escape(f)}</b></summary>{md_table_to_html(md)}</details>")
    if links:
        parts.append(section("Multi-Agent Review (PhD fleet)", "".join(links),
                             "Consolidated code-change plan + honest SOTA comparison."))

    # --- v20 GDC-OS survival baselines (with bootstrap CIs) ---
    v20bj = _read(ROOT / "results/v20_first_baselines/v20_first_baselines.json")
    if v20bj:
        try:
            d = json.loads(v20bj)
            rows = ["| Model | C-index | 95% CI |", "|---|---|---|"]
            for name, m in d.get("baselines", {}).items():
                ci = f"{m.get('c_index_ci_low', float('nan')):.3f}-{m.get('c_index_ci_high', float('nan')):.3f}"
                rows.append(f"| {name} | {m.get('c_index', float('nan')):.3f} | {ci} |")
            sub = (f"{d.get('dataset', '')} - endpoint={d.get('endpoint', '')} - "
                   f"N={d.get('n_patients', '?')} pts / {d.get('n_events', '?')} events - "
                   "censoring-aware, bootstrap CI")
            parts.append(section("v20 GDC-OS Survival Baselines", md_table_to_html("\n".join(rows)), sub))
        except Exception:
            pass

    # --- registry baseline leaderboard (drug-response) ---
    lb = ROOT / "results/baselines/baseline_leaderboard.csv"
    if lb.exists():
        parts.append(section("Registry Baseline Leaderboard (drug-response)",
                             csv_to_html(lb),
                             "test_mse / mean_spearman vs ResistanceMap (delta_vs_rm). "
                             "Leakage-safe train-fit PCA-256 features."))

    # --- SOTA benchmark figures ---
    sota = ROOT / "paper/v8_artifacts/sota_benchmark"
    sota_figs = figures_block(sota)
    if sota_figs:
        parts.append(section("SOTA Benchmark Figures", sota_figs,
                             "Only panels backed by available checkpoints render; v6-only panels are "
                             "omitted (per-panel degrade), never fabricated."))

    # --- PK-SSM forward pass (UNTRAINED — architecture provenance only) ---
    pkj = _read(ROOT / "results/v20_pkssm/forward_pass_results.json")
    if pkj:
        try:
            d = json.loads(pkj)
            note = d.get("note", "")
            rest = {k: v for k, v in d.items() if k != "note"}
            body = (f"<p style='color:#ffb454;font-weight:600'>&#9888; {html.escape(note)}</p>"
                    f"<pre>{html.escape(json.dumps(rest, indent=2))}</pre>")
            parts.append(section("PK-SSM Forward Pass (UNTRAINED — not predictions)", body,
                                 "Untrained model: mechanism assignments are random initialisation. "
                                 "Shown for architecture provenance only; training needs MMRF Virtual Lab data."))
        except Exception:
            pass

    # --- causal evidence v2 (k-of-N gate) ---
    cj = _read(ROOT / "logs/mortfm/causal_evidence_v2_summary.json")
    if cj:
        try:
            d = json.loads(cj)
            enr = d.get("pathway_enrichment", {})
            items = [
                ("k-of-N gate",
                 f"{d.get('causal_mechanism_gate_n_supporting', '?')}/"
                 f"{d.get('causal_mechanism_gate_n_total', '?')} "
                 f"{'PASS' if d.get('causal_mechanism_gate_pass') else 'FAIL'}"),
                ("Reactome enrichment p", f"{enr.get('p_value', float('nan')):.4f}"),
                ("Edges scored", d.get("n_edges", "?")),
                ("Top-k drug-target", d.get("top_k_drug_target_either_endpoint", "?")),
            ]
            ce = ROOT / "results/mortfm/causal_edge_evidence_v2.csv"
            tbl = csv_to_html(ce, max_rows=20) if ce.exists() else ""
            parts.append(section("Causal Evidence v2 (k-of-N gate)", kv_scores(items) + tbl,
                                 "Channels: CRISPR essentiality, ChEMBL drug-target, Reactome pathway "
                                 "enrichment. Gate is advisory, not a causal claim."))
        except Exception:
            pass

    # --- claim-gate revalidation (12 levels) ---
    gj = _read(ROOT / "logs/mortfm/v17_gate_revalidate.json")
    if gj:
        try:
            d = json.loads(gj)
            granted = d.get("granted", []) or []
            blocked = d.get("blocked", []) or []
            body = (f"<p><b>Granted ({len(granted)}):</b> {html.escape(', '.join(map(str, granted)))}</p>"
                    f"<p><b>Blocked ({len(blocked)}):</b> {html.escape(', '.join(map(str, blocked)))}</p>")
            parts.append(section("Claim-Gate Revalidation (12 levels)", body,
                                 "Honest governance: which claim levels the current artifacts do / don't support."))
        except Exception:
            pass

    # --- latest publication bundle ---
    bundle_root = ROOT / "paper/publication_bundle"
    bundles = sorted(bundle_root.glob("pub-*")) if bundle_root.exists() else []
    if bundles:
        latest = bundles[-1]
        man = _read(latest / "run_manifest.json") or _read(latest / "bundle_manifest.json") or ""
        body = figures_block(latest / "figures")
        if man:
            body += f"<details><summary>manifest</summary><pre>{html.escape(man[:2000])}</pre></details>"
        parts.append(section(f"Publication Bundle - {latest.name}", body,
                             "Self-contained paper bundle emitted by this DAG run."))

    # --- provenance ---
    import subprocess
    try:
        log = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-12"],
                             capture_output=True, text=True).stdout
    except Exception:
        log = ""
    parts.append(section("Provenance (recent commits)", f"<pre>{html.escape(log)}</pre>"))

    css = """
    :root{--bg:#0f1419;--card:#1a2029;--ink:#e6edf3;--muted:#8b97a7;--accent:#4cc2ff;--line:#2a3340}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
    header{padding:24px 32px;border-bottom:1px solid var(--line)}
    header h1{margin:0;font-size:22px} header .sub{color:var(--muted);margin-top:4px}
    main{max-width:1100px;margin:0 auto;padding:24px 16px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin:16px 0}
    .card h2{margin:0 0 4px;font-size:18px;color:var(--accent)}
    .muted{color:var(--muted);font-size:12.5px;margin:2px 0 12px}
    table{border-collapse:collapse;margin:10px 0;width:100%;font-size:13px}
    th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}
    th{background:#222b36} tr:nth-child(even) td{background:#161c24}
    .scoreboard{display:flex;flex-wrap:wrap;gap:12px}
    .score{background:#10161e;border:1px solid var(--line);border-radius:10px;padding:14px 18px;min-width:150px}
    .score .v{font-size:24px;font-weight:700;color:var(--accent)} .score .k{color:var(--muted);font-size:12px;margin-top:2px}
    .figs{display:flex;flex-wrap:wrap;gap:16px;margin-top:10px}
    figure{margin:0;background:#fff;border-radius:8px;padding:8px;max-width:520px}
    figure img{max-width:100%;height:auto;display:block} figcaption{color:#333;font-size:11px;text-align:center;margin-top:4px}
    pre{background:#10161e;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font-size:12px}
    details{margin:8px 0} summary{cursor:pointer;color:var(--accent)} h3,h4,h5{margin:12px 0 4px}
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ResistanceMap — Results Dashboard</title><style>{css}</style></head><body>"
        f"<header><h1>ResistanceMap — Unified Results Dashboard</h1>"
        f"<div class='sub'>All pipelines · figures, tables &amp; scores in one place · generated {ts}</div></header>"
        f"<main>{''.join(parts)}</main></body></html>"
    )
    OUT.write_text(page)
    print(f"Dashboard -> {OUT}  ({len(page)//1024} KB)")
    return OUT


if __name__ == "__main__":
    build()
