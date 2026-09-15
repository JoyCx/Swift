"""Markdown + self-contained HTML report (inline SVG, no external assets)."""
from __future__ import annotations
import html, json
from pathlib import Path


def _pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _f(x, d=0):
    return "n/a" if x is None else f"{x:,.{d}f}"


def markdown(run: dict) -> str:
    L = [f"# {run.get('name', 'swiftlab run')}\n"]
    L.append(f"Model: `{run.get('model', '?')}`  •  effort: `{run.get('effort', '?')}`  •  seeds/task: {run.get('seeds', '?')}  •  tasks: {run.get('n_tasks', '?')}\n")
    if "waste" in run:
        w = run["waste"]
        L.append("## 1. What the base model wastes (settle probing)\n")
        L.append(f"- traces probed: {w['n']}  •  categories: {w['categories']}")
        L.append(f"- overspent share of thinking: **{_pct(w['overspent_share'])}**; wrong-trace share: {_pct(w['wrong_share'])}; derailed: {_pct(w['derailed_share'])}")
        L.append(f"- upper bound on removable thinking: **{_pct(w['removable_share_upper_bound'])}**\n")
    if "scaling" in run:
        L.append("## 2. Dataset-size scaling (how many long traces are enough?)\n")
        L.append("| traces | overspent share | removable share | wrong | derailed | markers found | loop signatures | est. total (Chao1) | coverage |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in run["scaling"]["points"]:
            L.append(f"| {p['n']} | {_pct(p['overspent_share'])} | {_pct(p['removable_share'])} | {p['n_wrong']} | {p['n_derailed']} | {p['markers_discovered']} | {p['loop_signatures']} | {p['loop_signatures_chao1']:.1f} | {_pct(p['signature_coverage'])} |")
        L.append(f"\nRecommended dataset size: **{run['scaling']['recommended_n']}** ({run['scaling']['note']})\n")
    if "markers" in run:
        m = run["markers"]
        top = ", ".join(f"`{d['phrase']}` (z={d['z']:.1f})" for d in m["markers"][:15])
        L.append("## 3. Mined overthinking markers\n")
        L.append(f"{len(m['markers'])} phrases with z ≥ {m['z_threshold']} from {m['n_waste_docs']} waste vs {m['n_need_docs']} need segments.\n\nTop: {top}\n")
    if "bundle" in run:
        b = run["bundle"]
        L.append("## 4. Direction extraction and surgical cleaning\n")
        L.append("| layer | AUC dirty | AUC clean | energy removed | cos(capability atoms) dirty → clean |")
        L.append("|---:|---:|---:|---:|---|")
        for l, d in sorted(b["layers"].items(), key=lambda kv: int(kv[0])):
            cd = ", ".join(f"{k}:{v:+.2f}→{d['cos_atoms_clean'][k]:+.2f}" for k, v in d["cos_atoms_dirty"].items())
            L.append(f"| {l} | {d['auc_dirty']:.3f} | {d['auc_clean']:.3f} | {_pct(d['energy_removed'])} | {cd} |")
        L.append(f"\nBest layers: {b['best_layers']}\n")
    if "search" in run:
        s = run["search"]; bp = s["best"]["params"]; bm = s["best"]["metrics"]
        L.append("## 5. Edit hyper-parameter search\n")
        L.append(f"- method: {s['method']}, trials: {len(s['trials'])}")
        L.append(f"- best: γ={bp['gamma']:.2f}, kernel={bp['kernel']}, center={bp['center']:.1f}, width={bp['width']:.1f}, matrices={bp['matrices']}")
        L.append(f"- calib metrics: think ratio {bm['think_ratio']:.3f}, acc Δ {bm['acc_delta']:+.3f}, KL {bm.get('kl', 0):.4f}\n")
    if "eval" in run:
        e = run["eval"]
        L.append("## 6. Paired evaluation (same task, same seed, same context)\n")
        L.append("| arm | n | accuracy | think mean | think median | latency mean | truncated |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for n, a in e["arms"].items():
            L.append(f"| {n} | {a['n']} | {_pct(a['accuracy'])} | {_f(a['think_mean'])} | {_f(a['think_median'])} | {a['latency_mean']:.2f}s | {a['truncated']} |")
        L.append("")
        L.append("| pair | acc Δ (95% CI) | McNemar p | fixed / broken | think ↓ mean | think ↓ median | items shorter | overspend removed | needed cut | speedup |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n, p in e["pairs"].items():
            ci = p["accuracy_delta"]; cov = p.get("coverage", {})
            L.append(f"| {n} | {ci['delta']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] | {p['mcnemar_p']:.3f} | {p['fixed']} / {p['broken']} | {_pct(p['think_reduction_mean'])} | {_pct(p['think_reduction_median'])} | {_pct(p['share_items_shorter'])} | {_pct(cov.get('overspend_removed'))} | {_pct(cov.get('needed_cut'))} | {p['speedup']:.2f}x |")
        L.append("\nPer domain:\n")
        L.append("| pair | domain | n | acc base → other | think ↓ | fixed / broken |")
        L.append("|---|---|---:|---|---:|---:|")
        for n, p in e["pairs"].items():
            for d, v in p["domains"].items():
                L.append(f"| {n} | {d} | {v['n']} | {_pct(v['accuracy_base'])} → {_pct(v['accuracy_other'])} | {_pct(v['think_reduction'])} | {v['fixed']} / {v['broken']} |")
        L.append("")
    if "quant" in run:
        q = run["quant"]
        L.append("## 7. Quantization\n")
        for k, v in q.items():
            L.append(f"- {k}: {v}")
        L.append("")
    return "\n".join(L)


def _svg_bars(labels: list[str], values: list[float], title: str, fmt=lambda v: f"{v:.2f}", w=520, h=220) -> str:
    n = max(1, len(values)); vmax = max([abs(v) for v in values] + [1e-9])
    bw = (w - 60) / n
    parts = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img"><text x="8" y="16" font-size="13" font-weight="600">{html.escape(title)}</text>']
    for i, (lab, v) in enumerate(zip(labels, values)):
        bh = (h - 70) * abs(v) / vmax
        x = 40 + i * bw; y = h - 40 - bh
        parts.append(f'<rect x="{x + 4:.1f}" y="{y:.1f}" width="{bw - 8:.1f}" height="{bh:.1f}" fill="#4c78a8" rx="3"/>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{y - 4:.1f}" font-size="11" text-anchor="middle">{html.escape(fmt(v))}</text>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{h - 24}" font-size="11" text-anchor="middle">{html.escape(str(lab))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_line(xs: list[float], ys: list[float], title: str, w=520, h=220) -> str:
    if not xs:
        return ""
    xmin, xmax = min(xs), max(xs); ymin, ymax = 0.0, max(ys + [1e-9])
    def X(x): return 40 + (w - 60) * ((x - xmin) / (xmax - xmin) if xmax > xmin else 0.5)
    def Y(y): return h - 30 - (h - 60) * (y - ymin) / (ymax - ymin)
    pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img"><text x="8" y="16" font-size="13" font-weight="600">{html.escape(title)}</text>',
         f'<polyline points="{pts}" fill="none" stroke="#e45756" stroke-width="2"/>']
    for x, y in zip(xs, ys):
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="#e45756"/><text x="{X(x):.1f}" y="{h - 12}" font-size="10" text-anchor="middle">{x:g}</text>')
    s.append("</svg>")
    return "".join(s)


def html_report(run: dict) -> str:
    md = markdown(run)
    charts = []
    if "scaling" in run:
        pts = run["scaling"]["points"]
        charts.append(_svg_line([p["n"] for p in pts], [p["markers_discovered"] for p in pts], "Markers discovered vs. dataset size"))
        charts.append(_svg_line([p["n"] for p in pts], [p["removable_share"] for p in pts], "Removable thinking share vs. dataset size"))
    if "eval" in run:
        pairs = run["eval"]["pairs"]
        charts.append(_svg_bars(list(pairs), [p["think_reduction_mean"] for p in pairs.values()], "Thinking-token reduction (mean)", lambda v: f"{100 * v:.1f}%"))
        charts.append(_svg_bars(list(pairs), [p["accuracy_delta"]["delta"] for p in pairs.values()], "Accuracy delta vs base", lambda v: f"{100 * v:+.1f}pp"))
    body = _md_to_html(md)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(run.get('name', 'swiftlab'))}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:4px 8px}}th{{background:#f4f4f4}}code{{background:#f4f4f4;padding:1px 4px}}.charts{{display:flex;flex-wrap:wrap;gap:1rem}}</style></head>
<body>{body}<h2>Charts</h2><div class="charts">{''.join(charts)}</div><details><summary>raw json</summary><pre>{html.escape(json.dumps({k: v for k, v in run.items() if k != 'rows'}, indent=1)[:200000])}</pre></details></body></html>"""


def _md_to_html(md: str) -> str:
    out, in_table = [], False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                out.append("<table>"); in_table = True
                out.append("<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        if line.startswith("# "): out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "): out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("- "): out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip(): out.append(f"<p>{_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


def _inline(s: str) -> str:
    import re
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def write_report(run: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(markdown(run), encoding="utf-8")
    (out_dir / "report.html").write_text(html_report(run), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")
    return {"md": str(out_dir / "report.md"), "html": str(out_dir / "report.html"), "json": str(out_dir / "report.json")}
