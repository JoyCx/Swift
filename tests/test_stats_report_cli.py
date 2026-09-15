from swiftlab.stats import paired_bootstrap, mcnemar_exact, median
from swiftlab.quant import parse_kl_output, gguf_commands, build_calibration, size_table
from swiftlab.report import markdown, html_report


def test_stats():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
    b = paired_bootstrap([1, 2, 3, 4], [2, 3, 4, 5], n_boot=200)
    assert abs(b["delta"] - 1.0) < 1e-9 and b["lo"] <= 1.0 <= b["hi"]
    assert median([3, 1, 2]) == 2 and median([1, 2, 3, 4]) == 2.5


def test_quant_helpers():
    out = parse_kl_output("Mean KLD: 0.012345\n99.0% KLD: 0.2\nMaximum KLD: 1.5\nSame top p: 95.1")
    assert out["mean_kl"] == 0.012345 and out["top1_agreement"] == 95.1
    cmds = gguf_commands("/m/x", "/o", "c.txt", "h.txt", ["Q4_K_M"])
    assert any("llama-imatrix" in c for c in cmds) and any("--kl-divergence" in c for c in cmds)
    rows = [{"correct": True, "task_id": "a", "domain": "coding", "thinking": "t", "answer": "x"}]
    from swiftlab.tasks import Task
    cal = build_calibration(rows, {"a": Task("a", "coding", "p", {"kind": "exact", "gold": ["x"]})}, n=5)
    assert "<think>" in cal[0]["text"]
    assert size_table(27, ["Q4_K_M"])[0]["approx_gb"] > 10


def test_report_renders(paired_rows, settled_rows, tmp_path):
    from swiftlab.evaluate import compare_all
    from swiftlab.scaling import scaling_curve
    from swiftlab.report import write_report
    run = {"name": "t", "model": "real-model", "effort": "xhigh", "seeds": 5, "n_tasks": 60,
           "waste": __import__("swiftlab.settle", fromlist=["waste_summary"]).waste_summary(settled_rows),
           "scaling": scaling_curve(settled_rows, [15, 30, 60]),
           "eval": compare_all(paired_rows, base="base", settled_base=settled_rows, n_boot=100)}
    md = markdown(run); html = html_report(run)
    assert "Paired evaluation" in md and "<svg" in html and "base-&gt;edited" in html or "base->edited" in md
    paths = write_report(run, tmp_path / "r")
    assert (tmp_path / "r" / "report.html").exists()
