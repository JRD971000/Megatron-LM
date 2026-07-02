"""Overlay two Megatron SFT loss traces (eager vs ffpa_flash) -> V4 parity PNG.

Parses the per-step ``lm loss`` from two Megatron training logs (as emitted by
gemma4_e4b_sft_parity.sh, log-interval 1), overlays them, and plots the per-step
RELATIVE difference series so V4 training-curve parity is visible at a glance:
matched curves + a flat, small relative-diff line (that does NOT grow with step).

Usage (in a container with matplotlib):
    python examples/gemma4/plot_loss_overlay.py \
        --eager .../loss_eager.log --ffpa .../loss_ffpa_flash.log \
        --out .../loss_overlay.png

Dry-run the parser + diff logic anywhere (no matplotlib / no GPU needed):
    python examples/gemma4/plot_loss_overlay.py --self-test
"""
import argparse
import re
import sys

# ' iteration        1/      20 | ... | lm loss: 2.345678E+00 |'
_ITER_RE = re.compile(r"iteration\s+(\d+)\s*/")
_LOSS_RE_TMPL = r"{key}:\s*([-+0-9.]+[Ee][-+]?\d+|[-+0-9.]+)"


def parse_loss(path, loss_key="lm loss"):
    """Parse [(iteration, loss)] from a Megatron training log."""
    loss_re = re.compile(_LOSS_RE_TMPL.format(key=re.escape(loss_key)))
    out = []
    with open(path) as f:
        for line in f:
            if "iteration" not in line or loss_key not in line:
                continue
            it = _ITER_RE.search(line)
            lo = loss_re.search(line)
            if it and lo:
                out.append((int(it.group(1)), float(lo.group(1))))
    return out


def align(a, b):
    """Inner-join two [(it, loss)] traces on iteration -> (iters, la, lb)."""
    da, db = dict(a), dict(b)
    iters = sorted(set(da) & set(db))
    return iters, [da[i] for i in iters], [db[i] for i in iters]


def rel_diff(la, lb):
    """Per-step relative difference |a-b| / (|a| + eps)."""
    return [abs(x - y) / (abs(x) + 1e-12) for x, y in zip(la, lb)]


def summarize(iters, la, lb):
    rd = rel_diff(la, lb)
    max_abs = max(abs(x - y) for x, y in zip(la, lb)) if iters else float("nan")
    max_rel = max(rd) if rd else float("nan")
    print(f"  aligned steps      : {len(iters)}")
    print(f"  max abs loss diff  : {max_abs:.4e}")
    print(f"  max rel loss diff  : {max_rel:.4e}")
    if len(rd) >= 2:
        # A legit tolerance does NOT grow with step: compare first vs last half.
        half = len(rd) // 2
        early = sum(rd[:half]) / max(1, half)
        late = sum(rd[half:]) / max(1, len(rd) - half)
        print(f"  mean rel (early|late): {early:.4e} | {late:.4e} "
              f"({'stable' if late <= 2 * early + 1e-9 else 'GROWING -> investigate'})")
    return max_abs, max_rel, rd


def plot(iters, la, lb, rd, out, tol_loss=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(iters, la, "o-", label="eager (oracle)", ms=3)
    ax1.plot(iters, lb, "x--", label="ffpa_flash", ms=4)
    ax1.set_ylabel("lm loss")
    ax1.set_title("Gemma4 E4B SFT training-curve parity (V4)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, rd, "s-", color="crimson", ms=3, label="per-step rel diff")
    if tol_loss is not None:
        ax2.axhline(tol_loss, color="gray", ls=":", label=f"TOL_LOSS={tol_loss}")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("|Δloss| / |loss|")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def _self_test():
    """Validate parse + align + diff on synthetic Megatron-format log lines."""
    import os
    import tempfile

    def _fake_log(losses):
        lines = []
        for i, l in enumerate(losses, 1):
            lines.append(
                f" [2026-07-01] iteration {i:8d}/      20 | consumed samples: {i*4:6d} |"
                f" elapsed time per iteration (ms): 500.0 | learning rate: 1.000000E-05 |"
                f" global batch size:     4 | lm loss: {l:.6E} | grad norm: 1.234 |"
            )
        return "\n".join(lines) + "\n"

    eager = [2.5 - 0.05 * i for i in range(20)]
    ffpa = [v * (1 + 6e-3) for v in eager]  # ~V0 rel-6e-3 noise, flat in step
    d = tempfile.mkdtemp()
    pe, pf = os.path.join(d, "e.log"), os.path.join(d, "f.log")
    open(pe, "w").write(_fake_log(eager))
    open(pf, "w").write(_fake_log(ffpa))

    a, b = parse_loss(pe), parse_loss(pf)
    assert len(a) == 20 and len(b) == 20, (len(a), len(b))
    iters, la, lb = align(a, b)
    assert iters == list(range(1, 21)), iters
    assert abs(la[0] - 2.5) < 1e-6 and abs(lb[0] - 2.5 * 1.006) < 1e-4
    print("[self-test] parse/align OK (20 steps each)")
    max_abs, max_rel, rd = summarize(iters, la, lb)
    assert abs(max_rel - 6e-3) < 5e-4, max_rel
    print("[self-test] PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eager", help="eager (oracle) loss log")
    ap.add_argument("--ffpa", help="ffpa_flash loss log")
    ap.add_argument("--out", default="loss_overlay.png")
    ap.add_argument("--loss-key", default="lm loss")
    ap.add_argument("--tol-loss", type=float, default=None,
                    help="draw a TOL_LOSS reference line on the rel-diff panel")
    ap.add_argument("--no-plot", action="store_true", help="parse + summarize only")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not (args.eager and args.ffpa):
        ap.error("--eager and --ffpa are required (or use --self-test)")

    a = parse_loss(args.eager, args.loss_key)
    b = parse_loss(args.ffpa, args.loss_key)
    print(f"eager: {len(a)} steps  |  ffpa_flash: {len(b)} steps")
    if not a or not b:
        print("ERROR: no loss points parsed -- check the log / --loss-key", file=sys.stderr)
        sys.exit(1)
    iters, la, lb = align(a, b)
    _max_abs, _max_rel, rd = summarize(iters, la, lb)
    if not args.no_plot:
        plot(iters, la, lb, rd, args.out, args.tol_loss)


if __name__ == "__main__":
    main()
