#!/usr/bin/env python3
"""
Analyze iris embedding database quality.

Inputs: a pickle file produced by build_database.py
Structure: { "<person_id>_<L|R>": np.ndarray(shape=(512,), dtype=float32) }

Metrics reported:
- Basic: number of classes, embedding dimensionality, L2-norm stats
- Pairwise cosine similarity matrix (impostor distribution stats)
- Genuine distribution (L vs R of same person) if both eyes present
- Top-N most confusing impostor pairs
- Optional plots: histograms of similarities
"""
import os
import re
import argparse
import pickle
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# 新增：可选的scipy用于DET曲线的probit坐标
try:
    from scipy.stats import norm as _norm
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

COMBINED_ID_RE = re.compile(r"^(?P<pid>.+)_(?P<eye>[LR])$")


def l2_norm_stats(embs: np.ndarray):
    norms = np.linalg.norm(embs, axis=1)
    return norms.mean(), norms.std(), norms.min(), norms.max()


def cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    # assumes embs are approximately L2-normalized
    # fallback to normalized version to be safe
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    e = embs / norms
    return e @ e.T


def parse_groups(keys):
    person_to_indices = {}
    for idx, k in enumerate(keys):
        m = COMBINED_ID_RE.match(k)
        if not m:
            # treat whole key as person id if pattern not matched
            pid = k
            eye = None
        else:
            pid = m.group("pid")
            eye = m.group("eye")
        person_to_indices.setdefault(pid, []).append((idx, eye))
    return person_to_indices


def histogram(data: np.ndarray, bins: int = 100):
    hist, edges = np.histogram(data, bins=bins, range=(-1.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

# 新增：计算FAR/FRR与EER

def compute_far_frr(impostor_sims: np.ndarray, genuine_sims: np.ndarray, num_thresholds: int = 1001):
    """Compute FAR/FRR over thresholds and EER for cosine similarities.
    FAR(t): fraction of impostor >= t; FRR(t): fraction of genuine < t.
    Returns thresholds, FAR, FRR, EER, EER_threshold. If genuine is empty, returns None.
    """
    if genuine_sims is None or genuine_sims.size == 0:
        return None
    # Define threshold range from overall min/max, clipped to [-1, 1]
    mins = []
    maxs = []
    if impostor_sims is not None and impostor_sims.size > 0:
        mins.append(float(np.min(impostor_sims)))
        maxs.append(float(np.max(impostor_sims)))
    if genuine_sims is not None and genuine_sims.size > 0:
        mins.append(float(np.min(genuine_sims)))
        maxs.append(float(np.max(genuine_sims)))
    tmin = min(mins) if mins else -1.0
    tmax = max(maxs) if maxs else 1.0
    tmin = max(-1.0, tmin)
    tmax = min(1.0, tmax)
    thresholds = np.linspace(tmin, tmax, int(max(3, num_thresholds)), dtype=np.float32)

    imp = impostor_sims.astype(np.float32) if impostor_sims is not None else np.empty((0,), dtype=np.float32)
    gen = genuine_sims.astype(np.float32)

    FAR = (imp[:, None] >= thresholds[None, :]).mean(axis=0) if imp.size > 0 else np.zeros_like(thresholds, dtype=np.float32)
    FRR = (gen[:, None] < thresholds[None, :]).mean(axis=0)

    # Equal Error Rate (EER)
    idx = int(np.argmin(np.abs(FAR - FRR)))
    eer = float(0.5 * (FAR[idx] + FRR[idx]))
    eer_thr = float(thresholds[idx])
    return thresholds, FAR.astype(float), FRR.astype(float), eer, eer_thr


def main():
    ap = argparse.ArgumentParser(description="Analyze iris database quality")
    ap.add_argument("--database_path", type=str, required=True, help="Path to database .pkl")
    ap.add_argument("--out_dir", type=str, default="outputs/db_analysis", help="Directory to save reports/plots")
    ap.add_argument("--topk", type=int, default=20, help="Report top-K most similar impostor pairs")
    ap.add_argument("--no_plots", action="store_true", help="Disable plotting even if matplotlib is available")
    # 新增：阈值数量用于计算FAR/FRR、ROC与DET
    ap.add_argument("--num_thresholds", type=int, default=1001, help="Number of thresholds for FAR/FRR/ROC/DET computation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load database
    with open(args.database_path, "rb") as f:
        db = pickle.load(f)

    keys = list(db.keys())
    embs = np.stack([np.asarray(db[k], dtype=np.float32) for k in keys], axis=0)
    num_classes, dim = embs.shape

    # L2 norm stats
    mean_n, std_n, min_n, max_n = l2_norm_stats(embs)

    # Cosine similarity matrix
    sim = cosine_similarity_matrix(embs)
    # Mask out diagonal
    mask_offdiag = ~np.eye(num_classes, dtype=bool)
    impostor_sims = sim[mask_offdiag]

    impostor_mean = float(np.mean(impostor_sims))
    impostor_std = float(np.std(impostor_sims))
    impostor_p95 = float(np.percentile(impostor_sims, 95))
    impostor_max = float(np.max(impostor_sims))

    # Genuine distribution (only L vs R of same person if available)
    person_groups = parse_groups(keys)
    genuine_list = []
    for pid, idx_eyes in person_groups.items():
        # collect pairwise among this person's entries
        idxs = [idx for idx, _ in idx_eyes]
        if len(idxs) >= 2:
            sub = sim[np.ix_(idxs, idxs)]
            m = ~np.eye(len(idxs), dtype=bool)
            vals = sub[m]
            genuine_list.extend(vals.tolist())
    genuine_sims = np.array(genuine_list, dtype=np.float32) if genuine_list else np.array([], dtype=np.float32)

    # 新增：计算FAR/FRR与EER，并导出CSV
    rates_csv_path = None
    eer_value = None
    eer_threshold = None
    far = None
    frr = None
    thresholds = None
    comp = compute_far_frr(impostor_sims, genuine_sims, num_thresholds=args.num_thresholds)
    if comp is not None:
        thresholds, far, frr, eer_value, eer_threshold = comp
        rates_csv_path = os.path.join(args.out_dir, "threshold_rates.csv")
        with open(rates_csv_path, "w", encoding="utf-8") as f:
            f.write("threshold,FAR,FRR,TPR,FPR\n")
            for t, fa, fr in zip(thresholds, far, frr):
                tpr = 1.0 - fr
                fpr = fa
                f.write(f"{t:.6f},{fa:.6f},{fr:.6f},{tpr:.6f},{fpr:.6f}\n")

    # Top-K most similar impostor pairs
    # Flatten upper triangle (excluding diag)
    iu = np.triu_indices(num_classes, k=1)
    flat_vals = sim[iu]
    order = np.argsort(flat_vals)[::-1]
    topk = min(args.topk, order.size)
    top_pairs = [(keys[iu[0][o]], keys[iu[1][o]], float(flat_vals[o])) for o in order[:topk]]

    # Save CSV of top confusing pairs
    csv_path = os.path.join(args.out_dir, "top_impostor_pairs.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,probe,candidate,cosine_similarity\n")
        for r, (a, b, s) in enumerate(top_pairs, start=1):
            f.write(f"{r},{a},{b},{s:.6f}\n")

    # Plots
    if HAS_PLT and not args.no_plots:
        # Norms
        plt.figure(figsize=(6,4))
        plt.hist(np.linalg.norm(embs, axis=1), bins=50, color="#4472C4", alpha=0.85)
        plt.title("Embedding L2 Norms")
        plt.xlabel("L2 norm")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "norm_hist.png"))
        plt.close()

        # Impostor and genuine hists
        plt.figure(figsize=(6,4))
        plt.hist(impostor_sims, bins=100, range=(-1,1), color="#ED7D31", alpha=0.6, label="impostor")
        if genuine_sims.size > 0:
            plt.hist(genuine_sims, bins=100, range=(-1,1), color="#70AD47", alpha=0.6, label="genuine")
        plt.title("Cosine Similarity Distributions")
        plt.xlabel("cosine similarity")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "similarity_hists.png"))
        plt.close()

        # 新增：ROC曲线
        if thresholds is not None:
            fpr = far
            tpr = 1.0 - frr
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, color="#1f77b4", label="ROC")
            plt.plot([0,1],[0,1], linestyle="--", color="#888888", linewidth=1)
            if eer_value is not None:
                # 找到最接近EER的点用于可视化
                idx_e = int(np.argmin(np.abs((far - (1.0 - frr)))))
                plt.scatter([fpr[idx_e]],[tpr[idx_e]], color="#d62728", s=40, label=f"EER≈{eer_value*100:.2f}%")
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel("False Positive Rate (FAR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "roc_curve.png"))
            plt.close()

            # 新增：DET曲线（FRR vs FAR）。如有scipy则使用probit坐标。
            x = far
            y = frr
            if HAS_SCIPY:
                eps = 1e-6
                x = _norm.ppf(np.clip(far, eps, 1 - eps))
                y = _norm.ppf(np.clip(frr, eps, 1 - eps))
                xlabel = "FAR (norm deviate)"
                ylabel = "FRR (norm deviate)"
            else:
                xlabel = "FAR"
                ylabel = "FRR"
            plt.figure(figsize=(6,5))
            plt.plot(x, y, color="#2ca02c", label="DET")
            if eer_value is not None:
                if HAS_SCIPY:
                    ex = _norm.ppf(np.clip(eer_value, 1e-6, 1-1e-6))
                    ey = ex
                else:
                    ex = eer_value
                    ey = eer_value
                plt.scatter([ex],[ey], color="#d62728", s=40, label=f"EER≈{eer_value*100:.2f}% @ t={eer_threshold:.3f}")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title("DET Curve (FRR vs FAR)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "det_curve.png"))
            plt.close()

    # Text report
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Iris Database Quality Report\n")
        f.write("==============================\n\n")
        f.write(f"Database: {os.path.abspath(args.database_path)}\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"Embedding Dim: {dim}\n\n")
        f.write("Embedding L2 Norms:\n")
        f.write(f"  mean={mean_n:.6f}, std={std_n:.6f}, min={min_n:.6f}, max={max_n:.6f}\n\n")
        f.write("Impostor Cosine Similarities (off-diagonal):\n")
        f.write(f"  mean={impostor_mean:.6f}, std={impostor_std:.6f}, p95={impostor_p95:.6f}, max={impostor_max:.6f}\n\n")
        if genuine_sims.size > 0:
            f.write("Genuine Cosine Similarities (same person pairs):\n")
            f.write(f"  mean={float(np.mean(genuine_sims)):.6f}, std={float(np.std(genuine_sims)):.6f}, min={float(np.min(genuine_sims)):.6f}, max={float(np.max(genuine_sims)):.6f}\n\n")
        else:
            f.write("Genuine Cosine Similarities: N/A (no persons with multiple entries)\n\n")
        # 新增：EER摘要
        if eer_value is not None:
            f.write(f"FAR/FRR Analysis:\n  EER={eer_value*100:.3f}% at threshold={eer_threshold:.6f}\n")
            if rates_csv_path is not None:
                f.write(f"  Rates CSV: {os.path.abspath(rates_csv_path)}\n")
            if HAS_PLT and not args.no_plots:
                f.write("  Plots saved: roc_curve.png, det_curve.png\n")
        f.write(f"Top-{topk} Most Similar Impostor Pairs (see CSV): {os.path.abspath(csv_path)}\n")
        if HAS_PLT and not args.no_plots:
            f.write("Plots saved: norm_hist.png, similarity_hists.png\n")

    print("Analysis complete.")
    print(f"Report: {report_path}")
    print(f"CSV   : {csv_path}")
    if rates_csv_path is not None:
        print(f"Rates : {rates_csv_path}")
    if HAS_PLT and not args.no_plots:
        extra_plots = [os.path.join(args.out_dir, 'norm_hist.png'), os.path.join(args.out_dir, 'similarity_hists.png')]
        if thresholds is not None:
            extra_plots += [os.path.join(args.out_dir, 'roc_curve.png'), os.path.join(args.out_dir, 'det_curve.png')]
        print("Plots : " + ", ".join(extra_plots))


if __name__ == "__main__":
    main()


