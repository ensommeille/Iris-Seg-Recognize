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


def main():
    ap = argparse.ArgumentParser(description="Analyze iris database quality")
    ap.add_argument("--database_path", type=str, required=True, help="Path to database .pkl")
    ap.add_argument("--out_dir", type=str, default="outputs/db_analysis", help="Directory to save reports/plots")
    ap.add_argument("--topk", type=int, default=20, help="Report top-K most similar impostor pairs")
    ap.add_argument("--no_plots", action="store_true", help="Disable plotting even if matplotlib is available")
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
        f.write(f"Top-{topk} Most Similar Impostor Pairs (see CSV): {os.path.abspath(csv_path)}\n")
        if HAS_PLT and not args.no_plots:
            f.write("Plots saved: norm_hist.png, similarity_hists.png\n")

    print("Analysis complete.")
    print(f"Report: {report_path}")
    print(f"CSV   : {csv_path}")
    if HAS_PLT and not args.no_plots:
        print(f"Plots : {os.path.join(args.out_dir, 'norm_hist.png')}, {os.path.join(args.out_dir, 'similarity_hists.png')}")


if __name__ == "__main__":
    main()


