import torch
from pathlib import Path

STATS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "encoded_reps_stats.pt"
TOP_K = 10

def main():
    data = torch.load(STATS_PATH)
    means = data["means"]
    vars = data["variances"]

    # a list of sets. Indexed by race ID. Each set holds the indices into 
    # the variance vector that correspond the smallest variance values
    # ---> essentially a list of the most static features for each race
    all_lowest_var_indices = []

    for race in range(len(means)):
        mean = means[race].float()
        var = vars[race].float()
        lowest_var_indices = torch.topk(var, TOP_K, largest=False).indices.tolist()

        all_lowest_var_indices.append(set(lowest_var_indices))

        print(f"\nRace {race} — Top {TOP_K} low-variance features:")
        for idx in lowest_var_indices:
            print(f"  Feature {idx:6d} | mean = {mean[idx]:.4f}, var = {var[idx]:.6f}")

    # Optional: show overlap
    print("\nFeature overlap across races (how many races share a feature as low-variance):")
    from collections import Counter
    flat_indices = [idx for group in all_lowest_var_indices for idx in group]
    counts = Counter(flat_indices)
    for idx, count in counts.items():
        if count > 1:
            print(f"  Feature {idx:6d} appeared in {count} races' top {TOP_K} low-variance list")

if __name__ == "__main__":
    main()

