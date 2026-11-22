import argparse
import time

import faiss
import numpy as np
import torch
import tqdm

from patchcore.sampler import ApproximateGreedyCoresetSampler

# Disable tqdm in original sampler for fair timing.
import patchcore.sampler as sampler_module
sampler_module.tqdm.tqdm = lambda iterable, **kwargs: iterable



class OptimizedSampler(ApproximateGreedyCoresetSampler):
    def _compute_greedy_coreset_indices(self, features):
        n_start = min(self.number_of_starting_points, len(features))
        start_ids = np.random.choice(
            len(features), n_start, replace=False
        ).tolist()

        distances = self._compute_batchwise_differences(
            features, features[start_ids]
        )
        nearest = distances.mean(dim=-1).reshape(-1, 1)

        selected = []
        k = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in range(k):
                idx = torch.argmax(nearest).item()
                selected.append(idx)

                new_distance = self._compute_batchwise_differences(
                    features, features[idx:idx + 1]
                )

                nearest = torch.minimum(nearest, new_distance)

        return np.array(selected)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(sampler_cls, features, device, seed):
    set_seed(seed)
    sampler = sampler_cls(
        percentage=0.1,
        device=device,
        number_of_starting_points=10,
        dimension_to_project_features_to=128,
    )

    synchronize(device)
    start = time.perf_counter()

    selected = sampler.run(features)

    synchronize(device)
    elapsed = time.perf_counter() - start

    return elapsed, selected


def main(args):
    device = torch.device(args.device)

    index = faiss.read_index(args.index)
    features = index.reconstruct_n(
        0, index.ntotal
    ).astype(np.float32)

    print(f"Source features: {features.shape}")
    print(f"Device: {device}")

    rng = np.random.default_rng(42)
    results = []

    for n in args.sizes:
        ids = rng.choice(len(features), n, replace=False)
        x = np.ascontiguousarray(features[ids])

        print(f"\n===== N={n:,}, K={int(n * 0.1):,} =====")

        # GPU warm-up: 측정에 포함하지 않음
        for cls in [
            ApproximateGreedyCoresetSampler,
            OptimizedSampler,
        ]:
            benchmark(cls, x[:min(1000, n)], device, args.seed)

        original_times = []
        optimized_times = []
        identical = []

        for repeat in range(args.repeats):
            original_time, original_features = benchmark(
                ApproximateGreedyCoresetSampler,
                x, device, args.seed + repeat,
            )
            optimized_time, optimized_features = benchmark(
                OptimizedSampler,
                x, device, args.seed + repeat,
            )

            original_times.append(original_time)
            optimized_times.append(optimized_time)

            same = np.array_equal(
                original_features, optimized_features
            )
            identical.append(same)

            print(
                f"Repeat {repeat}: "
                f"Original={original_time:.4f}s, "
                f"Optimized={optimized_time:.4f}s, "
                f"Identical={same}"
            )

        orig = np.median(original_times)
        opt = np.median(optimized_times)

        print(f"\nOriginal median:  {orig:.4f}s")
        print(f"Optimized median: {opt:.4f}s")
        print(f"Speedup:          {orig / opt:.3f}x")
        print(f"All identical:    {all(identical)}")

        results.append((n, orig, opt, orig / opt, all(identical)))

    print("\n========== FINAL SUMMARY ==========")
    for n, orig, opt, speedup, identical in results:
        print(
            f"N={n:6d} | Original={orig:.4f}s | "
            f"Optimized={opt:.4f}s | "
            f"Speedup={speedup:.3f}x | Identical={identical}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[5000, 10000, 20000],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
