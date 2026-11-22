import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distance(a, b):
    # 기존 sampler.py의 거리 계산과 동일한 방식
    aa = (a * a).sum(dim=1, keepdim=True)
    bb = (b * b).sum(dim=1).unsqueeze(0)
    return (aa + bb - 2 * (a @ b.T)).clamp_min(0).sqrt()


@torch.no_grad()
def profile(features, n, ratio, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    rng = np.random.default_rng(seed)
    ids = rng.choice(len(features), size=n, replace=False)

    x = torch.from_numpy(
        np.ascontiguousarray(features[ids])
    ).to(device)

    timings = {
        "projection": 0.0,
        "initialization": 0.0,
        "argmax": 0.0,
        "distance_update": 0.0,
    }

    sync(device)
    t = time.perf_counter()

    # 기존 GreedyCoresetSampler의 랜덤 투영
    mapper = torch.nn.Linear(
        x.shape[1], 128, bias=False
    ).to(device)

    reduced = mapper(x)

    sync(device)
    timings["projection"] = time.perf_counter() - t

    # 기존 ApproximateGreedy의 초기화
    t = time.perf_counter()
    start_ids = np.random.choice(
        n, min(10, n), replace=False
    )

    initial_dist = distance(
        reduced, reduced[start_ids]
    )
    nearest_dist = initial_dist.mean(
        dim=1, keepdim=True
    )

    sync(device)
    timings["initialization"] = time.perf_counter() - t

    k = int(n * ratio)
    selected = []

    for _ in tqdm(
        range(k), desc=f"N={n}", leave=False
    ):
        sync(device)
        t = time.perf_counter()

        idx = torch.argmax(nearest_dist).item()
        selected.append(idx)

        sync(device)
        timings["argmax"] += time.perf_counter() - t

        t = time.perf_counter()

        new_dist = distance(
            reduced, reduced[idx:idx + 1]
        )

        nearest_dist = torch.minimum(
            nearest_dist, new_dist
        )

        sync(device)
        timings["distance_update"] += (
            time.perf_counter() - t
        )

    total = sum(timings.values())

    # 선택 결과의 Coverage 평가 (시간 측정에서 제외)
    # 전체 후보 x와 선택된 Coreset 간 최소 L2 거리
    chosen = x[selected]
    min_dist = torch.cdist(x, chosen).min(dim=1).values

    coverage_mean = min_dist.mean().item()
    coverage_max = min_dist.max().item()

    # 초기 기준점 10개 + 선택 반복 k회
    distance_pairs = n * (min(10, n) + k)

    return {
        "N": n,
        "K": k,
        "ratio": ratio,
        "projection_s": timings["projection"],
        "initialization_s": timings["initialization"],
        "argmax_s": timings["argmax"],
        "distance_update_s": timings["distance_update"],
        "total_s": total,
        "distance_pairs": distance_pairs,
        "coverage_mean": coverage_mean,
        "coverage_max": coverage_max,
    }


def main(args):
    index = faiss.read_index(str(args.index))
    if not isinstance(index, faiss.IndexFlatL2):
        raise TypeError("Expected IndexFlatL2")

    features = index.reconstruct_n(
        0, index.ntotal
    ).astype(np.float32)

    device = torch.device(args.device)

    print(f"Source: {args.index}")
    print(f"Available features: {features.shape}")
    print(f"Device: {device}")

    rows = []

    for n in args.sizes:
        if n > len(features):
            print(f"Skipping N={n}: insufficient features")
            continue

        result = profile(
            features, n, args.ratio, device, args.seed
        )
        rows.append(result)

        print(f"\nN={n}, K={result['K']}")
        print(f"  Total:          {result['total_s']:.3f} s")
        print(f"  Projection:     {result['projection_s']:.3f} s")
        print(f"  Initialization: {result['initialization_s']:.3f} s")
        print(f"  Argmax:         {result['argmax_s']:.3f} s")
        print(f"  Distance update:{result['distance_update_s']:.3f} s")
        print(f"  Coverage mean:  {result['coverage_mean']:.4f}")
        print(f"  Coverage max:   {result['coverage_max']:.4f}")

    if rows:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"\nSaved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[1000, 2000, 4000]
    )
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="experiments/coreset/profile_results.csv"
    )
    main(parser.parse_args())
