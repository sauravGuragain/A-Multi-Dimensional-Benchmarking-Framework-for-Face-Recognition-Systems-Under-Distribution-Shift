"""Leakage and pair-count invariants for identity-disjoint verification splits (any seed, any fold).

``check_split_and_pairs`` raises ``SplitInvariantError`` listing EVERY violation; it never repairs anything. It
restates, independently of the manifest builder, the properties a multi-split evaluation depends on:

* the train, validation and test partitions share no identity and no image;
* every image appears in exactly one partition (no duplicate record inside a partition);
* every genuine pair joins two different images of ONE identity, every impostor pair (pool and balanced subset) joins
  two images of DIFFERENT identities, all inside the partition the pair set was built for;
* the balanced impostor subset has exactly as many pairs as there are genuine pairs and is drawn from the pool;
* pairs are unique (no pair listed twice) and the pool is rho times the number of genuine pairs.
"""
from __future__ import annotations

from typing import Mapping, Sequence


class SplitInvariantError(RuntimeError):
    pass


def check_split_and_pairs(partitions: Mapping[str, Sequence[tuple[str, str]]], pair_sets: Mapping[str, object],
                          rho: int = 10) -> dict[str, dict[str, int]]:
    problems: list[str] = []
    ids: dict[str, set[str]] = {}
    imgs: dict[str, set[str]] = {}
    for part, rows in partitions.items():
        ids[part] = {i for i, _ in rows}
        imgs[part] = {m for _, m in rows}
        if len(imgs[part]) != len(rows):
            problems.append(f"partition {part}: duplicate image record(s)")
    names = list(partitions)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = names[a], names[b]
            if ids[x] & ids[y]:
                problems.append(f"identity leakage between {x} and {y}: {sorted(ids[x] & ids[y])[:3]}")
            if imgs[x] & imgs[y]:
                problems.append(f"image overlap between {x} and {y}: {sorted(imgs[x] & imgs[y])[:3]}")
    counts: dict[str, dict[str, int]] = {}
    for part, ps in pair_sets.items():
        table = ps.images                                    # (identity, image_id), sorted
        if {m for _, m in table} != imgs.get(part):
            problems.append(f"pairs[{part}]: the image table is not the {part} partition")
        n = len(table)
        gen = list(ps.genuine)
        pool = ps.sorted_pool()
        bal = ps.balanced_pairs()

        def bad_index(p):
            return not (0 <= p[0] < n and 0 <= p[1] < n and p[0] != p[1])
        if any(bad_index(p) for p in gen + pool):
            problems.append(f"pairs[{part}]: a pair index is out of range or a pair joins an image with itself")
            continue
        if any(table[a][0] != table[b][0] for a, b in gen):
            problems.append(f"pairs[{part}]: a genuine pair joins two different identities")
        if any(table[a][0] == table[b][0] for a, b in pool):
            problems.append(f"pairs[{part}]: an impostor pair joins two images of one identity")
        if len(set(gen)) != len(gen) or len(set(pool)) != len(pool):
            problems.append(f"pairs[{part}]: a pair is listed twice")
        if len(bal) != len(gen):
            problems.append(f"pairs[{part}]: balanced impostors {len(bal)} != genuine pairs {len(gen)}")
        if len(set(bal)) != len(bal) or not set(bal) <= set(pool):
            problems.append(f"pairs[{part}]: the balanced subset is not a duplicate-free subset of the pool")
        if len(pool) != rho * len(gen):
            problems.append(f"pairs[{part}]: pool {len(pool)} != rho({rho}) x genuine {len(gen)}")
        counts[part] = {"n_images": n, "n_genuine": len(gen), "n_pool": len(pool), "n_balanced": len(bal)}
    if problems:
        raise SplitInvariantError("; ".join(problems))
    return counts
