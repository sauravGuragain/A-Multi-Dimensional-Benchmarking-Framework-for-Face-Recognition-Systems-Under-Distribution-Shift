"""Verification pair builder (Revision 4.1, section 3).

Pure functions of an image table and seeds; no models, no images, no I/O.
Genuine pairs: per identity, all within-identity pairs if C(n,2) <= cap,
otherwise ``cap`` pairs drawn uniformly without replacement (sub-stream per
identity). Impostor pool: uniform draws over image-index pairs, processed
strictly in draw order, rejecting same-image, same-identity and duplicate
pairs, stopping at the K-th acceptance (so the pool is a uniform K-subset of
all valid cross-identity pairs, and every acceptance-order prefix is uniform).
"""
from __future__ import annotations

import hashlib
import itertools
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

STREAM_GENUINE = 1
STREAM_IMPOSTOR = 3
STREAM_BALANCED = 4
STREAM_ORIENTATION = 5
BATCH = 65_536  # fixed, independent of K (prefix property)

# Bumped whenever a change could alter any pair, rank, orientation or the
# canonical serialization. It is part of the manifest header and its hash.
ALGORITHM_VERSION = "faceeval-pairs/r4.1/1"

# Fold 0 acceptance counts (Revision 4.1): partition code -> (G, K).
FOLD0_EXPECTED: Dict[int, Tuple[int, int]] = {1: (684, 6840), 2: (764, 7640)}
FOLD0_DEFAULTS = {"seed": 42, "cap": 10, "rho": 10}
PARTITION_CODES = (1, 2)  # validation = 1, test = 2
MAX_SEED = 2**63 - 1


class Fold0AcceptanceError(RuntimeError):
    """Fold 0 pair counts differ from the Revision 4.1 acceptance values."""


def _int_param(name: str, value: Any, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < lo or (hi is not None and v > hi):
        raise ValueError(f"{name}={v} outside the allowed range [{lo}, {hi if hi is not None else 'inf'}]")
    return v


def validate_params(seed, partition_code, cap, rho) -> Tuple[int, int, int, int]:
    seed = _int_param("seed", seed, 0, MAX_SEED)
    partition_code = _int_param("partition_code", partition_code, 1)
    if partition_code not in PARTITION_CODES:
        raise ValueError(f"partition_code must be one of {PARTITION_CODES}, got {partition_code}")
    cap = _int_param("cap", cap, 1)
    rho = _int_param("rho", rho, 1)
    return seed, partition_code, cap, rho


def _clean_field(name: str, v: Any) -> str:
    if not isinstance(v, str) or v == "":
        raise ValueError(f"{name} must be a non-empty string, got {v!r}")
    if v != v.strip():
        raise ValueError(f"{name} has leading/trailing whitespace: {v!r}")
    if "|" in v or any(ord(c) < 32 or ord(c) == 127 for c in v):
        raise ValueError(f"{name} contains '|' or a control character: {v!r}")
    return str(v)


def validate_images(images) -> List[Tuple[str, str]]:
    """Structure check: a non-empty list/tuple of (identity, image_id) string pairs."""
    if not isinstance(images, (list, tuple)) or len(images) == 0:
        raise ValueError("image table must be a non-empty list or tuple of (identity, image_id) rows")
    out: List[Tuple[str, str]] = []
    for n, row in enumerate(images):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"image row {n} is not an (identity, image_id) pair: {row!r}")
        out.append((_clean_field(f"identity (row {n})", row[0]), _clean_field(f"image_id (row {n})", row[1])))
    return out


def stable_id(text: str) -> int:
    """First eight bytes of SHA-256 as an integer (never Python hash())."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _rng(seed: int, partition_code: int, stream: int, *extra: int) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, partition_code, stream, *extra]))
    )


@dataclass
class PairSet:
    images: List[Tuple[str, str]]                      # (identity, image_id), sorted
    genuine: List[Tuple[int, int]]                     # (lo, hi) image indices
    pool: List[Tuple[int, int, int]]                   # (lo, hi, draw_index) in acceptance order
    balanced: List[int]                                # indices into the canonically sorted pool
    probe_is_hi: Dict[Tuple[int, int], bool] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def sorted_pool(self) -> List[Tuple[int, int]]:
        return sorted((lo, hi) for lo, hi, _ in self.pool)

    def balanced_pairs(self) -> List[Tuple[int, int]]:
        sp = self.sorted_pool()
        return [sp[i] for i in self.balanced]


def sort_images(images: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Plain-string sort by (identity, image id); index order = this order."""
    out = sorted(validate_images(images))
    # sorted order puts duplicates next to each other: one O(n) pass
    dup = sorted({a for a, b in zip(out, out[1:]) if a == b})
    if dup:
        raise ValueError(f"duplicate (identity, image_id) rows: {dup[:5]}")
    return out


def _require_canonical(imgs: List[Tuple[str, str]]) -> None:
    """Helpers take the canonical table: strictly increasing (identity, image_id) rows, i.e. sorted
    with no duplicate record. Both conditions are checked, each with its own error."""
    for a, b in zip(imgs, imgs[1:]):
        if a == b:
            raise ValueError(f"duplicate (identity, image_id) row: {a}")
        if a > b:
            raise ValueError("images must be in canonical sorted order (use sort_images)")


def _pair_from_rank(n: int, r: int) -> Tuple[int, int]:
    """Unrank the r-th (0-based) pair of ``itertools.combinations(range(n), 2)`` (lexicographic).

    Pair (i, j), i < j, has rank i*(2n-i-1)/2 + (j-i-1); the first pair of row i has rank
    start(i) = i*(2n-i-1)/2. Exact integer arithmetic, O(log n), no enumeration.
    """
    lo, hi = 0, n - 2                      # largest i with start(i) <= r
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * (2 * n - mid - 1) // 2 <= r:
            lo = mid
        else:
            hi = mid - 1
    return lo, lo + 1 + (r - lo * (2 * n - lo - 1) // 2)


def build_genuine(images, seed: int, partition_code: int, cap: int) -> List[Tuple[int, int]]:
    """Per identity: all pairs if C(n,2) <= cap, else ``cap`` pair ranks drawn uniformly without
    replacement from the C(n,2) lexicographic ranks (sub-stream per identity), reported in rank
    order. Ranks are unranked arithmetically, so memory is O(cap) however large n is."""
    seed, partition_code, cap, _ = validate_params(seed, partition_code, cap, 1)
    imgs = validate_images(images)
    _require_canonical(imgs)
    by_id: Dict[str, List[int]] = {}
    for idx, (ident, _) in enumerate(imgs):
        by_id.setdefault(ident, []).append(idx)
    out: List[Tuple[int, int]] = []
    for ident in sorted(by_id):
        idxs = by_id[ident]
        n = len(idxs)
        if n < 2:
            continue
        total = n * (n - 1) // 2
        if total <= cap:
            out.extend(itertools.combinations(idxs, 2))
        else:
            rng = _rng(seed, partition_code, STREAM_GENUINE, stable_id(ident))
            ranks = sorted(int(r) for r in rng.choice(total, size=cap, replace=False))
            for r in ranks:
                a, b = _pair_from_rank(n, r)
                out.append((idxs[a], idxs[b]))
    return out


def count_valid_impostor_pairs(images) -> int:
    images = validate_images(images)
    _require_canonical(images)
    n = len(images)
    counts: Dict[str, int] = {}
    for ident, _ in images:
        counts[ident] = counts.get(ident, 0) + 1
    return n * (n - 1) // 2 - sum(c * (c - 1) // 2 for c in counts.values())


def build_impostor_pool(images, seed: int, partition_code: int, k: int) -> List[Tuple[int, int, int]]:
    seed, partition_code, _, _ = validate_params(seed, partition_code, 1, 1)
    k = _int_param("k", k, 1)
    images = validate_images(images)
    _require_canonical(images)
    n = len(images)
    m = count_valid_impostor_pairs(images)
    if k > m:
        raise ValueError(f"K={k} exceeds the number of valid impostor pairs M={m}")
    ident = [x[0] for x in images]
    rng = _rng(seed, partition_code, STREAM_IMPOSTOR)
    seen = set()
    order: List[Tuple[int, int, int]] = []
    draw = 0
    while len(order) < k:
        a = rng.integers(0, n, size=BATCH)
        b = rng.integers(0, n, size=BATCH)
        for t in range(BATCH):
            ai, bi = int(a[t]), int(b[t])
            this = draw
            draw += 1
            if ai == bi or ident[ai] == ident[bi]:
                continue
            key = (ai, bi) if ai < bi else (bi, ai)
            if key in seen:
                continue
            seen.add(key)
            order.append((key[0], key[1], this))
            if len(order) == k:
                break
    return order


def build_pairs(images: Sequence[Tuple[str, str]], seed: int = 42, partition_code: int = 1,
                cap: int = 10, rho: int = 10,
                expected: Optional[Tuple[int, int]] = None) -> PairSet:
    """Build genuine pairs, impostor pool and balanced subset.

    ``expected=(G, K)`` is an acceptance check: a mismatch raises
    ``Fold0AcceptanceError`` before the impostor pool is drawn; counts are never adjusted.
    """
    seed, partition_code, cap, rho = validate_params(seed, partition_code, cap, rho)
    imgs = sort_images(images)
    genuine = build_genuine(imgs, seed, partition_code, cap)
    g = len(genuine)
    k = rho * g
    if g == 0:
        raise ValueError("the image table yields no genuine pairs (no identity has two images)")
    if expected is not None and (g, k) != tuple(expected):
        raise Fold0AcceptanceError(f"pair counts G={g}, K={k} differ from the required G={expected[0]}, K={expected[1]}")
    pool = build_impostor_pool(imgs, seed, partition_code, k)
    rng4 = _rng(seed, partition_code, STREAM_BALANCED)
    balanced = sorted(int(i) for i in rng4.choice(k, size=g, replace=False))
    rng5 = _rng(seed, partition_code, STREAM_ORIENTATION)
    allp = list(genuine) + [(lo, hi) for lo, hi, _ in pool]
    coins = rng5.integers(0, 2, size=len(allp))
    probe = {p: bool(c) for p, c in zip(allp, coins)}
    params = {"algorithm_version": ALGORITHM_VERSION, "numpy_version": np.__version__,
              "seed": seed, "partition_code": partition_code, "cap": cap, "rho": rho,
              "n_images": len(imgs), "n_genuine": g, "n_pool": k, "n_balanced": len(balanced)}
    return PairSet(imgs, genuine, pool, balanced, probe, params)


def build_fold0_pairs(images: Sequence[Tuple[str, str]], partition_code: int) -> PairSet:
    """Fold 0 entry point: fixed seed 42, cap 10, rho 10 and the exact G/K acceptance check."""
    partition_code = _int_param("partition_code", partition_code, 1)   # strict: rejects 1.0, bool, str, None
    if partition_code not in FOLD0_EXPECTED:
        raise ValueError(f"partition_code must be one of {tuple(FOLD0_EXPECTED)}, got {partition_code!r}")
    ps = build_pairs(images, partition_code=partition_code, expected=FOLD0_EXPECTED[partition_code],
                     **FOLD0_DEFAULTS)
    check_fold0_counts(ps)
    return ps


def check_fold0_counts(ps: PairSet) -> None:
    """Post-hoc acceptance check of a built ``PairSet`` against Fold 0.

    Raises ``Fold0AcceptanceError`` (and only that) when G, K, the balanced size or any recorded
    parameter differs from Fold 0, OR when the recorded parameters themselves are not strict
    integers (a float such as ``1.0``, a bool, a string, ``None`` or an unsupported value cannot
    bypass the check through ``1.0 == 1`` / ``True == 1``). NumPy integer scalars are accepted,
    consistently with ``_int_param``.
    """
    params = getattr(ps, "params", None)
    if not isinstance(params, dict):
        raise Fold0AcceptanceError("PairSet.params is not a dict")

    def strict(key):
        try:
            return _int_param(key, params.get(key), 0)
        except ValueError as exc:
            raise Fold0AcceptanceError(f"invalid recorded parameter: {exc}") from None

    pc = strict("partition_code")
    if pc not in FOLD0_EXPECTED:
        raise Fold0AcceptanceError(f"partition code {pc!r} is not a Fold 0 partition")
    g_exp, k_exp = FOLD0_EXPECTED[pc]
    got = (len(ps.genuine), len(ps.pool), len(ps.balanced))
    if got != (g_exp, k_exp, g_exp):
        raise Fold0AcceptanceError(f"partition {pc}: G={got[0]}, K={got[1]}, balanced={got[2]}; "
                                   f"required G={g_exp}, K={k_exp}, balanced={g_exp}")
    for key, val in FOLD0_DEFAULTS.items():
        if strict(key) != val:
            raise Fold0AcceptanceError(f"parameter {key}={params.get(key)!r}; Fold 0 requires {val}")


def manifest_rows(ps: PairSet) -> List[str]:
    """Canonical rows: pair id, label, reference, probe, identities, membership, acceptance rank."""
    imgs = ps.images
    sp = ps.sorted_pool()
    bal = set(ps.balanced)
    rank = {(lo, hi): r for r, (lo, hi, _) in enumerate(ps.pool)}
    rows: List[str] = []

    def row(label, p, member, accept):
        lo, hi = p
        probe_hi = ps.probe_is_hi[p]
        ref, prb = (lo, hi) if probe_hi else (hi, lo)
        rows.append("|".join([label, f"{imgs[ref][0]}/{imgs[ref][1]}", f"{imgs[prb][0]}/{imgs[prb][1]}",
                              imgs[lo][0], imgs[hi][0], member, accept]))

    for p in ps.genuine:
        row("genuine", p, "genuine", "")
    for i, p in enumerate(sp):
        row("impostor", p, "pool+balanced" if i in bal else "pool", str(rank[p]))
    rows.sort()
    return [f"{n:08d}|{r}" for n, r in enumerate(rows)]


def manifest_bytes(ps: PairSet) -> bytes:
    """Canonical manifest: one header line (sorted ``key=value`` metadata joined by ``|``),
    then the sorted rows, UTF-8, LF line endings, one trailing LF, no BOM."""
    head = "|".join(f"{k}={ps.params[k]}" for k in sorted(ps.params))
    body = "\n".join(manifest_rows(ps)) + "\n"
    return (head + "\n" + body).encode("utf-8")


def manifest_sha256(ps: PairSet) -> str:
    """SHA-256 of ``manifest_bytes`` (covers the metadata header and every row)."""
    return hashlib.sha256(manifest_bytes(ps)).hexdigest()


def write_manifest(ps: PairSet, path: str) -> str:
    """Write the canonical bytes (binary mode, so no newline translation); return the SHA-256."""
    data = manifest_bytes(ps)
    with open(path, "xb") as fh:  # refuses to overwrite an existing manifest
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def diagnostics(ps: PairSet) -> Dict[str, float]:
    imgs = ps.images
    n_id = {}
    for ident, _ in imgs:
        n_id[ident] = n_id.get(ident, 0) + 1

    def endpoint_counts(pairs):
        cnt: Dict[str, int] = {}
        for lo, hi in pairs:
            for x in (lo, hi):
                cnt[imgs[x][0]] = cnt.get(imgs[x][0], 0) + 1
        return cnt

    def shares(pairs):
        cnt: Dict[str, int] = {}
        for lo, hi in pairs:
            for x in (lo, hi):
                cnt[imgs[x][0]] = cnt.get(imgs[x][0], 0) + 1
        tot = sum(cnt.values())
        s = sorted((v / tot for v in cnt.values()), reverse=True) if tot else []
        return s

    out: Dict[str, float] = {}
    for name, pairs in (("genuine", ps.genuine), ("pool", ps.sorted_pool()), ("balanced", ps.balanced_pairs())):
        s = shares(pairs)
        used_imgs = {x for p in pairs for x in p[:2]}
        out[f"{name}_images"] = len(used_imgs)
        out[f"{name}_identities"] = len({imgs[x][0] for x in used_imgs})
        out[f"{name}_max_share"] = s[0] if s else 0.0
        out[f"{name}_top5_share"] = sum(s[:5])
        out[f"{name}_effective_identities"] = (1.0 / sum(v * v for v in s)) if s else 0.0
        cnt = endpoint_counts(pairs)
        tot = sum(cnt.values())
        out[f"{name}_max_endpoints"] = max(cnt.values()) if cnt else 0
        out[f"{name}_total_endpoints"] = tot
        out[f"{name}_single_image_endpoint_share"] = (
            sum(v for i, v in cnt.items() if n_id[i] == 1) / tot if tot else 0.0)
    single = sum(1 for lo, hi, _ in ps.pool if n_id[imgs[lo][0]] == 1 or n_id[imgs[hi][0]] == 1)
    out["pool_frac_touching_single_image_identity"] = single / len(ps.pool) if ps.pool else 0.0
    return out
