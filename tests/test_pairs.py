"""Tests for the verification pair builder (Revision 4.1, section 3 and 3.9).

SCOPE: every population here is synthetic. The "Fold 0" tests use synthetic tables built to the
Fold 0 *shape* (image counts, multi-image identities, heaviest identity, G); they prove the
acceptance machinery works, NOT that the real Fold 0 tables give G=684/K=6840 and G=764/K=7640.
That check (``build_fold0_pairs`` on the real tables) has not been run: the real tables are not
in this repository.

All chi-square tests use fixed seeds, so each is a deterministic regression check of this
implementation (a constant p-value), not a mathematical proof of uniformity; uniformity follows from
the construction and is what these tests would catch breaking.
"""
import hashlib
import itertools
import math

import numpy as np
import pytest
from scipy.stats import chi2

from faceeval.evaluation import pairs as P


def fold0_like(total_ids=574, multi=174, big=144, seed=7):
    """Synthetic population shaped like the Fold 0 test partition (about 400
    single-image identities, 174 multi-image, one identity with 144 images)."""
    rng = np.random.default_rng(seed)
    counts = [big] + [int(c) for c in rng.integers(2, 12, size=multi - 1)]
    counts += [1] * (total_ids - multi)
    return [(f"id{i:04d}", f"img{j:03d}") for i, c in enumerate(counts) for j in range(c)]


@pytest.fixture(scope="module")
def ps():
    return P.build_pairs(fold0_like(), seed=42, partition_code=2, cap=10, rho=10)


def test_stable_id_is_sha256_prefix_not_python_hash():
    assert P.stable_id("Aaron_Eckhart") == int.from_bytes(
        __import__("hashlib").sha256(b"Aaron_Eckhart").digest()[:8], "big")


def test_counts_and_cap(ps):
    imgs = ps.images
    n = {}
    for ident, _ in imgs:
        n[ident] = n.get(ident, 0) + 1
    expected_g = sum(min(10, c * (c - 1) // 2) for c in n.values() if c >= 2)
    assert len(ps.genuine) == expected_g
    assert len(ps.pool) == 10 * expected_g
    assert len(ps.balanced) == expected_g


def test_small_identities_fully_enumerated_large_capped(ps):
    imgs = ps.images
    per = {}
    for lo, hi in ps.genuine:
        per.setdefault(imgs[lo][0], []).append((lo, hi))
    n = {}
    for ident, _ in imgs:
        n[ident] = n.get(ident, 0) + 1
    for ident, c in n.items():
        if c >= 2:
            assert len(per[ident]) == min(10, c * (c - 1) // 2)
            assert len(set(per[ident])) == len(per[ident])
        else:
            assert ident not in per
    assert len(per["id0000"]) == 10


def test_acceptance_checks_3_9(ps):
    imgs = ps.images
    pool = [(lo, hi) for lo, hi, _ in ps.pool]
    assert len(set(pool)) == len(pool)                      # no duplicate unordered pair
    assert all(lo < hi and imgs[lo][0] != imgs[hi][0] for lo, hi in pool)   # no within-identity impostor
    assert all(imgs[lo][0] == imgs[hi][0] and lo < hi for lo, hi in ps.genuine)  # no cross-identity genuine
    assert not set(pool) & set(ps.genuine)
    assert set(ps.balanced_pairs()) <= set(pool)            # balanced inside pool
    assert len(set(ps.balanced)) == len(ps.balanced)
    draws = [d for _, _, d in ps.pool]
    assert draws == sorted(draws) and len(set(draws)) == len(draws)


def test_manifest_hash_stable_and_seed_sensitive():
    imgs = fold0_like()
    a = P.manifest_sha256(P.build_pairs(imgs, seed=42, partition_code=1))
    b = P.manifest_sha256(P.build_pairs(list(reversed(imgs)), seed=42, partition_code=1))
    c = P.manifest_sha256(P.build_pairs(imgs, seed=43, partition_code=1))
    d = P.manifest_sha256(P.build_pairs(imgs, seed=42, partition_code=2))
    assert a == b            # input order does not matter, the canonical sort does
    assert len({a, c, d}) == 3


def test_prefix_property():
    imgs = P.sort_images(fold0_like())
    big = P.build_impostor_pool(imgs, 42, 1, 3000)
    small = P.build_impostor_pool(imgs, 42, 1, 1200)
    assert big[:1200] == small


def test_k_larger_than_valid_pairs_raises():
    imgs = P.sort_images([("a", "1"), ("a", "2"), ("b", "1")])
    assert P.count_valid_impostor_pairs(imgs) == 2
    with pytest.raises(ValueError):
        P.build_impostor_pool(imgs, 1, 1, 3)


def test_duplicate_rows_rejected():
    with pytest.raises(ValueError):
        P.sort_images([("a", "1"), ("a", "1")])


def test_orientation_is_roughly_fair(ps):
    frac = np.mean(list(ps.probe_is_hi.values()))
    assert 0.45 < frac < 0.55


def _uniformity_samples(monkeypatch, n):
    monkeypatch.setattr(P, "BATCH", 32)   # forces many rejections and batch refills
    imgs = P.sort_images([(f"i{a}", f"{b}") for a in range(4) for b in range(2)])
    valid = [p for p in itertools.combinations(range(8), 2) if imgs[p[0]][0] != imgs[p[1]][0]]
    assert len(valid) == 24
    cells = {t: 0 for t in itertools.permutations(valid, 3)}
    for s in range(n):                      # seeds 0..n-1: fully deterministic
        pool = P.build_impostor_pool(imgs, s, 1, 3)
        cells[tuple((lo, hi) for lo, hi, _ in pool)] += 1
    return cells


def _chi2(cells, n):
    exp = n / len(cells)
    return sum((c - exp) ** 2 / exp for c in cells.values())


def test_impostor_uniformity_chi_square_on_tiny_population(monkeypatch):
    """Section 3.5: 8 images, 4 identities, K=3. Every ordered 3-tuple of distinct valid
    pairs (24*23*22 = 12,144 cells, expected 8.2 per cell, so the chi-square approximation
    holds) must be equally likely over 100,000 fixed seeds 0..99,999.

    A fixed-seed p-value is a regression check on this implementation, not a proof of uniformity;
    uniformity follows from the construction (uniform i.i.d. draws, rejection of invalid ones, so
    every valid ordered K-tuple is equally likely) and this test would catch gross breakage.

    Calibration: seeds are fixed, so the p-value is a constant, not a random variable;
    this run gives chi2 = 11,961.9 on 12,143 df, p = 0.878 (about 5 s). alpha = 1e-3 bounds the false-alarm rate of a *different* correct sampler at
    0.1 %. Power (measured, see the next test): order-destroyed samples and a sampler that
    under-weights one pair by 50 % fail decisively; a 20 % under-weight gives p = 0.004 and a
    10 % under-weight is NOT detectable (p = 0.29) at this sample size.
    """
    n = 100_000
    cells = _uniformity_samples(monkeypatch, n)
    stat = _chi2(cells, n)
    p = chi2.sf(stat, len(cells) - 1)
    assert len(cells) == 12_144 and sum(cells.values()) == n
    assert n / len(cells) > 5                                # chi-square approximation conditions
    assert p > 1e-3, (stat, p)
    # power check on the same samples: collapsing order (sorting each tuple) is a different,
    # non-uniform-over-ordered-tuples distribution and must be rejected decisively
    collapsed = {t: 0 for t in cells}
    for t, c in cells.items():
        collapsed[tuple(sorted(t))] += c
    assert chi2.sf(_chi2(collapsed, n), len(cells) - 1) < 1e-10


def test_chi_square_has_power_against_an_incorrect_sampler(monkeypatch):
    """The same statistic must reject a deliberately wrong rejection sampler that accepts the
    first valid pair only half as often (everything else identical)."""
    monkeypatch.setattr(P, "BATCH", 32)
    imgs = P.sort_images([(f"i{a}", f"{b}") for a in range(4) for b in range(2)])
    valid = [p for p in itertools.combinations(range(8), 2) if imgs[p[0]][0] != imgs[p[1]][0]]
    cells = {t: 0 for t in itertools.permutations(valid, 3)}

    def wrong(seed):
        rng = np.random.default_rng(seed + 10**6)
        out = []
        while len(out) < 3:
            a, b = (int(x) for x in rng.integers(0, 8, 2))
            if a == b or imgs[a][0] == imgs[b][0]:
                continue
            k = (min(a, b), max(a, b))
            if k in out or (k == valid[0] and rng.random() > 0.5):
                continue
            out.append(k)
        return tuple(out)

    n = 100_000
    for s in range(n):
        cells[wrong(s)] += 1
    assert chi2.sf(_chi2(cells, n), len(cells) - 1) < 1e-10


def test_balanced_subset_is_uniform_over_all_g_subsets_of_the_pool():
    """Stream 4: for G=3, K=6 every one of the C(6,3)=20 index subsets is equally likely."""
    imgs = [("a", "1"), ("a", "2"), ("a", "3")] + [(f"s{i}", "1") for i in range(8)]
    counts = {}
    n = 6000
    for s in range(n):
        ps_ = P.build_pairs(imgs, seed=s, partition_code=1, cap=10, rho=2)
        assert len(ps_.genuine) == 3 and len(ps_.pool) == 6
        counts[tuple(ps_.balanced)] = counts.get(tuple(ps_.balanced), 0) + 1
    assert len(counts) == 20
    exp = n / 20
    stat = sum((c - exp) ** 2 / exp for c in counts.values())
    assert chi2.sf(stat, 19) > 1e-3, stat


def test_orientation_matches_independent_stream_5_and_is_independent_of_membership(small):
    allp = list(small.genuine) + [(lo, hi) for lo, hi, _ in small.pool]
    coins = P._rng(42, 2, P.STREAM_ORIENTATION).integers(0, 2, size=len(allp))
    assert [small.probe_is_hi[p] for p in allp] == [bool(c) for c in coins]
    # 2x2: balanced membership vs orientation among impostors (fixed data, alpha 1e-3)
    sp = small.sorted_pool()
    bal = set(small.balanced)
    tab = np.zeros((2, 2))
    for i, p in enumerate(sp):
        tab[int(i in bal), int(small.probe_is_hi[p])] += 1
    from scipy.stats import chi2_contingency
    assert chi2_contingency(tab)[1] > 1e-3, tab


STREAMS = {"genuine": P.STREAM_GENUINE, "impostor": P.STREAM_IMPOSTOR,
           "balanced": P.STREAM_BALANCED, "orientation": P.STREAM_ORIENTATION}


def test_stream_ids_and_seed_sequence_key_are_as_specified():
    """Implementation contract (Revision 4.1 section 3): stream IDs 1/3/4/5 and the generator key
    SeedSequence([seed, partition_code, stream_id, *extra]) feeding PCG64."""
    assert STREAMS == {"genuine": 1, "impostor": 3, "balanced": 4, "orientation": 5}
    for seed, pc, st, extra in [(42, 1, 3, ()), (42, 2, 1, (12345,)), (7, 2, 5, ())]:
        want = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, pc, st, *extra])))
        got = P._rng(seed, pc, st, *extra)
        assert (got.integers(0, 2**62, size=8) == want.integers(0, 2**62, size=8)).all()


def test_each_stream_changes_with_seed_and_with_partition_code_for_these_values():
    """EMPIRICAL, not a universal guarantee: SeedSequence maps distinct entropy lists to distinct
    states with overwhelming probability but not provably for every pair of inputs. For the specific
    values below each of the four streams produces different raw draws."""
    draw = lambda seed, pc, st: tuple(P._rng(seed, pc, st).integers(0, 2**62, size=6))
    for name, st in STREAMS.items():
        assert draw(42, 1, st) != draw(43, 1, st), name          # seed matters
        assert draw(42, 1, st) != draw(42, 2, st), name          # partition code matters
    assert len({draw(42, 1, st) for st in STREAMS.values()}) == 4   # the four streams differ for seed 42, partition 1


def test_genuine_stream_depends_only_on_seed_partition_cap_and_the_identity_itself():
    one = [("keep", str(i)) for i in range(9)]
    other = [("aaa", str(i)) for i in range(7)] + [("zzz", str(i)) for i in range(9)]
    a, b = P.sort_images(one), P.sort_images(one + other)
    name = lambda t, p: (t[p[0]], t[p[1]])
    pa = [name(a, p) for p in P.build_genuine(a, 42, 1, 10)]
    pb = [name(b, p) for p in P.build_genuine(b, 42, 1, 10) if b[p[0]][0] == "keep"]
    assert pa == pb                                                       # other identities are irrelevant
    assert [name(a, p) for p in P.build_genuine(a, 42, 1, 10)] == pa     # repeatable


def test_impostor_stream_depends_only_on_seed_partition_and_image_table():
    imgs = fold0_like(total_ids=80, multi=30, big=20)
    base = P.build_pairs(imgs, seed=42, partition_code=2, cap=10, rho=10)
    for kw in ({"cap": 3}, {"cap": 20}, {"rho": 4}):                      # cap and rho do not touch stream 3
        other = P.build_pairs(imgs, seed=42, partition_code=2, **{"cap": 10, "rho": 10, **kw})
        k = min(len(base.pool), len(other.pool))
        assert base.pool[:k] == other.pool[:k], kw


def test_balanced_stream_depends_only_on_seed_partition_K_and_G_not_on_the_images():
    """Stream 4 draws G of K positions; two different image tables with the same G and K get the
    same positions (the positions index each table's own sorted pool)."""
    t1 = [("a", "1"), ("a", "2"), ("a", "3")] + [(f"s{i}", "1") for i in range(8)]
    t2 = [("x", "1"), ("x", "2"), ("x", "3")] + [(f"q{i}", "z") for i in range(20)]
    p1 = P.build_pairs(t1, seed=11, partition_code=1, rho=5)
    p2 = P.build_pairs(t2, seed=11, partition_code=1, rho=5)
    assert (len(p1.genuine), len(p1.pool)) == (len(p2.genuine), len(p2.pool)) == (3, 15)
    assert p1.balanced == p2.balanced and p1.pool != p2.pool


def test_orientation_stream_depends_only_on_seed_partition_and_position():
    """Stream 5 yields one coin per pair in the order genuine pairs then pool (acceptance) order, so
    the coin sequences of two different tables agree on their common prefix."""
    t1 = fold0_like(total_ids=60, multi=20, big=12)
    t2 = fold0_like(total_ids=90, multi=40, big=15, seed=3)
    p1 = P.build_pairs(t1, seed=9, partition_code=2)
    p2 = P.build_pairs(t2, seed=9, partition_code=2, cap=4, rho=3)
    seq = lambda ps_: [ps_.probe_is_hi[q] for q in list(ps_.genuine) + [(lo, hi) for lo, hi, _ in ps_.pool]]
    s1, s2 = seq(p1), seq(p2)
    n = min(len(s1), len(s2))
    assert s1[:n] == s2[:n] and n > 50
    ref = P._rng(9, 2, P.STREAM_ORIENTATION).integers(0, 2, size=n)
    assert s1[:n] == [bool(c) for c in ref]


def test_rho_changes_leave_genuine_pairs_and_genuine_orientation_unchanged():
    imgs = fold0_like(total_ids=80, multi=30, big=20)
    a = P.build_pairs(imgs, seed=42, partition_code=2, rho=10)
    b = P.build_pairs(imgs, seed=42, partition_code=2, rho=4)
    assert a.genuine == b.genuine
    assert all(a.probe_is_hi[q] == b.probe_is_hi[q] for q in a.genuine)


# ---------------------------------------------------------------- build_genuine
def _reference_genuine(imgs, seed, partition_code, cap):
    """The original enumerating implementation, kept as an oracle."""
    by_id = {}
    for idx, (ident, _) in enumerate(imgs):
        by_id.setdefault(ident, []).append(idx)
    out = []
    for ident in sorted(by_id):
        idxs = by_id[ident]
        if len(idxs) < 2:
            continue
        allp = list(itertools.combinations(idxs, 2))
        if len(allp) <= cap:
            out.extend(allp)
        else:
            rng = P._rng(seed, partition_code, P.STREAM_GENUINE, P.stable_id(ident))
            ranks = sorted(int(r) for r in rng.choice(len(allp), size=cap, replace=False))
            out.extend(allp[r] for r in ranks)
    return out


@pytest.mark.parametrize("seed,pc,cap", [(42, 1, 10), (42, 2, 10), (7, 2, 3), (0, 1, 1), (2**40, 1, 25)])
def test_build_genuine_identical_to_enumeration_oracle(seed, pc, cap):
    imgs = P.sort_images(fold0_like())
    assert P.build_genuine(imgs, seed, pc, cap) == _reference_genuine(imgs, seed, pc, cap)


@pytest.mark.parametrize("n", range(2, 30))
def test_pair_from_rank_matches_combinations_exactly(n):
    allp = list(itertools.combinations(range(n), 2))
    assert [P._pair_from_rank(n, r) for r in range(len(allp))] == allp


def test_build_genuine_does_not_enumerate_large_identities(monkeypatch):
    """One identity with 100,000 images has C(n,2) ~ 5e9 pairs. ``itertools.combinations`` is
    replaced by a function that raises, so any enumeration fails the test; the behavioural
    assertions (10 distinct, rank-ordered, in-range pairs) are the actual evidence.

    What the measurements include: only the ``build_genuine`` call (its internal validation copy of
    the 100,000-row table, the sampling and the unranking). What they exclude: construction of the
    input list, which happens BEFORE ``tracemalloc.start()``, so the peak-memory figure does not
    include the input list itself. Time and memory are printed (run with ``-s``) and bounded only by
    deliberately generous limits; a bound is a smoke check, not a proof of algorithmic complexity.
    """
    n = 100_000
    imgs = [("big", f"{i:06d}") for i in range(n)]                  # built before tracing starts
    monkeypatch.setattr(P.itertools, "combinations", lambda *a, **k: (_ for _ in ()).throw(AssertionError("enumerated")))
    import time, tracemalloc
    tracemalloc.start()
    t = time.perf_counter()
    out = P.build_genuine(imgs, 42, 1, 10)
    dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"\nPERF build_genuine n={n}: {dt:.3f} s, tracemalloc peak {peak / 1e6:.2f} MB (input list excluded)")
    assert len(out) == 10 and len(set(out)) == 10
    assert all(0 <= a < b < n for a, b in out)
    rank = lambda a, b: a * (2 * n - a - 1) // 2 + (b - a - 1)
    ranks = [rank(a, b) for a, b in out]
    assert ranks == sorted(ranks)                                   # reported in lexicographic rank order
    assert dt < 10.0 and peak < 200_000_000, (dt, peak)             # generous smoke bounds only


def test_build_genuine_cap_sampling_is_uniform_without_replacement():
    """5 images -> 10 pairs, cap 3: all C(10,3)=120 subsets equally likely over 24,000 seeds."""
    imgs = [("a", str(i)) for i in range(5)]
    counts = {}
    n = 24_000
    for s in range(n):
        key = tuple(P.build_genuine(imgs, s, 1, 3))
        assert len(set(key)) == 3
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 120
    exp = n / 120
    stat = sum((c - exp) ** 2 / exp for c in counts.values())
    assert chi2.sf(stat, 119) > 1e-3, stat


# ---------------------------------------------------------------- helper validation
@pytest.mark.parametrize("kw", [
    {"cap": 0}, {"cap": -1}, {"cap": True}, {"cap": 2.0}, {"cap": "3"}, {"cap": None},
    {"seed": -1}, {"seed": False}, {"seed": 1.0}, {"partition_code": 0}, {"partition_code": 1.0},
    {"partition_code": True}, {"partition_code": 5},
])
def test_build_genuine_rejects_invalid_parameters(kw):
    args = dict(seed=42, partition_code=1, cap=10)
    args.update(kw)
    with pytest.raises(ValueError):
        P.build_genuine(P.sort_images(fold0_like(total_ids=20, multi=8, big=6)), **args)


@pytest.mark.parametrize("kw", [
    {"k": 0}, {"k": -5}, {"k": True}, {"k": 2.0}, {"k": "5"}, {"k": None},
    {"seed": -1}, {"seed": True}, {"seed": 1.5}, {"partition_code": 0}, {"partition_code": False},
])
def test_build_impostor_pool_rejects_invalid_parameters(kw):
    args = dict(seed=42, partition_code=1, k=5)
    args.update(kw)
    with pytest.raises(ValueError):
        P.build_impostor_pool(P.sort_images(fold0_like(total_ids=20, multi=8, big=6)), **args)


def test_helpers_reject_unsorted_or_duplicate_or_malformed_images():
    good = P.sort_images(fold0_like(total_ids=20, multi=8, big=6))
    for f in (lambda i: P.build_genuine(i, 42, 1, 10), lambda i: P.build_impostor_pool(i, 42, 1, 5)):
        with pytest.raises(ValueError):
            f(list(reversed(good)))
        with pytest.raises(ValueError):
            f(good + [good[0]])
        with pytest.raises(ValueError):
            f([])
        with pytest.raises(ValueError):
            f([("a", 1), ("a", 2)])


def test_sorted_duplicate_records_rejected_by_name_not_by_ordering():
    """Inputs that ARE in canonical sorted order but contain a repeated record must be rejected for
    the duplicate, so these cases cannot pass merely because of an ordering error."""
    dup_adjacent = [("a", "1"), ("a", "1"), ("a", "2"), ("b", "1")]
    dup_tail = [("a", "1"), ("a", "2"), ("b", "1"), ("b", "1")]
    for table in (dup_adjacent, dup_tail):
        assert table == sorted(table)                                   # genuinely sorted
        with pytest.raises(ValueError, match="duplicate"):
            P.build_genuine(table, 42, 1, 10)
        with pytest.raises(ValueError, match="duplicate"):
            P.build_impostor_pool(table, 42, 1, 1)
        with pytest.raises(ValueError, match="duplicate"):
            P.count_valid_impostor_pairs(table)
        with pytest.raises(ValueError, match="duplicate"):
            P.sort_images(table)
    unsorted_unique = [("b", "1"), ("a", "1"), ("a", "2")]
    for f in (lambda: P.build_genuine(unsorted_unique, 42, 1, 10), lambda: P.build_impostor_pool(unsorted_unique, 42, 1, 1),
              lambda: P.count_valid_impostor_pairs(unsorted_unique)):
        with pytest.raises(ValueError, match="sorted"):
            f()


@pytest.mark.parametrize("entry", ["sort_images", "build_genuine", "count_valid_impostor_pairs",
                                   "build_impostor_pool", "build_pairs", "build_fold0_pairs"])
def test_every_public_entry_point_rejects_a_sorted_duplicate_record(entry):
    table = [("a", "1"), ("a", "1"), ("a", "2"), ("b", "1"), ("b", "2")]
    assert table == sorted(table)
    calls = {
        "sort_images": lambda: P.sort_images(table),
        "build_genuine": lambda: P.build_genuine(table, 42, 1, 10),
        "count_valid_impostor_pairs": lambda: P.count_valid_impostor_pairs(table),
        "build_impostor_pool": lambda: P.build_impostor_pool(table, 42, 1, 1),
        "build_pairs": lambda: P.build_pairs(table),
        "build_fold0_pairs": lambda: P.build_fold0_pairs(table, 1),
    }
    with pytest.raises(ValueError, match="duplicate"):
        calls[entry]()


def test_numpy_integers_accepted_by_helpers():
    imgs = P.sort_images(fold0_like(total_ids=20, multi=8, big=6))
    assert P.build_genuine(imgs, np.int64(42), np.int8(1), np.int32(10)) == P.build_genuine(imgs, 42, 1, 10)
    assert P.build_impostor_pool(imgs, np.int64(42), 1, np.int64(5)) == P.build_impostor_pool(imgs, 42, 1, 5)


def test_sort_images_handles_large_tables_and_reports_duplicates():
    """Behaviour test on 300,000 rows (a quadratic pass over this table would not finish in the
    time limit). Measured: wall time of ``sort_images`` only, input construction excluded. The time
    bound is a generous smoke check, not a complexity proof."""
    import time
    n = 300_000
    rows = [(f"id{i // 3:06d}", f"im{i % 3}") for i in range(n)]
    t = time.perf_counter()
    P.sort_images(rows)
    ok_time = time.perf_counter() - t
    dup = rows + [rows[17]]
    t = time.perf_counter()
    with pytest.raises(ValueError, match="duplicate"):
        P.sort_images(dup)
    bad_time = time.perf_counter() - t
    print(f"\nPERF sort_images n={n}: ok {ok_time:.3f} s, duplicate {bad_time:.3f} s")
    assert ok_time < 30 and bad_time < 30, (ok_time, bad_time)


def test_cross_batch_duplicates_are_rejected(monkeypatch):
    """Duplicate check must span batches (assumption A4)."""
    monkeypatch.setattr(P, "BATCH", 4)
    imgs = P.sort_images([(f"i{a}", f"{b}") for a in range(3) for b in range(2)])
    pool = P.build_impostor_pool(imgs, 5, 1, P.count_valid_impostor_pairs(imgs))
    keys = [(lo, hi) for lo, hi, _ in pool]
    assert len(set(keys)) == len(keys) == 12


# ---------------------------------------------------------------- Fold 0 populations
# Synthetic tables with exactly the Revision 4.1 Fold 0 shape (image counts, multi-image
# identities, heaviest identity, G). The real Fold 0 image tables are not in this repository;
# the acceptance functions below are what must be run on them.
#   size -> number of identities
TEST_SHAPE = {144: 1, 10: 8, 9: 52, 4: 1, 3: 18, 2: 94, 1: 400}     # 1,338 images, G = 764
VAL_SHAPE = {121: 1, 10: 8, 9: 47, 4: 2, 2: 112, 1: 403}            # 1,259 images, G = 684


def population(shape, prefix="p"):
    out, i = [], 0
    for size, count in sorted(shape.items(), reverse=True):
        for _ in range(count):
            out.extend((f"{prefix}{i:04d}", f"im{j:03d}") for j in range(size))
            i += 1
    return out


@pytest.fixture(scope="module")
def val_pop():
    return population(VAL_SHAPE, "v")


@pytest.fixture(scope="module")
def test_pop():
    return population(TEST_SHAPE, "t")


@pytest.fixture(scope="module")
def fold0_val(val_pop):
    return P.build_fold0_pairs(val_pop, 1)


@pytest.fixture(scope="module")
def fold0_test(test_pop):
    return P.build_fold0_pairs(test_pop, 2)


def test_synthetic_populations_have_fold0_shape(val_pop, test_pop):
    assert (len(val_pop), len(test_pop)) == (1259, 1338)
    for shape, pop in ((VAL_SHAPE, val_pop), (TEST_SHAPE, test_pop)):
        assert sum(k * v for k, v in shape.items()) == len(pop)


def test_fold0_acceptance_validation(fold0_val):
    assert (len(fold0_val.genuine), len(fold0_val.pool), len(fold0_val.balanced)) == (684, 6840, 684)
    P.check_fold0_counts(fold0_val)


def test_fold0_acceptance_test(fold0_test):
    assert (len(fold0_test.genuine), len(fold0_test.pool), len(fold0_test.balanced)) == (764, 7640, 764)
    P.check_fold0_counts(fold0_test)


def test_fold0_mismatch_stops_before_any_pool_is_drawn(monkeypatch, val_pop):
    drawn = []
    monkeypatch.setattr(P, "build_impostor_pool", lambda *a, **k: drawn.append(1))
    one_more = val_pop + [("v0999", "a"), ("v0999", "b")]          # G becomes 685
    with pytest.raises(P.Fold0AcceptanceError, match="G=685, K=6850"):
        P.build_fold0_pairs(one_more, 1)
    with pytest.raises(P.Fold0AcceptanceError):
        P.build_fold0_pairs(val_pop, 2)                              # validation table, test expectation
    assert drawn == []                                               # stopped; nothing adjusted or drawn


def test_fold0_post_hoc_check_rejects_edited_sets_and_parameters(fold0_val):
    import copy
    bad = copy.deepcopy(fold0_val)
    bad.balanced.pop()
    with pytest.raises(P.Fold0AcceptanceError):
        P.check_fold0_counts(bad)
    bad = copy.deepcopy(fold0_val)
    bad.params["seed"] = 43
    with pytest.raises(P.Fold0AcceptanceError, match="seed"):
        P.check_fold0_counts(bad)
    bad = copy.deepcopy(fold0_val)
    bad.params["partition_code"] = 2
    with pytest.raises(P.Fold0AcceptanceError):
        P.check_fold0_counts(bad)


INVALID_POSTHOC = [1.0, 2.0, np.float64(1.0), np.float32(1.0), True, False, "1", b"1", None, -1, 0, 3, 2**70,
                   [1], {}, (1,), object()]


@pytest.mark.parametrize("bad_pc", INVALID_POSTHOC)
def test_check_fold0_counts_rejects_invalid_partition_code_metadata(fold0_val, bad_pc):
    """A PairSet whose recorded partition_code is not a strict integer cannot pass, even when the
    value compares equal to a valid code (1.0 == 1, True == 1). Only Fold0AcceptanceError is raised."""
    import copy
    bad = copy.deepcopy(fold0_val)
    bad.params["partition_code"] = bad_pc
    with pytest.raises(P.Fold0AcceptanceError):
        P.check_fold0_counts(bad)


@pytest.mark.parametrize("key,bad_val", [("seed", 42.0), ("seed", np.float64(42.0)), ("seed", True), ("seed", "42"),
                                          ("cap", 10.0), ("cap", True), ("rho", 10.0), ("rho", "10"), ("rho", None)])
def test_check_fold0_counts_rejects_non_integer_seed_cap_rho(fold0_val, key, bad_val):
    import copy
    bad = copy.deepcopy(fold0_val)
    bad.params[key] = bad_val
    with pytest.raises(P.Fold0AcceptanceError):
        P.check_fold0_counts(bad)


def test_check_fold0_counts_rejects_missing_params_and_accepts_valid_forms(fold0_val, fold0_test):
    import copy
    for key in ("partition_code", "seed", "cap", "rho"):
        bad = copy.deepcopy(fold0_val)
        del bad.params[key]
        with pytest.raises(P.Fold0AcceptanceError):
            P.check_fold0_counts(bad)
    bad = copy.deepcopy(fold0_val)
    bad.params = "not a dict"
    with pytest.raises(P.Fold0AcceptanceError):
        P.check_fold0_counts(bad)
    P.check_fold0_counts(fold0_val)
    P.check_fold0_counts(fold0_test)
    ok = copy.deepcopy(fold0_val)
    ok.params.update(partition_code=np.int64(1), seed=np.int32(42), cap=np.int64(10), rho=np.int16(10))
    P.check_fold0_counts(ok)                                               # NumPy integers accepted, like _int_param


def test_fold0_entry_point_rejects_bad_partition(val_pop):
    for bad in (0, 3, -1, 2**70, 1.0, 2.0, np.float64(1.0), np.float64(2.0), np.float32(1.0), True, False,
                "1", "2", b"1", None, [1], [], {}, {1}, (1,), object()):
        with pytest.raises(ValueError):           # ValueError for all of them, never TypeError
            P.build_fold0_pairs(val_pop, bad)
    # genuine integers (Python and NumPy) are accepted
    assert P.build_fold0_pairs(val_pop, np.int64(1)).params["partition_code"] == 1


# ---------------------------------------------------------------- strict validation
@pytest.mark.parametrize("kw", [
    {"cap": 0}, {"cap": -1}, {"cap": 2.5}, {"cap": "10"}, {"cap": True}, {"cap": None},
    {"rho": 0}, {"rho": -3}, {"rho": 1.0}, {"rho": "10"}, {"rho": False},
    {"seed": -1}, {"seed": 1.5}, {"seed": "42"}, {"seed": None}, {"seed": 2**63}, {"seed": True},
    {"partition_code": 0}, {"partition_code": 3}, {"partition_code": "1"}, {"partition_code": 1.0},
    {"partition_code": True},
])
def test_invalid_scalar_parameters_rejected(kw):
    with pytest.raises(ValueError):
        P.build_pairs(fold0_like(), **kw)


def test_numpy_integer_parameters_are_accepted_and_equal_python_ints():
    imgs = fold0_like(total_ids=60, multi=20, big=12)
    a = P.manifest_sha256(P.build_pairs(imgs, seed=np.int64(42), partition_code=np.int32(1), cap=np.int64(10), rho=np.int64(10)))
    b = P.manifest_sha256(P.build_pairs(imgs, seed=42, partition_code=1, cap=10, rho=10))
    assert a == b


@pytest.mark.parametrize("table", [
    None, [], (), "ab", {"a": "1"}, [("a", "1", "x")], [("a",)], ["ab"], [("a", 1)], [(1, "a")],
    [("", "1")], [("a", "")], [("a", None)], [(b"a", "1")], [("a|b", "1")], [("a", "1\n2")],
    [("a", "1\r")], [(" a", "1")], [("a", "1 ")], [("a", "\x00")],
])
def test_invalid_image_tables_rejected(table):
    with pytest.raises(ValueError):
        P.build_pairs(table)


def test_duplicate_image_records_rejected_even_if_nonadjacent():
    with pytest.raises(ValueError, match="duplicate"):
        P.build_pairs([("a", "1"), ("b", "1"), ("a", "2"), ("a", "1")])


def test_same_image_id_under_different_identities_is_allowed():
    P.sort_images([("a", "1"), ("b", "1")])


def test_table_without_genuine_pairs_rejected():
    with pytest.raises(ValueError, match="no genuine"):
        P.build_pairs([("a", "1"), ("b", "1"), ("c", "1")])


def test_input_table_is_not_mutated():
    imgs = [("b", "2"), ("a", "2"), ("a", "1"), ("b", "1"), ("c", "1")]
    before = list(imgs)
    P.build_pairs(imgs, rho=1)
    assert imgs == before


# ---------------------------------------------------------------- manifest contract
def parse(ps):
    rows = []
    for line in P.manifest_rows(ps):
        f = line.split("|")
        assert len(f) == 8, line
        rows.append(dict(id=f[0], label=f[1], ref=f[2], probe=f[3], id_lo=f[4], id_hi=f[5],
                         member=f[6], rank=f[7]))
    return rows


@pytest.fixture(scope="module")
def small():
    return P.build_pairs(fold0_like(total_ids=80, multi=30, big=20), seed=42, partition_code=2)


def test_metadata_header_fields_and_values(small):
    raw = P.manifest_bytes(small).decode("utf-8")
    head = raw.split("\n", 1)[0]
    kv = dict(x.split("=", 1) for x in head.split("|"))
    assert kv["algorithm_version"] == P.ALGORITHM_VERSION
    assert kv["numpy_version"] == np.__version__
    assert (kv["seed"], kv["partition_code"], kv["cap"], kv["rho"]) == ("42", "2", "10", "10")
    assert kv["n_genuine"] == str(len(small.genuine)) and kv["n_pool"] == str(len(small.pool))
    assert kv["n_balanced"] == str(len(small.balanced)) and kv["n_images"] == str(len(small.images))
    assert head == "|".join(sorted(head.split("|")))          # keys sorted: canonical


def test_hash_covers_every_metadata_field_and_every_row(small, monkeypatch):
    base = P.manifest_sha256(small)
    import copy
    for key, new in (("algorithm_version", "other"), ("numpy_version", "0.0.0"), ("seed", 1),
                     ("partition_code", 1), ("cap", 9), ("rho", 9)):
        m = copy.deepcopy(small)
        m.params[key] = new
        assert P.manifest_sha256(m) != base, key
    monkeypatch.setattr(P, "ALGORITHM_VERSION", "faceeval-pairs/r4.1/999")
    assert P.manifest_sha256(P.build_pairs(small.images, seed=42, partition_code=2)) != base
    monkeypatch.undo()
    monkeypatch.setattr(np, "__version__", "0.0.0-test")
    assert P.manifest_sha256(P.build_pairs(small.images, seed=42, partition_code=2)) != base
    monkeypatch.undo()
    # every row participates: flip one orientation, one membership, one rank
    m = copy.deepcopy(small)
    k = next(iter(m.probe_is_hi))
    m.probe_is_hi[k] = not m.probe_is_hi[k]
    assert P.manifest_sha256(m) != base
    m = copy.deepcopy(small)
    m.balanced[0] = next(i for i in range(len(m.pool)) if i not in set(m.balanced))
    assert P.manifest_sha256(m) != base
    m = copy.deepcopy(small)
    m.pool[0], m.pool[1] = m.pool[1], m.pool[0]               # swaps two acceptance ranks
    assert P.manifest_sha256(m) != base


def test_pair_ids_are_sequential_unique_and_stable_across_builds_and_input_order(small):
    ids = [r["id"] for r in parse(small)]
    n = len(small.genuine) + len(small.pool)
    assert ids == [f"{i:08d}" for i in range(n)]
    again = P.build_pairs(list(reversed(small.images)), seed=42, partition_code=2)
    assert P.manifest_rows(again) == P.manifest_rows(small)


def test_label_and_membership_contract(small):
    rows = parse(small)
    gen = [r for r in rows if r["label"] == "genuine"]
    imp = [r for r in rows if r["label"] == "impostor"]
    assert len(gen) == len(small.genuine) and len(imp) == len(small.pool)
    assert {r["member"] for r in gen} == {"genuine"} and {r["rank"] for r in gen} == {""}
    assert all(r["id_lo"] == r["id_hi"] for r in gen)
    assert all(r["id_lo"] != r["id_hi"] for r in imp)
    assert {r["member"] for r in imp} <= {"pool", "pool+balanced"}
    assert sum(r["member"] == "pool+balanced" for r in imp) == len(small.balanced) == len(small.genuine)
    # balanced rows are exactly the pairs selected by the balanced indices of the sorted pool
    sp = small.sorted_pool()
    img = small.images
    name = lambda x: f"{img[x][0]}/{img[x][1]}"
    bal_pairs = {frozenset((name(lo), name(hi))) for lo, hi in (sp[i] for i in small.balanced)}
    got = {frozenset((r["ref"], r["probe"])) for r in imp if r["member"] == "pool+balanced"}
    assert got == bal_pairs


def test_reference_probe_orientation_contract(small):
    img = small.images
    name = lambda x: f"{img[x][0]}/{img[x][1]}"
    by_pair = {}
    for r in parse(small):
        assert r["ref"] != r["probe"]
        by_pair[frozenset((r["ref"], r["probe"]))] = r
    pairs = list(small.genuine) + [(lo, hi) for lo, hi, _ in small.pool]
    assert len(by_pair) == len(pairs)                       # no pair appears twice, none lost
    for lo, hi in pairs:
        r = by_pair[frozenset((name(lo), name(hi)))]
        want_ref, want_probe = (lo, hi) if small.probe_is_hi[(lo, hi)] else (hi, lo)
        assert (r["ref"], r["probe"]) == (name(want_ref), name(want_probe))
    # orientation comes from stream 5 only: rebuilding gives identical orientation
    again = P.build_pairs(img, seed=42, partition_code=2)
    assert again.probe_is_hi == small.probe_is_hi


def test_impostor_acceptance_ranks_are_a_permutation_in_draw_order(small):
    imp = [r for r in parse(small) if r["label"] == "impostor"]
    ranks = sorted(int(r["rank"]) for r in imp)
    assert ranks == list(range(len(small.pool)))
    img = small.images
    name = lambda x: f"{img[x][0]}/{img[x][1]}"
    by_pair = {frozenset((r["ref"], r["probe"])): int(r["rank"]) for r in imp}
    for rank, (lo, hi, draw) in enumerate(small.pool):
        assert by_pair[frozenset((name(lo), name(hi)))] == rank
    draws = [d for _, _, d in small.pool]
    assert draws == sorted(draws)
    # rank 0..K-1 of the manifest equals an independent run of the sampler
    ref = P.build_impostor_pool(small.images, 42, 2, len(small.pool))
    assert ref == small.pool


def test_rows_are_sorted_and_serialization_is_utf8_lf(small, tmp_path):
    data = P.manifest_bytes(small)
    assert isinstance(data, bytes) and data.endswith(b"\n") and not data.endswith(b"\n\n")
    assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8")
    body = text.split("\n")[1:-1]
    assert [b.split("|", 1)[1] for b in body] == sorted(b.split("|", 1)[1] for b in body)
    assert len(body) == len(small.genuine) + len(small.pool)
    path = tmp_path / "m.tsv"
    sha = P.write_manifest(small, str(path))
    assert path.read_bytes() == data and sha == hashlib.sha256(data).hexdigest() == P.manifest_sha256(small)
    with pytest.raises(FileExistsError):
        P.write_manifest(small, str(path))                  # never overwrites


def test_non_ascii_identities_serialize_as_utf8():
    imgs = [("José_García", "a"), ("José_García", "b"), ("Zoë", "a"), ("Zoë", "b"), ("Åsa", "a")]
    ps_ = P.build_pairs(imgs, rho=1)
    data = P.manifest_bytes(ps_)
    assert "José_García/a".encode("utf-8") in data and b"\r" not in data
    assert P.manifest_sha256(ps_) == hashlib.sha256(data).hexdigest()


def test_hash_stable_across_repeated_builds(small):
    h = {P.manifest_sha256(P.build_pairs(small.images, seed=42, partition_code=2)) for _ in range(3)}
    assert h == {P.manifest_sha256(small)}


def test_fold0_manifest_hash_repeatable(fold0_test, test_pop):
    again = P.build_fold0_pairs(list(reversed(test_pop)), 2)
    assert P.manifest_sha256(again) == P.manifest_sha256(fold0_test)


# ---------------------------------------------------------------- diagnostics vs Revision 4.1 section 3.8
def _analytic_endpoint_share(shape, size):
    """Expected share of impostor endpoints held by one identity of ``size`` images:
    n_i (N - n_i) / (N^2 - sum n_j^2), since a uniform cross-identity pair hits i with that weight."""
    n_img = sum(k * v for k, v in shape.items())
    sq = sum(k * k * v for k, v in shape.items())
    return size * (n_img - size) / (n_img ** 2 - sq)


@pytest.mark.parametrize("which,spec_pool,spec_gen,spec_single", [
    ("test", 144 / 1338, 10 / 764, 0.30),
    ("val", 121 / 1259, 10 / 684, 0.32),
])
def test_diagnostics_match_revision_4_1_section_3_8(which, spec_pool, spec_gen, spec_single,
                                                    fold0_test, fold0_val):
    ps_, shape, big = (fold0_test, TEST_SHAPE, 144) if which == "test" else (fold0_val, VAL_SHAPE, 121)
    d = P.diagnostics(ps_)
    # heaviest identity: about 10.8 % / 9.6 % of impostor endpoints vs 1.3 % / 1.5 % of genuine
    assert d["pool_max_share"] == pytest.approx(spec_pool, abs=0.015)
    assert d["pool_max_share"] == pytest.approx(_analytic_endpoint_share(shape, big), abs=0.01)
    assert d["genuine_max_share"] == pytest.approx(spec_gen, abs=1e-12)   # exactly 10 of G pairs
    assert d["genuine_max_endpoints"] == 20 and d["genuine_total_endpoints"] == 2 * len(ps_.genuine)
    assert d["pool_max_share"] > 5 * d["genuine_max_share"]
    # single-image identities: about 30 % / 32 % of impostor endpoints, none of genuine
    assert d["pool_single_image_endpoint_share"] == pytest.approx(spec_single, abs=0.03)
    n_img = sum(k * v for k, v in shape.items())
    sq = sum(k * k * v for k, v in shape.items())
    analytic_single = shape[1] * (n_img - 1) / (n_img ** 2 - sq)            # sum over singles of n(N-n)/(N^2-sum n^2)
    assert d["pool_single_image_endpoint_share"] == pytest.approx(analytic_single, abs=0.012)
    assert d["genuine_single_image_endpoint_share"] == 0.0
    assert abs(d["balanced_single_image_endpoint_share"] - d["pool_single_image_endpoint_share"]) < 0.05
    # (the balanced subset is a uniform G-subset of the pool, so it inherits the pool's skew)
    # genuine cap bounds every identity at cap/G of genuine pairs
    assert d["genuine_max_endpoints"] <= 2 * 10
