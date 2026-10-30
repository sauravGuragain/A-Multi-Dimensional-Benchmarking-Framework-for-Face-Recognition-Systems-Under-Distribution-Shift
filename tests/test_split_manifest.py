"""Tests for the split manifest (Revision 4.1, section 2.2).

SCOPE: all tables are SYNTHETIC, built to the V1 counts (identities 4,601/574/574; images
10,636/1,259/1,338). They show the acceptance machinery works; they are not the real Fold 0 split,
which is not in this repository (the splitter and data are not tracked), so the real manifest has
not been generated or validated.
"""
import copy
import errno
import hashlib
import os
import pickle

import pytest

from faceeval.evaluation import split_manifest as S

META = {"dataset_name": "lfw", "fold": 0, "seed": 42, "python_version": "3.12.14",
        "numpy_version": "2.0.0", "code_version": "abc1234", "shuffled_order_sha256": S.V25_SHUFFLED_ORDER_SHA256}


def table(prefix, n_ids, n_img):
    extra = n_img - n_ids
    out = []
    for i in range(n_ids):
        size = 1 + extra // n_ids + (1 if i < extra % n_ids else 0)
        out.extend((f"{prefix}{i:05d}", f"img{j:03d}") for j in range(size))
    assert len(out) == n_img
    return out


@pytest.fixture(scope="module")
def parts():
    return {p: table(p[:2], S.V1_IDENTITIES[p], S.V1_IMAGES[p]) for p in S.PARTITIONS}


@pytest.fixture(scope="module")
def man(parts):
    return S.build_manifest(parts, META)


def test_v1_counts_accepted(man):
    assert {p: len({i for i, _ in man.partitions[p]}) for p in S.PARTITIONS} == S.V1_IDENTITIES
    assert {p: len(man.partitions[p]) for p in S.PARTITIONS} == S.V1_IMAGES


@pytest.mark.parametrize("part,drop", [("train", 1), ("validation", 1), ("test", 1)])
def test_count_mismatch_stops_and_is_never_adjusted(parts, part, drop):
    bad = copy.deepcopy(parts)
    bad[part].pop()                                          # one image fewer
    with pytest.raises(S.ManifestAcceptanceError, match="counts differ"):
        S.build_manifest(bad, META)
    bad = copy.deepcopy(parts)
    bad[part] = [r for r in bad[part] if r[0] != bad[part][0][0]]   # one identity fewer
    with pytest.raises(S.ManifestAcceptanceError, match="counts differ"):
        S.build_manifest(bad, META)


def test_identity_and_image_overlap_rejected_legacy(parts):
    bad = copy.deepcopy(parts)
    bad["test"][0] = ("tr00000", "img000")                    # an identity of train inside test
    with pytest.raises(S.ManifestAcceptanceError, match="overlap"):
        S.build_manifest(bad, META)
    bad = copy.deepcopy(parts)
    bad["validation"][0] = bad["test"][0]                     # identical record in two partitions
    with pytest.raises(S.ManifestAcceptanceError, match="overlap"):
        S.build_manifest(bad, META)


def test_duplicate_record_in_a_partition_rejected(parts):
    bad = copy.deepcopy(parts)
    bad["train"][1] = bad["train"][0]
    with pytest.raises(ValueError, match="duplicate"):
        S.build_manifest(bad, META)


@pytest.mark.parametrize("mut", [
    lambda p: p.pop("test"), lambda p: p.update(extra=[("a", "1")]), lambda p: p.update(test=[]),
    lambda p: p.update(test=[("a|b", "1")]), lambda p: p.update(test=[("", "1")]),
    lambda p: p.update(test="not rows"), lambda p: p.update(test=[("a", 1)]),
])
def test_malformed_partitions_rejected(parts, mut):
    bad = copy.deepcopy(parts)
    mut(bad)
    with pytest.raises(ValueError):
        S.build_manifest(bad, META)


@pytest.mark.parametrize("mut", [
    lambda m: m.pop("seed"), lambda m: m.update(unexpected=1), lambda m: m.update(seed=True),
    lambda m: m.update(seed=None), lambda m: m.update(code_version="a|b"), lambda m: m.update(code_version=""),
    lambda m: m.update(python_version="3.12\n"), lambda m: m.update(shuffled_order_sha256="XYZ"),
    lambda m: m.update(shuffled_order_sha256="A" * 64), lambda m: m.update(numpy_version=2.0),
])
def test_metadata_validation(parts, mut):
    bad = dict(META)
    mut(bad)
    with pytest.raises(ValueError):
        S.build_manifest(parts, bad)


def test_serialization_is_canonical_utf8_lf(man):
    data = S.manifest_bytes(man)
    assert data.endswith(b"\n") and b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8")
    head, *rows = text.split("\n")[:-1]
    kv = dict(x.split("=", 1) for x in head.split("|"))
    assert set(kv) == set(S.REQUIRED_METADATA) | {"manifest_version"} and kv["seed"] == "42"
    assert head == "|".join(sorted(head.split("|")))
    assert rows == sorted(rows) and len(rows) == sum(S.V1_IMAGES.values())
    assert all(r.split("|")[0] in S.PARTITIONS for r in rows)             # partition tag on every row
    assert S.manifest_sha256(man) == hashlib.sha256(data).hexdigest()


def test_hash_independent_of_input_order_and_sensitive_to_content(parts, man):
    shuffled = {p: list(reversed(v)) for p, v in parts.items()}
    assert S.manifest_sha256(S.build_manifest(shuffled, META)) == S.manifest_sha256(man)
    for key, val in (("seed", 43), ("code_version", "other"), ("python_version", "3.14.7"), ("numpy_version", "9")):
        assert S.manifest_sha256(S.build_manifest(parts, {**META, key: val})) != S.manifest_sha256(man), key
    moved = copy.deepcopy(parts)                                           # swap one image id label
    moved["test"][0] = (moved["test"][0][0], "zzz")
    assert S.manifest_sha256(S.build_manifest(moved, META)) != S.manifest_sha256(man)


def test_non_ascii_identities_roundtrip(parts, tmp_path):
    ren = {"te00000": "José_García", "te00001": "Zoë_Åsa"}
    p = {k: list(v) for k, v in parts.items()}
    p["test"] = [(ren.get(i, i), im) for i, im in p["test"]]              # same counts, UTF-8 names
    m = S.build_manifest(p, META)
    path = str(tmp_path / "m.txt")
    sha = S.write_manifest(m, path)
    assert "José_García|img000".encode("utf-8") in open(path, "rb").read()
    assert S.manifest_sha256(S.load_manifest(path, expected_sha256=sha)) == sha


def test_write_load_roundtrip_repeated_loads_same_hash(man, tmp_path):
    path = str(tmp_path / "split_manifest.txt")
    sha = S.write_manifest(man, path)
    assert open(path + ".sha256").read() == f"{sha}  split_manifest.txt\n"
    hashes = {S.manifest_sha256(S.load_manifest(path)) for _ in range(3)}
    assert hashes == {sha}
    assert S.load_manifest(path).partitions == man.partitions
    with pytest.raises(FileExistsError):
        S.write_manifest(man, path)                                         # never overwrites


def test_load_detects_tampering_and_wrong_expected_hash(man, tmp_path):
    path = tmp_path / "m.txt"
    sha = S.write_manifest(man, str(path))
    with pytest.raises(S.ManifestAcceptanceError, match="differs from the expected"):
        S.load_manifest(str(path), expected_sha256="0" * 64)
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"seed=42", b"seed=43", 1))                # edit without updating the sidecar
    with pytest.raises(S.ManifestAcceptanceError, match="sidecar"):
        S.load_manifest(str(path))
    # tampering WITH a consistent sidecar must still fail the acceptance checks
    lines = raw.split(b"\n")
    dropped = b"\n".join(lines[:5] + lines[6:])                              # one row removed
    path.write_bytes(dropped)
    (tmp_path / "m.txt.sha256").write_text(f"{hashlib.sha256(dropped).hexdigest()}  m.txt\n")
    with pytest.raises(S.ManifestAcceptanceError, match="counts differ"):
        S.load_manifest(str(path))
    # CRLF or unsorted rows are not canonical
    crlf = raw.replace(b"\n", b"\r\n")
    path.write_bytes(crlf)
    (tmp_path / "m.txt.sha256").write_text(f"{hashlib.sha256(crlf).hexdigest()}  m.txt\n")
    with pytest.raises(S.ManifestAcceptanceError, match="LF"):
        S.load_manifest(str(path))
    rows = raw.split(b"\n")
    swapped = b"\n".join([rows[0], rows[2], rows[1]] + rows[3:])
    path.write_bytes(swapped)
    (tmp_path / "m.txt.sha256").write_text(f"{hashlib.sha256(swapped).hexdigest()}  m.txt\n")
    with pytest.raises(S.ManifestAcceptanceError, match="canonical"):
        S.load_manifest(str(path))
    (tmp_path / "m.txt.sha256").unlink()
    with pytest.raises(S.ManifestAcceptanceError, match="sidecar"):
        S.load_manifest(str(path))


def test_shuffled_order_comparison_reports_and_does_not_reconcile(parts):
    ok = S.compare_shuffled_order(S.build_manifest(parts, META))
    assert ok["status"] == "match"
    other = S.build_manifest(parts, {**META, "shuffled_order_sha256": "1" * 64})
    r = S.compare_shuffled_order(other)
    assert r["status"] == "MISMATCH" and other.metadata["shuffled_order_sha256"] == "1" * 64   # recorded as is
    nr = S.compare_shuffled_order(S.build_manifest(parts, {**META, "shuffled_order_sha256": S.NOT_RECORDED}))
    assert nr["status"] == "not_recorded"


# ------------------------------------------------------------------ B6 disjointness semantics
def test_identical_record_in_two_partitions_is_rejected_as_an_image_overlap(parts):
    bad = copy.deepcopy(parts)
    bad["validation"][0] = bad["test"][0]
    with pytest.raises(S.ManifestAcceptanceError, match="image records of .* overlap"):
        S.build_manifest(bad, META)


def test_shared_identity_with_distinct_images_is_rejected_as_an_identity_overlap(parts):
    bad = copy.deepcopy(parts)
    bad["test"].append(("tr00000", "only_in_test"))                         # same identity, a NEW image id (no shared record)
    with pytest.raises(S.ManifestAcceptanceError, match="identity sets of .* overlap"):
        S.build_manifest(bad, META)


def test_same_image_id_under_different_identities_in_different_partitions_is_allowed(parts, man):
    """Section 2.2 says image SETS are disjoint, not that image ids are globally unique: an image is the
    (identity, image_id) record. Every partition here reuses 'img000', and the manifest is accepted."""
    ids = {p: {i for i, im in man.partitions[p] if im == "img000"} for p in S.PARTITIONS}
    assert all(ids.values()) and len(set.union(*ids.values())) == sum(len(v) for v in ids.values())


# ------------------------------------------------------------------ B4 immutability
def test_manifest_is_deeply_immutable(parts):
    m = S.build_manifest(parts, META)
    with pytest.raises(TypeError):
        m.metadata["seed"] = "1"
    with pytest.raises(TypeError):
        m.metadata["new"] = "x"
    with pytest.raises(TypeError):
        del m.metadata["seed"]
    with pytest.raises(TypeError):
        m.partitions["train"] = ()
    with pytest.raises(TypeError):
        m.partitions["extra"] = ()
    with pytest.raises(TypeError):
        m.partitions["train"][0] = ("a", "b")
    with pytest.raises(AttributeError):
        m.partitions["train"].append(("a", "b"))
    with pytest.raises(TypeError):
        m.partitions["train"][0][0] = "x"
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.metadata = {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.partitions = {}


def test_manifest_copies_inputs_and_serialization_is_unaffected_by_later_caller_mutation(parts):
    caller_parts = {k: list(v) for k, v in parts.items()}
    caller_meta = dict(META)
    m = S.build_manifest(caller_parts, caller_meta)
    before = S.manifest_bytes(m)
    caller_parts["test"].clear()
    caller_meta["seed"] = 999
    assert S.manifest_bytes(m) == before


def test_manifest_equality_hash_pickle_and_deepcopy(parts, man):
    again = S.build_manifest({k: list(reversed(v)) for k, v in parts.items()}, META)
    assert again == man and hash(again) == hash(man) and man != "x"
    for clone in (pickle.loads(pickle.dumps(man)), copy.deepcopy(man)):
        assert clone == man and S.manifest_sha256(clone) == S.manifest_sha256(man)
        with pytest.raises(TypeError):
            clone.metadata["seed"] = "1"


def test_v1_defaults_are_read_only():
    with pytest.raises(TypeError):
        S.V1_IDENTITIES["train"] = 1
    with pytest.raises(TypeError):
        S.V1_IMAGES["test"] = 1


# ------------------------------------------------------------------ B3 expected counts
@pytest.mark.parametrize("which", ["expected_identities", "expected_images"])
@pytest.mark.parametrize("bad", [
    {"train": 1, "validation": 1},                                     # missing key
    {"train": 1, "validation": 1, "test": 1, "extra": 1},              # extra key
    {"train": 1, "validation": 1, "Test": 1},                          # wrong key
    {"train": True, "validation": 1, "test": 1}, {"train": 1.0, "validation": 1, "test": 1},
    {"train": "1", "validation": 1, "test": 1}, {"train": None, "validation": 1, "test": 1},
    {"train": -1, "validation": 1, "test": 1}, {"train": 0, "validation": 1, "test": 1},
    {"train": [1], "validation": 1, "test": 1}, {1: 1, 2: 1, 3: 1}, [("train", 1)], "train", 5,
])
def test_expected_counts_are_validated_before_comparison(parts, which, bad):
    with pytest.raises(ValueError):
        S.build_manifest(parts, META, **{which: bad})


def test_expected_counts_accept_numpy_integers_and_custom_values(parts):
    import numpy as np
    ok_id = {p: np.int64(S.V1_IDENTITIES[p]) for p in S.PARTITIONS}
    S.build_manifest(parts, META, expected_identities=ok_id, expected_images=dict(S.V1_IMAGES))
    with pytest.raises(S.ManifestAcceptanceError):                           # valid type, wrong value: acceptance error
        S.build_manifest(parts, META, expected_identities={**dict(S.V1_IDENTITIES), "test": 573})


# ------------------------------------------------------------------ B5 metadata
@pytest.mark.parametrize("bad_meta", [
    "not a mapping", None, [("seed", 1)], 5,
    {**META, 1: "x"}, {**META, None: "x"}, {**META, b"seed": "x"},
    {**META, "": "x"}, {**META, "a=b": "x"}, {**META, "a|b": "x"}, {**META, "a\nb": "x"}, {**META, " k": "x"},
    {**META, "extra": "x"}, {**META, "manifest_version": "forged"},
    {**META, "seed": 1.5}, {**META, "seed": [1]}, {**META, "seed": b"1"}, {**META, "code_version": "a\x00b"},
    {**META, "code_version": " lead"}, {**META, "code_version": "trail "}, {**META, "code_version": "a\tb"},
])
def test_metadata_malformed_keys_values_and_containers(parts, bad_meta):
    with pytest.raises(ValueError):
        S.build_manifest(parts, bad_meta)


def test_metadata_value_may_contain_equals_and_roundtrips(parts, tmp_path):
    """'=' is legal in values (the parser splits each header field on its first '=') and illegal in keys."""
    meta = {**META, "code_version": "git=abc==def", "python_version": "3.12.14 (main, Oct=1)"}
    m = S.build_manifest(parts, meta)
    path = str(tmp_path / "m.txt")
    S.write_manifest(m, path)
    back = S.load_manifest(path)
    assert back.metadata["code_version"] == "git=abc==def" and back == m


def test_integer_metadata_is_stored_as_its_decimal_text(parts):
    import numpy as np
    a = S.build_manifest(parts, {**META, "seed": 42, "fold": 0})
    b = S.build_manifest(parts, {**META, "seed": np.int64(42), "fold": "0"})
    assert a.metadata["seed"] == "42" and a == b


# ------------------------------------------------------------------ B1 write safety
def _listing(d):
    return sorted(os.listdir(d))


def test_write_leaves_exactly_the_pair_and_no_temporary_files(man, tmp_path):
    S.write_manifest(man, str(tmp_path / "m.txt"))
    assert _listing(tmp_path) == ["m.txt", "m.txt.sha256"]


@pytest.mark.parametrize("existing", ["m.txt", "m.txt.sha256", "both"])
def test_write_never_overwrites_and_leaves_existing_files_untouched(man, tmp_path, existing):
    names = ["m.txt", "m.txt.sha256"] if existing == "both" else [existing]
    for n in names:
        (tmp_path / n).write_bytes(b"precious " + n.encode())
    with pytest.raises(FileExistsError):
        S.write_manifest(man, str(tmp_path / "m.txt"))
    assert _listing(tmp_path) == sorted(names)
    for n in names:
        assert (tmp_path / n).read_bytes() == b"precious " + n.encode()


def test_write_refuses_dangling_symlink_destination(man, tmp_path):
    os.symlink(str(tmp_path / "nowhere"), str(tmp_path / "m.txt"))
    with pytest.raises(FileExistsError):
        S.write_manifest(man, str(tmp_path / "m.txt"))
    assert not (tmp_path / "nowhere").exists()


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_failure_while_writing_either_temp_file_leaves_nothing(man, tmp_path, monkeypatch, fail_at):
    """fsync is called once per temporary file (manifest, then sidecar); failures at the open/write of the
    first or second temp file and at their fsync are all simulated."""
    real_fdopen, real_fsync, calls = os.fdopen, os.fsync, {"fsync": 0, "fdopen": 0}

    if fail_at in (1, 2):
        def bad_fsync(fd):
            calls["fsync"] += 1
            if calls["fsync"] == fail_at:
                raise OSError(errno.EIO, "injected fsync failure")
            return real_fsync(fd)
        monkeypatch.setattr(os, "fsync", bad_fsync)
    else:
        def bad_fdopen(fd, *a, **k):
            calls["fdopen"] += 1
            if calls["fdopen"] == fail_at - 2:
                os.close(fd)
                raise OSError(errno.ENOSPC, "injected open failure")
            return real_fdopen(fd, *a, **k)
        monkeypatch.setattr(os, "fdopen", bad_fdopen)
    with pytest.raises(OSError):
        S.write_manifest(man, str(tmp_path / "m.txt"))
    assert _listing(tmp_path) == []


def test_failure_publishing_the_sidecar_rolls_back_the_manifest(man, tmp_path, monkeypatch):
    real_link, n = os.link, {"n": 0}

    def bad_link(src, dst, *a, **k):
        n["n"] += 1
        if n["n"] == 2:
            raise OSError(errno.EIO, "injected link failure")
        return real_link(src, dst, *a, **k)
    monkeypatch.setattr(os, "link", bad_link)
    with pytest.raises(OSError):
        S.write_manifest(man, str(tmp_path / "m.txt"))
    assert _listing(tmp_path) == []                                          # manifest link removed, temps removed


def test_failure_publishing_the_manifest_leaves_nothing(man, tmp_path, monkeypatch):
    def bad_link(src, dst, *a, **k):
        raise OSError(errno.EPERM, "injected link failure")
    monkeypatch.setattr(os, "link", bad_link)
    with pytest.raises(OSError):
        S.write_manifest(man, str(tmp_path / "m.txt"))
    assert _listing(tmp_path) == []


@pytest.mark.parametrize("racer", ["m.txt", "m.txt.sha256"])
def test_concurrent_creation_of_a_destination_is_never_overwritten_and_ours_is_rolled_back(man, tmp_path, monkeypatch, racer):
    """Another process creates one destination after our up-front existence check but just before our
    link to that name: the link fails with FileExistsError, the racer's file is untouched and nothing
    of ours (including an already-published manifest) remains."""
    real_link = os.link

    def racing_link(src, dst, *a, **k):
        if os.path.basename(dst) == racer:
            (tmp_path / racer).write_bytes(b"racer")
        return real_link(src, dst, *a, **k)
    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        S.write_manifest(man, str(tmp_path / "m.txt"))
    assert _listing(tmp_path) == [racer]
    assert (tmp_path / racer).read_bytes() == b"racer"


def test_keyboard_interrupt_during_write_also_cleans_up(man, tmp_path, monkeypatch):
    real_link, n = os.link, {"n": 0}

    def interrupt(src, dst, *a, **k):
        n["n"] += 1
        if n["n"] == 2:
            raise KeyboardInterrupt
        return real_link(src, dst, *a, **k)
    monkeypatch.setattr(os, "link", interrupt)
    with pytest.raises(KeyboardInterrupt):
        S.write_manifest(man, str(tmp_path / "m.txt"))
    assert _listing(tmp_path) == []


def test_write_into_missing_directory_raises_and_creates_nothing(man, tmp_path):
    with pytest.raises(OSError):
        S.write_manifest(man, str(tmp_path / "nodir" / "m.txt"))
    assert _listing(tmp_path) == []


def test_orphan_manifest_without_sidecar_is_detected_by_load_and_write(man, tmp_path):
    """The documented crash window: manifest published, sidecar not. Both functions refuse to proceed."""
    (tmp_path / "m.txt").write_bytes(S.manifest_bytes(man))
    with pytest.raises(S.ManifestAcceptanceError, match="missing sidecar"):
        S.load_manifest(str(tmp_path / "m.txt"))
    with pytest.raises(FileExistsError):
        S.write_manifest(man, str(tmp_path / "m.txt"))


# ------------------------------------------------------------------ B2 load hardening
@pytest.fixture
def written(man, tmp_path):
    path = tmp_path / "m.txt"
    S.write_manifest(man, str(path))
    return path


def _rewrite(path, data, sidecar="auto"):
    path.write_bytes(data)
    side = path.parent / (path.name + ".sha256")
    if sidecar == "auto":
        side.write_bytes(f"{hashlib.sha256(data).hexdigest()}  {path.name}\n".encode())
    elif sidecar is not None:
        side.write_bytes(sidecar)


def _expect(path, match, **kw):
    with pytest.raises(S.ManifestAcceptanceError, match=match):
        S.load_manifest(str(path), **kw)


def test_load_good_file(written, man):
    assert S.load_manifest(str(written)) == man


def test_load_empty_manifest(written):
    _rewrite(written, b"")
    _expect(written, "manifest is empty")


def test_load_invalid_utf8(written):
    _rewrite(written, b"\xff\xfe\x00bad\n")
    _expect(written, "UTF-8")


@pytest.mark.parametrize("sidecar,match", [
    (b"", "sidecar hash file is empty"), (b"\n", "sidecar hash file is empty"),
    (b"\xff\xfe\n", "sidecar is not valid UTF-8"),
    (b"zz  m.txt\n", "malformed sidecar"), (b"abc  m.txt\n", "malformed sidecar"),
    (b"A" * 64 + b"  m.txt\n", "malformed sidecar"), (b"a" * 64 + b" m.txt\n", "malformed sidecar"),
    (b"a" * 64 + b"  m.txt", "malformed sidecar"), (b"a" * 64 + b"  m.txt\r\n", "malformed sidecar"),
    (b"a" * 64 + b"  \n", "malformed sidecar"), (b"a" * 64 + b"  m.txt\nextra\n", "malformed sidecar"),
])
def test_load_malformed_sidecar(written, sidecar, match):
    data = written.read_bytes()
    _rewrite(written, data, sidecar=sidecar)
    _expect(written, match)


def test_load_incorrect_sidecar_hash_and_missing_sidecar(written):
    data = written.read_bytes()
    _rewrite(written, data, sidecar=("0" * 64 + "  m.txt\n").encode())
    _expect(written, "sidecar says")
    (written.parent / "m.txt.sha256").unlink()
    _expect(written, "missing sidecar")


def test_load_expected_hash_wrong_vs_malformed(written, man):
    sha = S.manifest_sha256(man)
    assert S.load_manifest(str(written), expected_sha256=sha) == man
    _expect(written, "differs from the expected", expected_sha256="0" * 64)
    for bad in ("abc", "A" * 64, 5, b"x", ["0" * 64]):
        with pytest.raises(ValueError):
            S.load_manifest(str(written), expected_sha256=bad)


def _lines(path):
    return path.read_bytes().split(b"\n")


@pytest.mark.parametrize("edit,match", [
    (lambda L: [L[0]] + [b""] + L[1:], "malformed manifest row"),                              # blank row
    (lambda L: [L[0], b"train|only_two"] + L[1:], "malformed manifest row"),
    (lambda L: [L[0], b"train|a|b|c"] + L[1:], "malformed manifest row"),
    (lambda L: [L[0], b"holdout|a|b"] + L[1:], "invalid partition name"),
    (lambda L: [L[0], b"Train|a|b"] + L[1:], "invalid partition name"),
    (lambda L: [b"seed42"] + L[1:], "malformed metadata field"),
    (lambda L: [b"=x|" + L[0]] + L[1:], "malformed metadata field"),
    (lambda L: [L[0] + b"|seed=43"] + L[1:], "duplicate metadata key"),
    (lambda L: [b"|".join(f for f in L[0].split(b"|") if not f.startswith(b"seed="))] + L[1:], "missing"),
    (lambda L: [L[0] + b"|extra=1"] + L[1:], "unexpected"),
    (lambda L: [L[0].replace(b"faceeval-split-manifest/r4.1/1", b"other/9")] + L[1:], "manifest version"),
    (lambda L: [b"|".join(f for f in L[0].split(b"|") if not f.startswith(b"manifest_version="))] + L[1:], "manifest version"),
    (lambda L: [L[0].replace(b"shuffled_order_sha256=" + S.V25_SHUFFLED_ORDER_SHA256.encode(), b"shuffled_order_sha256=ZZ")] + L[1:], "invalid"),
    (lambda L: [L[0], L[2], L[1]] + L[3:], "canonical form"),                                 # swapped rows
    (lambda L: [b"|".join(reversed(L[0].split(b"|")))] + L[1:], "canonical form"),            # header order
])
def test_load_malformed_content_gives_manifest_acceptance_error(written, edit, match):
    data = b"\n".join(edit(_lines(written)))
    _rewrite(written, data)
    _expect(written, match)


def test_load_newline_format_errors(written):
    raw = written.read_bytes()
    _rewrite(written, raw.replace(b"\n", b"\r\n"))
    _expect(written, "CR")
    _rewrite(written, raw.rstrip(b"\n"))
    _expect(written, "does not end with LF")
    _rewrite(written, raw + b"\n")                                                           # extra trailing blank line
    _expect(written, "malformed manifest row")
    _rewrite(written, raw.split(b"\n")[0] + b"\n")                                          # header only
    _expect(written, "no header and rows")


def test_load_count_and_disjointness_failures_come_from_content(written):
    L = _lines(written)
    _rewrite(written, b"\n".join(L[:5] + L[6:]))                                              # one row removed
    _expect(written, "counts differ")


def test_load_does_not_hide_filesystem_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_manifest(str(tmp_path / "absent.txt"))
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(OSError) as ei:
        S.load_manifest(str(d))
    assert not isinstance(ei.value, S.ManifestAcceptanceError)


@pytest.mark.parametrize("target", ["manifest", "sidecar"])
def test_load_permission_error_propagates_as_oserror(written, monkeypatch, target):
    """Simulated with a patched ``open`` (the sandbox runs as root, where permission bits are not enforced,
    so a real chmod test would be vacuous): the PermissionError must not be converted or hidden."""
    import builtins
    real_open = builtins.open
    victim = str(written) if target == "manifest" else str(written) + ".sha256"

    def deny(file, *a, **k):
        if os.fspath(file) == victim:
            raise PermissionError(errno.EACCES, "denied", victim)
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", deny)
    with pytest.raises(PermissionError) as ei:
        S.load_manifest(str(written))
    assert not isinstance(ei.value, S.ManifestAcceptanceError)


def test_load_custom_expected_counts_validated_before_reading(written):
    with pytest.raises(ValueError):
        S.load_manifest(str(written), expected_identities={"train": 1})
    assert S.load_manifest(str(written), expected_identities=dict(S.V1_IDENTITIES)).metadata["seed"] == "42"


# ------------------------------------------------------------------ B7 shuffled-order digest
def test_shuffled_order_report_states_that_it_is_a_comparison_not_a_reproduction(parts):
    for value, status in ((S.V25_SHUFFLED_ORDER_SHA256, "match"), ("1" * 64, "MISMATCH"), (S.NOT_RECORDED, "not_recorded")):
        r = S.compare_shuffled_order(S.build_manifest(parts, {**META, "shuffled_order_sha256": value}))
        assert r["status"] == status and r["recorded"] == value and r["expected"] == S.V25_SHUFFLED_ORDER_SHA256
        assert r["reproduced_from_source"] is False


# ------------------------------------------------------------------ integration
def test_import_compatibility_and_determinism():
    import importlib
    import faceeval.evaluation.pairs as pairs_mod
    importlib.reload(pairs_mod)
    importlib.reload(S)
    assert S.validate_images is pairs_mod.validate_images
    a = S.build_manifest({p: table(p[:2], S.V1_IDENTITIES[p], S.V1_IMAGES[p]) for p in S.PARTITIONS}, META)
    b = S.build_manifest({p: table(p[:2], S.V1_IDENTITIES[p], S.V1_IMAGES[p]) for p in S.PARTITIONS}, META)
    assert S.manifest_sha256(a) == S.manifest_sha256(b)
