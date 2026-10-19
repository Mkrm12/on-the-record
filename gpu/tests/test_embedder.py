"""The embedding worker."""

import sys
import types
import zlib

import numpy as np
import pytest
from conftest import add_chunk_with_context, add_chunks

import embedder
from embedder import (
    DIMENSIONS,
    INDEX_NAME,
    EMBEDDING_RECIPE,
    MODEL_NAME,
    clean_for_embedding,
    connect,
    count_pending,
    fetch_page,
    l2_normalize,
    rebuild_index,
    run,
    status,
    validate_vectors,
    write_embeddings,
)

# --------------------------------------------------------------------------- clean_for_embedding: what may be sent to
# the model at all ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [None, "", " ", "\n\t  ", "...", "—", "[]", "\x00", " \x00 ", "\ud800", "\udc00 \udfff"])
def test_nothing_embeddable_gives_none(text):
    assert clean_for_embedding(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello world.", "Hello world."),
        ("  Hello   world.\n", "Hello world."),
        ("a\x00b", "ab"),
        # a JSON body can carry a lone UTF-16 surrogate; the tokenizer raises TypeError on it (found while building the
        # HTTP service), so it must be dropped here, while real astral characters and accents survive
        ("a\ud800b", "ab"),
        ("abc \udc00 def", "abc def"),
        ("Café naïve 😀 £5", "Café naïve 😀 £5"),
        ("Line one\nline two", "Line one line two"),
        ("The scheme costs £60.5 billion.", "The scheme costs £60.5 billion."),
        ("2026", "2026"),
        ("Yes.", "Yes."),
    ],
)
def test_embeddable_text_is_tidied_but_not_rewritten(text, expected):
    assert clean_for_embedding(text) == expected


# --------------------------------------------------------------------------- l2_normalize / validate_vectors
# ---------------------------------------------------------------------------


def unit(n=3, seed=0):
    return l2_normalize(np.random.default_rng(seed).standard_normal((n, DIMENSIONS)))


def test_l2_normalize_gives_unit_rows_in_float32():
    out = l2_normalize(np.random.default_rng(1).standard_normal((5, DIMENSIONS)) * 37.0)
    assert out.dtype == np.float32 and out.shape == (5, DIMENSIONS)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_l2_normalize_keeps_direction():
    a = np.random.default_rng(2).standard_normal((1, DIMENSIONS))
    assert np.allclose(l2_normalize(a)[0], (a / np.linalg.norm(a))[0], atol=1e-6)


def test_l2_normalize_leaves_a_zero_row_zero_rather_than_nan():
    out = l2_normalize(np.zeros((2, DIMENSIONS)))
    assert np.all(out == 0) and np.isfinite(out).all()


def test_valid_vectors_pass():
    validate_vectors(unit())


@pytest.mark.parametrize(
    "bad,message",
    [
        (np.ones((2, DIMENSIONS - 1), dtype=np.float32) / 10, "shape"),
        (np.ones(DIMENSIONS, dtype=np.float32), "shape"),
        (np.zeros((2, DIMENSIONS), dtype=np.float32), "norm"),
        (np.full((2, DIMENSIONS), np.nan, dtype=np.float32), "finite"),
        (np.full((2, DIMENSIONS), np.inf, dtype=np.float32), "finite"),
        (np.full((2, DIMENSIONS), 1.0, dtype=np.float32), "norm"),  # not unit length
    ],
)
def test_invalid_vectors_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_vectors(bad)


# --------------------------------------------------------------------------- a deterministic stand-in for the model
# ---------------------------------------------------------------------------


class FakeEncoder:
    """Same text -> same unit vector. Optionally misbehaves on the nth call, or calls a hook."""

    def __init__(self, fail_on_call=None, interrupt_on_call=None, on_call=None):
        self.calls = []
        self.fail_on_call, self.interrupt_on_call, self.on_call = fail_on_call, interrupt_on_call, on_call

    def encode_documents(self, texts):
        self.calls.append(list(texts))
        n = len(self.calls)
        if self.on_call:
            self.on_call(n)
        if n == self.fail_on_call:
            raise RuntimeError("simulated GPU failure")
        if n == self.interrupt_on_call:
            raise KeyboardInterrupt
        out = []
        for text in texts:
            cleaned = clean_for_embedding(text)
            if cleaned is None:
                out.append(None)
                continue
            v = np.random.default_rng(zlib.crc32(cleaned.encode())).standard_normal(DIMENSIONS)
            out.append(l2_normalize(v[None, :])[0])
        return out

    def vram_mib(self):
        return 0.0, 0.0


def index_info(conn):
    return conn.execute(
        "SELECT c.oid, am.amname, c.reloptions, pg_get_indexdef(c.oid) FROM pg_class c JOIN pg_am am ON am.oid = c.relam"
        " WHERE c.relname = %s",
        (INDEX_NAME,),
    ).fetchone()


def embedded_ids(conn):
    return {r[0] for r in conn.execute("SELECT id FROM chunk WHERE embedding IS NOT NULL")}


SENTENCES = [f"Sentence number {i} concerns housing policy." for i in range(6)]


# --------------------------------------------------------------------------- the work queue
# ---------------------------------------------------------------------------


def test_count_pending_ignores_boilerplate_and_embedded(db, scratch_dsn):
    ids = add_chunks(db, ["one", ("Question put and agreed to.", True), "three"])
    with connect(scratch_dsn) as conn:
        assert count_pending(conn) == 2
        write_embeddings(conn, [ids[0]], unit(1))
        conn.commit()
        assert count_pending(conn) == 1


def test_fetch_page_is_keyset_ordered_and_skips_boilerplate(db, scratch_dsn):
    ids = add_chunks(db, ["a1", ("boiler", True), "a3", "a4", "a5"])
    with connect(scratch_dsn) as conn:
        first = fetch_page(conn, after_id=0, size=2)
        assert [i for i, _ in first] == [ids[0], ids[2]] and [b for _, b in first] == ["a1", "a3"]
        second = fetch_page(conn, after_id=first[-1][0], size=2)
        assert [i for i, _ in second] == [ids[3], ids[4]]
        assert fetch_page(conn, after_id=second[-1][0], size=2) == []


def test_fetch_page_skips_rows_that_already_have_a_vector(db, scratch_dsn):
    ids = add_chunks(db, ["a", "b", "c"])
    with connect(scratch_dsn) as conn:
        write_embeddings(conn, [ids[1]], unit(1))
        conn.commit()
        assert [i for i, _ in fetch_page(conn, 0, 10)] == [ids[0], ids[2]]


# --------------------------------------------------------------------------- writing vectors
# ---------------------------------------------------------------------------


def test_write_sets_vector_model_and_timestamp(db, scratch_dsn):
    ids = add_chunks(db, ["a", "b"])
    vectors = unit(2, seed=5)
    with connect(scratch_dsn) as conn:
        assert write_embeddings(conn, ids, vectors) == 2
        conn.commit()
        rows = conn.execute("SELECT embedding, embedding_model, embedded_at, vector_dims(embedding) FROM chunk ORDER BY id").fetchall()
    for (vec, model, at, dims), expected in zip(rows, vectors):
        assert np.allclose(vec.to_numpy(), expected) and model == MODEL_NAME and at is not None and dims == DIMENSIONS


def test_write_never_overwrites_an_existing_vector(db, scratch_dsn):
    (cid,) = add_chunks(db, ["a"])
    first, second = unit(1, seed=1), unit(1, seed=2)
    with connect(scratch_dsn) as conn:
        assert write_embeddings(conn, [cid], first) == 1
        assert write_embeddings(conn, [cid], second) == 0  # already embedded: a concurrent or repeated run is harmless
        conn.commit()
        assert np.allclose(conn.execute("SELECT embedding FROM chunk").fetchone()[0].to_numpy(), first[0])


def test_write_refuses_bad_vectors_before_touching_the_database(db, scratch_dsn):
    (cid,) = add_chunks(db, ["a"])
    with connect(scratch_dsn) as conn, pytest.raises(ValueError):
        write_embeddings(conn, [cid], np.ones((1, DIMENSIONS - 1), dtype=np.float32))


def test_write_refuses_a_different_number_of_ids_and_vectors(db, scratch_dsn):
    (cid,) = add_chunks(db, ["a"])
    with connect(scratch_dsn) as conn, pytest.raises(ValueError, match="ids but"):
        write_embeddings(conn, [cid], unit(2))


def test_write_with_no_rows_is_a_no_op(db, scratch_dsn):
    with connect(scratch_dsn) as conn:
        assert write_embeddings(conn, [], np.zeros((0, DIMENSIONS), dtype=np.float32)) == 0


def test_write_leaves_other_columns_alone(db, scratch_dsn):
    (cid,) = add_chunks(db, ["Some body text."])
    before = db.execute("SELECT body, parent_body, seq, sent_idx, speech_id, is_boilerplate FROM chunk").fetchone()
    with connect(scratch_dsn) as conn:
        write_embeddings(conn, [cid], unit(1))
        conn.commit()
    assert db.execute("SELECT body, parent_body, seq, sent_idx, speech_id, is_boilerplate FROM chunk").fetchone() == before


# --------------------------------------------------------------------------- run: the whole worker, with a fake encoder
# ---------------------------------------------------------------------------


def test_run_embeds_every_pending_chunk_and_leaves_boilerplate_null(db, scratch_dsn):
    ids = add_chunks(db, [*SENTENCES, ("Question put and agreed to.", True)])
    result = run(scratch_dsn, FakeEncoder(), page_size=4, index_mode="never")
    assert result.embedded == 6 and result.skipped == 0
    assert embedded_ids(db) == set(ids[:6])
    assert db.execute("SELECT embedding IS NULL FROM chunk WHERE is_boilerplate").fetchone()[0] is True
    assert db.execute("SELECT count(*) FROM chunk WHERE embedding IS NOT NULL AND embedding_model = %s AND embedded_at IS NOT NULL", (MODEL_NAME,)).fetchone()[0] == 6


def test_run_never_sends_boilerplate_to_the_encoder(db, scratch_dsn):
    add_chunks(db, ["real sentence one", ("Question put and agreed to.", True), "real sentence two"])
    enc = FakeEncoder()
    run(scratch_dsn, enc, index_mode="never")
    assert [t for call in enc.calls for t in call] == ["real sentence one", "real sentence two"]


def test_run_is_idempotent(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    run(scratch_dsn, FakeEncoder(), index_mode="never")
    again = FakeEncoder()
    result = run(scratch_dsn, again, index_mode="never")
    assert result.embedded == 0 and again.calls == []


def test_run_honours_the_limit(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    result = run(scratch_dsn, FakeEncoder(), limit=4, page_size=3, index_mode="never")
    assert result.embedded == 4 and count_pending_via(db) == 2


def count_pending_via(conn):
    return conn.execute("SELECT count(*) FROM chunk WHERE embedding IS NULL AND NOT is_boilerplate").fetchone()[0]


def test_run_resumes_after_a_failure_without_redoing_finished_pages(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    with pytest.raises(RuntimeError, match="simulated"):
        run(scratch_dsn, FakeEncoder(fail_on_call=2), page_size=2, index_mode="never")
    assert len(embedded_ids(db)) == 2  # page 1 was committed before page 2 failed
    resumed = FakeEncoder()
    result = run(scratch_dsn, resumed, page_size=2, index_mode="never")
    assert result.embedded == 4 and len(embedded_ids(db)) == 6
    assert [t for call in resumed.calls for t in call] == SENTENCES[2:]


def test_a_page_that_fails_leaves_no_partial_writes(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    enc = FakeEncoder(fail_on_call=1)
    with pytest.raises(RuntimeError):
        run(scratch_dsn, enc, page_size=6, index_mode="never")
    assert embedded_ids(db) == set()


def test_unembeddable_bodies_stay_null_and_the_run_still_finishes(db, scratch_dsn):
    ids = add_chunks(db, ["good sentence", "...", "   ", "another good sentence"])
    result = run(scratch_dsn, FakeEncoder(), page_size=2, index_mode="never")
    assert result.embedded == 2 and result.skipped == 2
    assert embedded_ids(db) == {ids[0], ids[3]}


def test_identical_bodies_get_identical_vectors(db, scratch_dsn):
    add_chunks(db, ["Yes.", "Yes.", "No."])
    run(scratch_dsn, FakeEncoder(), index_mode="never")
    a, b, c = (r[0].to_numpy() for r in connect(scratch_dsn).execute("SELECT embedding FROM chunk ORDER BY id"))
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_on_page_reports_progress(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    seen = []
    run(scratch_dsn, FakeEncoder(), page_size=4, index_mode="never", on_page=seen.append)
    assert [p.done for p in seen] == [4, 6] and all(p.total == 6 for p in seen)
    assert seen[-1].embedded == 6


# --------------------------------------------------------------------------- the HNSW index lifecycle
# ---------------------------------------------------------------------------


def test_a_managed_run_drops_the_index_first_and_rebuilds_it_after_the_load(db, scratch_dsn, monkeypatch):
    add_chunks(db, SENTENCES)
    assert index_info(db) is not None  # schema.sql creates it, empty
    events = []
    for name in ("drop_index", "create_index", "vacuum_analyze"):
        real = getattr(embedder, name)
        monkeypatch.setattr(embedder, name, lambda *a, _real=real, _name=name, **k: (events.append(_name), _real(*a, **k))[1])

    def during_load(n):
        events.append(f"encode{n}")
        assert index_info(db) is None, "the index must not exist while vectors are being written"

    run(scratch_dsn, FakeEncoder(on_call=during_load), page_size=2, index_mode="always")
    assert events == ["drop_index", "encode1", "encode2", "encode3", "create_index", "vacuum_analyze"]


def test_the_rebuilt_index_is_hnsw_cosine_with_the_agreed_parameters(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    run(scratch_dsn, FakeEncoder(), index_mode="always")
    _, method, options, definition = index_info(db)
    assert method == "hnsw" and set(options) == {"m=16", "ef_construction=64"}
    form = embedder._vector_index().frozen_form()  # the FILE, not embedder.index_form: a check of the embedder
    assert embedder._vector_index().form_of(definition) == form, definition  # against itself would prove nothing
    assert embedder._vector_index().OPCLASS[form] in definition and "(embedding" in definition


def test_the_rebuilt_index_actually_answers_nearest_neighbour_queries(db, scratch_dsn):
    """Through the index, in the form the dense leg queries it: an ORDER BY in the other form would be answered by a
    sequential scan, and this test would pass while proving nothing about the index.
    """
    add_chunks(db, SENTENCES)
    run(scratch_dsn, FakeEncoder(), index_mode="always")
    target = FakeEncoder().encode_documents([SENTENCES[3]])[0]
    order = embedder._vector_index().order_expression(embedder.index_form(), alias="", param="%s")
    with connect(scratch_dsn) as conn, conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")
        plan = " ".join(r[0] for r in conn.execute(f"EXPLAIN SELECT body FROM chunk ORDER BY {order} LIMIT 1",
                                                   (target,)).fetchall())
        nearest = conn.execute(f"SELECT body FROM chunk ORDER BY {order} LIMIT 1", (target,)).fetchone()[0]
    assert INDEX_NAME in plan, plan
    assert nearest == SENTENCES[3]


def test_a_failed_managed_run_still_leaves_an_index_behind(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    with pytest.raises(RuntimeError):
        run(scratch_dsn, FakeEncoder(fail_on_call=2), page_size=2, index_mode="always")
    assert index_info(db) is not None


def test_a_successful_run_whose_index_rebuild_fails_is_a_failure_not_a_warning(db, scratch_dsn, monkeypatch):
    """It printed a warning and exited 0, so a pipeline recorded "embed: done" over a corpus with no index."""
    add_chunks(db, SENTENCES)

    def broken(*a, **k):
        raise RuntimeError("could not extend file: No space left on device")

    monkeypatch.setattr(embedder, "create_index", broken)
    with pytest.raises(RuntimeError, match="No space left"):
        run(scratch_dsn, FakeEncoder(), page_size=2, index_mode="always")
    assert count_pending_via(db) == 0  # the vectors themselves were kept: a re-run only has to build the index


def test_a_failed_rebuild_never_masks_the_error_that_stopped_the_run(db, scratch_dsn, monkeypatch, capsys):
    add_chunks(db, SENTENCES)

    def broken(*a, **k):
        raise RuntimeError("index rebuild failed too")

    monkeypatch.setattr(embedder, "create_index", broken)
    with pytest.raises(RuntimeError) as err:
        run(scratch_dsn, FakeEncoder(fail_on_call=2), page_size=2, index_mode="always")
    assert "index rebuild" not in str(err.value), "the rebuild's error replaced the one that stopped the run"
    assert "could not rebuild" in capsys.readouterr().err


def test_an_interrupted_run_does_not_spend_minutes_rebuilding_but_the_next_run_repairs_it(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    with pytest.raises(KeyboardInterrupt):
        run(scratch_dsn, FakeEncoder(interrupt_on_call=2), page_size=2, index_mode="always")
    assert index_info(db) is None
    run(scratch_dsn, FakeEncoder(), page_size=2, index_mode="always")
    assert index_info(db) is not None and count_pending_via(db) == 0


def test_a_small_run_leaves_the_existing_index_alone(db, scratch_dsn, monkeypatch, capsys):
    add_chunks(db, SENTENCES[:3])
    before = index_info(db)[0]
    events = []
    monkeypatch.setattr(embedder, "drop_index", lambda *a, **k: events.append("drop"))
    run(scratch_dsn, FakeEncoder(), index_mode="auto", index_threshold=5000)
    assert events == [] and index_info(db)[0] == before
    assert capsys.readouterr().err == ""  # and it did not even try to build one (which would only warn "already exists")


def test_auto_mode_goes_managed_when_the_queue_reaches_the_threshold(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    before = index_info(db)[0]
    run(scratch_dsn, FakeEncoder(), index_mode="auto", index_threshold=6)
    assert index_info(db)[0] != before  # dropped and rebuilt


def test_never_mode_neither_drops_nor_creates(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    before = index_info(db)[0]
    run(scratch_dsn, FakeEncoder(), index_mode="never")
    assert index_info(db)[0] == before
    db.execute(f"DROP INDEX {INDEX_NAME}")
    add_chunks(db, ["one more sentence"])
    run(scratch_dsn, FakeEncoder(), index_mode="never")
    assert index_info(db) is None


def test_a_missing_index_is_repaired_even_when_there_is_no_work(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    run(scratch_dsn, FakeEncoder(), index_mode="never")
    db.execute(f"DROP INDEX {INDEX_NAME}")
    result = run(scratch_dsn, FakeEncoder(), index_mode="auto")
    assert result.embedded == 0 and index_info(db) is not None


def test_vacuum_analyze_only_runs_when_something_was_written(db, scratch_dsn, monkeypatch):
    add_chunks(db, SENTENCES)
    events = []
    monkeypatch.setattr(embedder, "vacuum_analyze", lambda *a, **k: events.append("vacuum"))
    run(scratch_dsn, FakeEncoder(), index_mode="never")
    assert events == ["vacuum"]
    run(scratch_dsn, FakeEncoder(), index_mode="never")  # nothing left to do
    assert events == ["vacuum"]
    add_chunks(db, ["late arrival"])
    run(scratch_dsn, FakeEncoder(), index_mode="never", vacuum=False)
    assert events == ["vacuum"]


def test_rebuild_index_replaces_the_index_and_reports_its_size(db, scratch_dsn):
    add_chunks(db, SENTENCES)
    run(scratch_dsn, FakeEncoder(), index_mode="never")
    before = index_info(db)[0]
    seconds = rebuild_index(scratch_dsn)
    assert index_info(db)[0] != before and seconds >= 0


# --------------------------------------------------------------------------- command line
# ---------------------------------------------------------------------------


def test_cli_status_prints_the_counts_and_touches_nothing(db, scratch_dsn, capsys):
    add_chunks(db, SENTENCES[:2])
    assert embedder.main(["--dsn", scratch_dsn, "--status"]) == 0
    out = capsys.readouterr().out
    assert "pending" in out and "2" in out and INDEX_NAME in out
    assert count_pending_via(db) == 2


def test_cli_index_only_rebuilds_the_index(db, scratch_dsn):
    before = index_info(db)[0]
    assert embedder.main(["--dsn", scratch_dsn, "--index-only"]) == 0
    assert index_info(db)[0] != before


# --------------------------------------------------------------------------- status
# ---------------------------------------------------------------------------


def test_status_counts(db, scratch_dsn):
    add_chunks(db, [*SENTENCES[:4], ("Question put.", True)])
    run(scratch_dsn, FakeEncoder(), limit=3, index_mode="never")
    s = status(scratch_dsn)
    assert (s.total, s.boilerplate, s.embedded, s.pending) == (5, 1, 3, 1)
    # Keyed on the RECIPE, not the model: two vectors are only comparable if the text recipe matches too, so status has
    # to be able to show a column holding both.
    assert s.index_exists is True and s.models == {EMBEDDING_RECIPE: 3}


# --------------------------------------------------------------------------- loading the model: the local cache first,
# the network only when the weights are not there
# ---------------------------------------------------------------------------


def fake_sentence_transformer(monkeypatch, fail_offline=False, fail_always=False):
    calls = []

    class FakeST:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))
            if fail_always:
                raise RuntimeError("corrupt weights")
            if fail_offline and kwargs.get("local_files_only"):
                raise OSError("not in the local cache")

    # a stand-in module: importing the real sentence_transformers would pull in torch and transformers (~5 s)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    return calls


def test_the_model_is_loaded_from_the_local_cache_without_touching_the_network(monkeypatch):
    calls = fake_sentence_transformer(monkeypatch)
    embedder._load_model("cpu")
    assert calls == [(MODEL_NAME, {"revision": embedder.MODEL_REVISION, "device": "cpu", "local_files_only": True})]


def test_the_model_is_downloaded_only_when_it_is_not_cached(monkeypatch):
    calls = fake_sentence_transformer(monkeypatch, fail_offline=True)
    embedder._load_model("cpu")
    assert [kw.get("local_files_only", False) for _, kw in calls] == [True, False]
    assert all(kw["revision"] == embedder.MODEL_REVISION for _, kw in calls)


def test_a_load_error_that_is_not_a_cache_miss_is_not_retried_or_hidden(monkeypatch):
    calls = fake_sentence_transformer(monkeypatch, fail_always=True)
    with pytest.raises(RuntimeError, match="corrupt"):
        embedder._load_model("cpu")
    assert len(calls) == 1


# --------------------------------------------------------------------------- the real model on the GPU
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    """The real model on the GPU. torch is imported here, not at module level, so the tests above run without it."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    return embedder.Embedder(device="cuda:0", dtype="float16", batch_size=64)


def test_gpu_vectors_have_384_dimensions_and_unit_length(model):
    out = model.encode_documents(["The Minister will make a statement.", "Will the hon. Member give way?", "Housing targets."])
    assert all(v is not None for v in out)
    matrix = np.stack(out)
    assert matrix.shape == (3, 384) and matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_gpu_empty_and_null_inputs_give_none_and_keep_their_position(model):
    out = model.encode_documents(["First real sentence.", None, "", "   ", "Second real sentence."])
    assert [v is None for v in out] == [False, True, True, True, False]
    assert out[0].shape == (384,) and out[4].shape == (384,)


def test_gpu_all_empty_input_does_not_touch_the_model(model):
    assert model.encode_documents([None, "", " "]) == [None, None, None]
    assert model.encode_documents([]) == []


def test_gpu_same_text_gives_the_same_vector_and_different_text_does_not(model):
    a, b, c = model.encode_documents(["Cut waiting lists.", "Cut waiting lists.", "Build more homes near the railway."])
    assert np.allclose(a, b, atol=1e-4) and float(a @ c) < 0.95


def test_gpu_result_does_not_depend_on_batch_composition(model):
    texts = [f"Sentence about topic number {i} in the debate." for i in range(70)]
    together = np.stack(model.encode_documents(texts))
    alone = np.stack(model.encode_documents([texts[7]]))
    assert float(together[7] @ alone[0]) > 0.999


def test_gpu_related_sentences_are_closer_than_unrelated_ones(model):
    q, related, unrelated = model.encode_documents(
        ["The NHS dentist waiting list is far too long.", "Patients cannot get an appointment with an NHS dentist.", "The Bill amends the Civil Aviation Act 2006."]
    )
    assert float(q @ related) > float(q @ unrelated)


def test_gpu_queries_carry_the_bge_instruction_and_documents_do_not(model):
    text = "What did the Minister say about dentists?"
    (as_query,) = model.encode_queries([text])
    (as_document,) = model.encode_documents([text])
    assert as_query.shape == (384,) and float(as_query @ as_document) < 0.9999
    assert embedder.QUERY_INSTRUCTION.startswith("Represent this sentence for searching")


def test_gpu_model_stays_far_below_the_vram_budget(model):
    model.encode_documents(["Warm up sentence."] * 300)
    allocated, peak = model.vram_mib()
    assert 0 < peak < 2048


def test_gpu_truncation_of_a_very_long_input_is_safe(model):
    (v,) = model.encode_documents(["word " * 5000])
    assert v.shape == (384,) and np.isfinite(v).all()


def test_gpu_end_to_end_through_the_command_line_stores_real_vectors(db, scratch_dsn, model):
    bodies = ["The Minister will make a statement on housing.", "Will the hon. Member give way on that point?", "Dentists are leaving the NHS."]
    ids = add_chunks(db, bodies)
    assert embedder.main(["--dsn", scratch_dsn, "--limit", "3", "--index", "never", "--no-vacuum"]) == 0
    with connect(scratch_dsn) as reader:  # the worker's connection knows the vector type; the plain fixture one does not
        rows = reader.execute("SELECT embedding, embedding_model FROM chunk ORDER BY id").fetchall()
    stored = np.stack([r[0].to_numpy() for r in rows])
    assert {r[1] for r in rows} == {MODEL_NAME} and stored.shape == (3, 384)
    fresh = np.stack(model.encode_documents(bodies))
    assert np.all((stored * fresh).sum(axis=1) > 0.999)
    assert len(ids) == 3


def test_a_chunk_embedded_under_another_recipe_is_pending_again(db, scratch_dsn):
    """`embedding_model` alone cannot tell a bare sentence apart from a sentence under a contextual header, so a recipe
    experiment would leave two incomparable populations in one column and nothing would say so. A row whose recipe
    does not match the target is work, not finished work.
    """
    add_chunks(db, SENTENCES[:3])
    ids = [r[0] for r in db.execute("SELECT id FROM chunk ORDER BY id").fetchall()]
    with connect(scratch_dsn) as conn:
        write_embeddings(conn, ids, unit(3, seed=1), recipe="model|header|v2")
        conn.commit()
        assert count_pending(conn, recipe="model|header|v2") == 0
        assert count_pending(conn, recipe=EMBEDDING_RECIPE) == 3, "another recipe's vectors are not this recipe's"
        assert [i for i, _ in fetch_page(conn, 0, 10, recipe=EMBEDDING_RECIPE)] == ids
        # Re-embedding under the target recipe overwrites, rather than skipping as "already embedded".
        assert write_embeddings(conn, ids, unit(3, seed=2)) == 3
        conn.commit()
        assert count_pending(conn, recipe=EMBEDDING_RECIPE) == 0
        recipes = dict(conn.execute("SELECT embedding_recipe, count(*) FROM chunk GROUP BY 1").fetchall())
        assert recipes == {EMBEDDING_RECIPE: 3}


# --------------------------------------------------------------------------- the adopted recipe: bge-small under a full
# contextual header ---------------------------------------------------------------------------


def test_the_adopted_recipe_is_the_one_the_bakeoff_chose():
    """The tag written into every row, spelled out. If someone changes TEXT_RECIPE without meaning to, the whole corpus
    silently becomes pending and the index becomes a mix of two incomparable populations, so the constant is worth
    pinning rather than trusting.
    """
    assert embedder.TEXT_RECIPE == "full"
    assert embedder.RECIPE_VERSION == "v2"
    assert EMBEDDING_RECIPE == "BAAI/bge-small-en-v1.5|full|v2"


def test_the_recipe_comes_from_the_module_that_measured_it():
    """Not a second implementation. A recipe that renders one way in the bake-off and another in production is a
    regression no test on either side would catch.
    """
    from retrieval.recipe import RECIPES_BY_KEY

    assert embedder.header_recipe() is RECIPES_BY_KEY["full"]
    assert embedder.header_recipe().fields == ("date", "house", "heading", "speaker_name", "party")
    with pytest.raises(ValueError, match="unknown text recipe"):
        embedder.header_recipe("no-such-recipe")


def test_render_row_puts_the_metadata_in_front_of_a_cleaned_body():
    row = {"body": "  The SaxaVord\tspaceport   in Shetland. ", "date": "2024-03-14", "house": "Commons",
           "heading": "Space Industry", "speaker_name": "Ms Smith", "party": "Labour"}  # fmt: skip
    assert embedder.render_row(row, embedder.header_recipe()) == (
        "[2024-03-14 | Commons | Space Industry | Ms Smith | Labour] The SaxaVord spaceport in Shetland."
    )


def test_render_row_omits_missing_fields_rather_than_writing_none():
    row = {"body": "Text.", "date": "2024-03-14", "house": None, "heading": "Heading", "speaker_name": "",
           "party": "None"}  # fmt: skip
    assert embedder.render_row(row, embedder.header_recipe()) == "[2024-03-14 | Heading] Text."


def test_render_row_falls_back_to_the_bare_body_when_there_is_no_metadata():
    row = {"body": "Text.", "date": None, "house": None, "heading": None, "speaker_name": None, "party": None}
    assert embedder.render_row(row, embedder.header_recipe()) == "Text."


@pytest.mark.parametrize("body", [None, "", "   ", "\x00", "...", "\ud800"])
def test_render_row_refuses_to_embed_a_header_with_no_speech_attached(body):
    """The failure this guards against is subtle: a chunk with an empty body would still have a full header, and a
    vector of pure metadata matches anything about that debate. The body is cleaned first, so the row is skipped
    instead.
    """
    row = {"body": body, "date": "2024-03-14", "heading": "Space Industry", "speaker_name": "Ms Smith"}
    assert embedder.render_row(row, embedder.header_recipe()) is None


def test_fetch_page_renders_the_header_from_the_joined_metadata(db, scratch_dsn):
    """The join is the part that can silently go wrong: chunk carries the date and house, but the heading lives on
    debate and the speaker on speech.
    """
    chunk_id = add_chunk_with_context(db, "Shetland hosts the spaceport.")
    with connect(scratch_dsn) as conn:
        page = fetch_page(conn, after_id=0, size=10)
    assert page == [(chunk_id,
                     "[2024-03-14 | Commons | Space Industry | Ms Smith | Labour] Shetland hosts the spaceport.")]


def test_fetch_page_still_embeds_a_chunk_whose_speech_and_debate_rows_are_thin(db, scratch_dsn):
    """LEFT JOINs, deliberately."""
    chunk_id = add_chunk_with_context(db, "Royal Assent.", house=None, heading=None, party=None, suffix="thin")
    with connect(scratch_dsn) as conn:
        assert fetch_page(conn, after_id=0, size=10) == [(chunk_id, "[2024-03-14 | Ms Smith] Royal Assent.")]


def test_fetch_page_reports_an_unembeddable_body_as_none_and_the_run_skips_it(db, scratch_dsn):
    ok = add_chunk_with_context(db, "Real text.", suffix="ok")
    empty = add_chunk_with_context(db, "   ", suffix="empty")
    with connect(scratch_dsn) as conn:
        page = dict(fetch_page(conn, after_id=0, size=10))
    assert page[empty] is None and page[ok].endswith("Real text.")
    result = run(scratch_dsn, FakeEncoder(), index_mode="never", vacuum=False)
    assert (result.embedded, result.skipped) == (1, 1)


# --------------------------------------------------------------------------- status, the index build, and the guard on
# which database gets written ---------------------------------------------------------------------------


def test_status_counts_a_row_under_an_older_recipe_as_pending(db, scratch_dsn):
    """The bug this closes: `pending` counted only NULL vectors, so on the day the recipe changed, status said "0
    pending" while count_pending said 1,074,297 -- and the CLI sizes its progress bar from status.
    """
    ids = add_chunks(db, ["a", "b", "c"])
    with connect(scratch_dsn) as conn:
        write_embeddings(conn, ids[:2], unit(2), recipe="an|older|recipe")
        write_embeddings(conn, ids[2:], unit(1))
        conn.commit()
    s = status(scratch_dsn)
    assert (s.embedded, s.pending, s.never_embedded, s.stale) == (3, 2, 0, 2)


def test_the_embedder_refuses_to_build_when_it_and_the_shared_definition_disagree(monkeypatch):
    """Two copies of the index parameters (here and retrieval.vector_index) are tolerable only because they are checked
    against each other: an index built with other parameters than the ones measured is a silent change.
    """
    monkeypatch.setattr(embedder, "HNSW_M", 8)
    with pytest.raises(RuntimeError, match="disagree"):
        embedder.index_sql()


def test_maintenance_work_mem_is_sized_from_the_row_count():
    """Too small and pgvector finishes the graph one tuple at a time, which is an order of magnitude slower."""
    assert embedder.maintenance_work_mem_for(90_040) == f"{embedder.INDEX_MEM_FLOOR_MB}MB"
    assert embedder.maintenance_work_mem_for(1_074_297, "vector") == "2098MB"
    assert embedder.maintenance_work_mem_for(1_074_297, "halfvec") == "1639MB"  # 1,600 B a row (1,483 measured)
    assert embedder.maintenance_work_mem_for(1_422_379, "halfvec") == "2170MB"  # hansard_v2 today: fits in memory
    assert embedder.maintenance_work_mem_for(1_422_379) == embedder.maintenance_work_mem_for(1_422_379,
                                                                                            embedder.index_form())
    assert embedder.maintenance_work_mem_for(500_000_000) == f"{embedder.INDEX_MEM_CEILING_MB}MB"


def test_the_embedder_refuses_to_write_to_the_rollback_corpus():
    """`hansard` is the only corpus anything was ever measured on and it cannot be rebuilt. It also has no
    embedding_recipe column, so a bulk write there fails late and confusingly; this fails immediately.
    """
    with pytest.raises(SystemExit, match="rollback corpus"):
        embedder.guard_database("postgresql://postgres:postgres@127.0.0.1:5432/hansard")
    target = "postgresql://postgres:postgres@127.0.0.1:5432/hansard_v2"
    assert embedder.guard_database(target) == target
    assert embedder.DEFAULT_DSN.endswith("/hansard_v2"), "the default must not be the protected corpus"
