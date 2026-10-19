"""The HNSW index and the dense leg's ORDER BY are one coupled pair, defined in one place."""

from __future__ import annotations

import json
import re

import pytest

from answer.pipeline import FROZEN_PATH, Frozen, PipelineError
from retrieval import vector_index as V
from retrieval.hybrid import SearchConfig, SearchFilters, fuse_sql


def test_there_are_exactly_two_forms_and_the_dimension_is_the_models():
    from retrieval.gpu_client import EMBEDDING_DIM

    assert V.FORMS == ("vector", "halfvec")
    assert V.DIM == EMBEDDING_DIM == 384


@pytest.mark.parametrize("form", ["vector", "halfvec"])
def test_each_form_orders_by_exactly_the_expression_its_index_is_built_on(form):
    index = V.index_sql(form)
    order = V.order_expression(form, alias="c")
    left = order.split(" <=> ")[0].removeprefix("c.")
    assert f"({V.index_expression(form)} {V.OPCLASS[form]})" in index
    assert V.index_expression(form) in (left, f"({left})"), "the index expression and the ORDER BY's operand differ"
    assert order.endswith(V.CAST[form]), "the query vector must be cast to the same type the index stores"


def test_the_halfvec_index_is_an_expression_index_over_the_unchanged_column():
    sql = V.index_sql("halfvec")
    assert sql == ("CREATE INDEX chunk_embedding_idx ON chunk USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops) "
                   "WITH (m = 16, ef_construction = 64)")
    assert V.index_sql("vector") == ("CREATE INDEX chunk_embedding_idx ON chunk USING hnsw (embedding vector_cosine_ops) "
                                     "WITH (m = 16, ef_construction = 64)")


def test_a_concurrent_build_under_another_name_is_the_same_index():
    sql = V.index_sql("halfvec", name="chunk_embedding_next", concurrently=True)
    assert sql.startswith("CREATE INDEX CONCURRENTLY chunk_embedding_next ON chunk USING hnsw ")
    assert sql.split(" USING ")[1] == V.index_sql("halfvec").split(" USING ")[1]


@pytest.mark.parametrize("bad", ["", "half", "HALFVEC", "bit", None, "halfvec; DROP TABLE chunk"])
def test_an_unknown_form_is_refused_before_any_sql_is_written(bad):
    for build in (V.index_sql, V.order_expression, V.index_expression):
        with pytest.raises(ValueError, match="form"):
            build(bad)


@pytest.mark.parametrize("name", ["chunk embedding", "x;drop", "", "1abc", "a" * 64])
def test_an_index_name_that_is_not_a_plain_identifier_is_refused(name):
    with pytest.raises(ValueError, match="name"):
        V.index_sql("halfvec", name=name)


@pytest.mark.parametrize(("definition", "form"), [
    # what pg_indexes.indexdef prints for each, copied from the live database
    ("CREATE INDEX chunk_embedding_idx ON public.chunk USING hnsw (embedding vector_cosine_ops) "
     "WITH (m='16', ef_construction='64')", "vector"),
    ("CREATE INDEX chunk_embedding_idx ON public.chunk USING hnsw (((embedding)::halfvec(384)) halfvec_cosine_ops) "
     "WITH (m='16', ef_construction='64')", "halfvec"),
    ("CREATE INDEX chunk_embedding_idx ON public.chunk USING hnsw (embedding vector_l2_ops)", None),  # wrong metric
    ("CREATE INDEX chunk_embedding_idx ON public.chunk USING hnsw (((embedding)::halfvec(384)) halfvec_l2_ops)", None),
    ("CREATE INDEX chunk_embedding_idx ON public.chunk USING ivfflat (embedding vector_cosine_ops)", None),
    ("", None),
])
def test_the_form_of_a_live_index_is_read_from_its_definition(definition, form):
    assert V.form_of(definition) == form
    for candidate, pattern in V.INDEXDEF_LIKE.items():  # the gate's SQL spelling agrees with form_of
        assert like(definition, pattern) == (candidate == form), (candidate, definition)


def like(text: str, pattern: str) -> bool:
    """SQL LIKE, for the test: % is any run, _ any one character."""
    regex = "".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern)
    return re.fullmatch(regex, text, flags=re.S) is not None


def test_row_bytes_cover_the_graph_as_measured_in_memory():
    """Sized below the measured in-memory figure, a build quietly takes the slow path."""
    assert V.ROW_BYTES["halfvec"] >= 1_483 * 1.05
    assert V.ROW_BYTES["vector"] >= 2_047
    assert V.ROW_BYTES["halfvec"] < V.ROW_BYTES["vector"]


# --- the dense leg -------------------------------------------------------------------------------------------------


def test_search_config_names_the_index_form_and_refuses_anything_else():
    assert SearchConfig().dense_index == "halfvec", "the default is the index that exists (T3)"
    assert SearchConfig(dense_index="vector").dense_index == "vector"
    with pytest.raises(ValueError, match="dense_index"):
        SearchConfig(dense_index="bit")


def test_the_vector_form_is_byte_for_byte_the_block_11_dense_leg():
    """Unmasked, as the W5 exam's own run sends it."""
    from app import sealed

    with sealed.lifted_for_exam("the W5 exam: the unmasked dense leg (test)"):
        sql = fuse_sql(SearchFilters(), exact=False, dense_index="vector")
    assert ("        SELECT c.id, c.embedding <=> %(qvec)s::vector AS dist\n"
            "        FROM chunk c\n"
            "        WHERE c.embedding IS NOT NULL AND NOT c.is_boilerplate\n"
            "        ORDER BY c.embedding <=> %(qvec)s::vector\n"
            "        LIMIT %(dense_limit)s\n") in sql


def test_the_halfvec_form_selects_by_the_index_but_ranks_by_full_precision():
    """Only candidate SELECTION uses the half-precision index; the rank the leg gives each candidate is still the
    float32 distance, so wherever the halfvec graph finds the same 50 sentences the dense ranks are unchanged.
    """
    sql = fuse_sql(SearchFilters(), exact=False, dense_index="halfvec")
    dense = sql.split("dense AS (")[1].split("),")[0]
    assert "ORDER BY c.embedding::halfvec(384) <=> %(qvec)s::halfvec(384)" in dense
    assert "SELECT c.id, c.embedding <=> %(qvec)s::vector AS dist" in dense
    assert "row_number() OVER (ORDER BY dist, id)" in dense


@pytest.mark.parametrize("form", ["vector", "halfvec"])
def test_the_exact_leg_and_the_speaker_leg_stay_full_precision_whatever_the_index(form):
    """They scan a filtered set exactly, with no index, so there is nothing to gain and precision to lose."""
    exact = fuse_sql(SearchFilters(windows=("W1",)), exact=True, dense_index=form)
    assert "halfvec" not in exact
    assert "member_chunks AS MATERIALIZED (\n    SELECT c.id, c.speech_id, c.embedding <=> %(qvec)s::vector AS dist" in exact


def test_a_filtered_halfvec_search_keeps_its_filter_in_the_indexed_leg():
    sql = fuse_sql(SearchFilters(windows=("W1",)), exact=False, dense_index="halfvec")
    dense = sql.split("dense AS (")[1].split("),")[0]
    assert "window_id" in dense and "::halfvec(384)" in dense


# --- the frozen configuration names the form -----------------------------------------------------------------------


def test_the_frozen_file_names_the_index_form_and_its_hash_covers_it():
    frozen = Frozen.load()
    assert frozen.config.get("dense_index") in V.FORMS
    assert frozen.verify_hash()
    assert frozen.search_config().dense_index == frozen.config["dense_index"]
    assert V.frozen_form() == frozen.config["dense_index"]


def test_a_frozen_config_that_does_not_name_a_knob_is_refused(tmp_path):
    """Symmetric with the unknown-key refusal: a knob the file does not name is decided by a default nobody froze, and
    the hash would describe a configuration that is not the one running. Adding `dense_index` is the case.
    """
    data = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    del data["config"]["dense_index"]
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PipelineError, match="dense_index"):
        Frozen.load(path).search_config()


def test_a_frozen_file_from_before_the_switch_is_read_as_the_full_precision_form(tmp_path):
    data = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    del data["config"]["dense_index"]
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert V.frozen_form(path) == "vector"


def test_the_frozen_file_records_why_and_what_it_replaced():
    """The amendments form an unbroken chain of hashes from E1's freeze to the hash the file carries now."""
    data = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    amendments = data.get("amendments") or []
    assert amendments, "a change to frozen retrieval is logged in the file it changes"
    assert amendments[0]["previous_hash"] == "e57c6cdfd8793d76", "E1's freeze (Session 15)"
    for before, after in zip(amendments, amendments[1:]):
        assert after["previous_hash"] == before["config_hash"]
    assert amendments[-1]["config_hash"] == data["config_hash"]
    for entry in amendments:
        assert re.match(r"\d{4}-\d{2}-\d{2}$", entry["date"]) and entry["changed"]
        assert entry["reason"] and entry["evidence"], "why, and the measurement it rests on"
    assert any("dense_index" in entry["changed"] for entry in amendments)
