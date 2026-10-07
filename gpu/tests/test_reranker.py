"""The reranker."""

import sys
import types

import numpy as np
import pytest

import reranker
from reranker import DEFAULT_RERANK_BATCH, RERANKER_NAME, RERANKER_REVISION, Reranker, score_in_batches

# --------------------------------------------------------------------------- score_in_batches: order, batching, safety
# ---------------------------------------------------------------------------


class Recorder:
    """A forward that scores each text by its length (so the right answer is known) and remembers every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, texts):
        self.calls.append((query, list(texts)))
        return np.array([len(t) for t in texts], dtype=np.float32)


TEXTS = ["a" * 5, "b" * 50, "c" * 1, "d" * 500, "e" * 20, "f" * 3, "g" * 200]


def test_scores_come_back_in_the_order_the_texts_were_given():
    out = score_in_batches("q", TEXTS, Recorder(), batch_size=3)
    assert out.tolist() == [5, 50, 1, 500, 20, 3, 200]


def test_every_text_is_scored_exactly_once_in_batches_no_larger_than_the_limit():
    rec = Recorder()
    score_in_batches("q", TEXTS, rec, batch_size=3)
    assert [len(t) for _, t in rec.calls] == [3, 3, 1]
    assert sorted(t for _, batch in rec.calls for t in batch) == sorted(TEXTS)


def test_longest_texts_go_first_so_an_out_of_memory_happens_before_any_work_is_wasted():
    rec = Recorder()
    score_in_batches("q", TEXTS, rec, batch_size=2)
    lengths = [len(t) for _, batch in rec.calls for t in batch]
    assert lengths == sorted(lengths, reverse=True)


def test_texts_of_equal_length_keep_their_given_order():
    rec = Recorder()
    score_in_batches("q", ["x1", "x2", "x3", "x4"], rec, batch_size=4)
    assert rec.calls[0][1] == ["x1", "x2", "x3", "x4"]


def test_the_query_is_passed_through_unchanged_on_every_call():
    rec = Recorder()
    score_in_batches("What did the minister say?", TEXTS, rec, batch_size=2)
    assert {q for q, _ in rec.calls} == {"What did the minister say?"}


def test_one_call_when_everything_fits_in_a_batch_and_one_per_text_at_batch_size_one():
    rec = Recorder()
    score_in_batches("q", TEXTS, rec, batch_size=100)
    assert len(rec.calls) == 1
    rec = Recorder()
    score_in_batches("q", TEXTS, rec, batch_size=1)
    assert len(rec.calls) == len(TEXTS)


def test_no_texts_means_no_work():
    rec = Recorder()
    out = score_in_batches("q", [], rec)
    assert out.shape == (0,) and out.dtype == np.float32 and rec.calls == []


def test_the_result_is_float32_whatever_forward_returns():
    out = score_in_batches("q", ["a", "bb"], lambda q, t: [float(len(x)) for x in t], batch_size=2)
    assert out.dtype == np.float32


@pytest.mark.parametrize("bad", [0, -1])
def test_a_batch_size_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="batch_size"):
        score_in_batches("q", ["a"], Recorder(), batch_size=bad)


def test_a_forward_that_returns_the_wrong_number_of_scores_is_refused_not_misaligned():
    with pytest.raises(ValueError, match="forward returned scores"):
        score_in_batches("q", ["a", "b", "c"], lambda q, t: np.zeros(len(t) - 1), batch_size=3)


def test_a_failing_forward_propagates_and_nothing_is_swallowed():
    def boom(q, t):
        raise RuntimeError("kernel failed")

    with pytest.raises(RuntimeError, match="kernel failed"):
        score_in_batches("q", ["a"], boom)


def test_the_default_batch_size_keeps_the_worst_case_small():
    assert DEFAULT_RERANK_BATCH == 32  # measured: 32 pairs of 512 tokens peak at +264 MiB


# --------------------------------------------------------------------------- the loader: cache first, download only on
# a miss, fp16 cast before the move to the GPU
# ---------------------------------------------------------------------------


class FakeModel:
    def __init__(self, log):
        self.log = log

    def half(self):
        self.log.append("half")
        return self

    def to(self, device):
        self.log.append(f"to {device}")
        return self


def install_fake_transformers(monkeypatch, cache_miss=False, other_error=None):
    log, calls = [], []

    def from_pretrained(kind):
        def load(name, **options):
            calls.append((kind, name, options))
            if other_error and options["local_files_only"]:
                raise other_error
            if cache_miss and options["local_files_only"]:
                raise OSError("not in the cache")
            return FakeModel(log) if kind == "model" else f"tokenizer-{options['local_files_only']}"

        return types.SimpleNamespace(from_pretrained=load)

    module = types.ModuleType("transformers")
    module.AutoTokenizer = from_pretrained("tokenizer")
    module.AutoModelForSequenceClassification = from_pretrained("model")
    monkeypatch.setitem(sys.modules, "transformers", module)
    return log, calls


def test_the_pinned_model_is_loaded_from_the_local_cache_without_the_network(monkeypatch):
    log, calls = install_fake_transformers(monkeypatch)
    tokenizer, model = reranker._load_reranker("cuda:0", "float16")
    assert tokenizer == "tokenizer-True"
    assert [c[2]["local_files_only"] for c in calls] == [True, True]
    assert {c[1] for c in calls} == {RERANKER_NAME} and {c[2]["revision"] for c in calls} == {RERANKER_REVISION}


def test_a_cache_miss_falls_back_to_a_download_of_the_same_pinned_revision(monkeypatch):
    log, calls = install_fake_transformers(monkeypatch, cache_miss=True)
    tokenizer, _ = reranker._load_reranker("cuda:0", "float16")
    assert tokenizer == "tokenizer-False"
    assert calls[-1][2] == {"revision": RERANKER_REVISION, "local_files_only": False}


def test_an_error_that_is_not_a_cache_miss_is_not_retried_or_hidden(monkeypatch):
    install_fake_transformers(monkeypatch, other_error=RuntimeError("corrupt weights"))
    with pytest.raises(RuntimeError, match="corrupt weights"):
        reranker._load_reranker("cuda:0", "float16")


def test_fp16_is_cast_on_the_cpu_before_the_move_so_the_gpu_never_holds_the_float32_copy(monkeypatch):
    log, _ = install_fake_transformers(monkeypatch)
    reranker._load_reranker("cuda:0", "float16")
    assert log == ["half", "to cuda:0"]


def test_float32_is_not_cast(monkeypatch):
    log, _ = install_fake_transformers(monkeypatch)
    reranker._load_reranker("cuda:0", "float32")
    assert log == ["to cuda:0"]


def test_a_bad_dtype_is_refused_before_anything_is_loaded(monkeypatch):
    log, calls = install_fake_transformers(monkeypatch)
    with pytest.raises(ValueError, match="dtype"):
        Reranker(dtype="bfloat16")
    assert calls == []


def test_the_revision_is_a_full_commit_hash():
    assert len(RERANKER_REVISION) == 40 and set(RERANKER_REVISION) <= set("0123456789abcdef")


def test_the_pair_limit_is_the_models_position_limit_less_the_two_reserved_positions():
    assert reranker.RERANK_MAX_LENGTH == 514 - 2  # XLM-RoBERTa has 514 positions; a longer pair raises inside the model


# --------------------------------------------------------------------------- _forward with a fake tokenizer and model,
# on the CPU: how the pair is tokenised and the score read out
# ---------------------------------------------------------------------------


def make_forward_rig(logits, fail_with=None):
    torch = pytest.importorskip("torch")
    seen = {}

    class Encoded(dict):
        def to(self, device):
            seen["device"] = device
            return self

    def tokenizer(queries, texts, **kwargs):
        seen.update(queries=queries, texts=texts, kwargs=kwargs)
        return Encoded(input_ids=torch.zeros(len(texts), 3, dtype=torch.long))

    def model(**encoded):
        if fail_with:
            raise fail_with
        return types.SimpleNamespace(logits=torch.tensor(logits, dtype=torch.float16))

    rig = object.__new__(Reranker)
    rig._torch, rig.device, rig.tokenizer, rig.model = torch, "cpu", tokenizer, model
    return rig, seen


def test_forward_pairs_the_query_with_every_text_and_truncates_longest_side_first_to_the_model_limit():
    rig, seen = make_forward_rig([[1.0], [2.0]])
    rig._forward("the query", ["first", "second"])
    assert seen["queries"] == ["the query", "the query"] and seen["texts"] == ["first", "second"]
    assert seen["kwargs"] == {"padding": True, "truncation": "longest_first", "max_length": 512, "return_tensors": "pt"}
    assert seen["device"] == "cpu"


def test_forward_returns_the_relevance_logit_as_float32_numpy_one_per_text():
    rig, _ = make_forward_rig([[1.5], [-3.0], [0.25]])
    out = rig._forward("q", ["a", "b", "c"])
    assert isinstance(out, np.ndarray) and out.dtype == np.float32 and out.tolist() == [1.5, -3.0, 0.25]


def test_forward_turns_a_cuda_out_of_memory_into_gpu_out_of_memory():
    torch = pytest.importorskip("torch")
    rig, _ = make_forward_rig([[0.0]], fail_with=torch.cuda.OutOfMemoryError("CUDA out of memory"))
    with pytest.raises(reranker.GpuOutOfMemory, match="1 pairs"):
        rig._forward("q", ["a"])


def test_forward_lets_any_other_error_through_untouched():
    rig, _ = make_forward_rig([[0.0]], fail_with=RuntimeError("kernel failed"))
    with pytest.raises(RuntimeError, match="kernel failed"):
        rig._forward("q", ["a"])


# --------------------------------------------------------------------------- the real model on the GPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    return Reranker()


NHS_QUERY = "What is the Government doing about NHS waiting lists?"
NHS_RELEVANT = "The Government are investing billions of pounds in the NHS to cut waiting lists for hospital treatment."
UNRELATED = "The recipe calls for two eggs, a cup of flour and a pinch of salt."


def test_gpu_a_relevant_passage_scores_far_above_an_irrelevant_one(model):
    relevant, irrelevant = model.score(NHS_QUERY, [NHS_RELEVANT, UNRELATED])
    assert relevant - irrelevant > 5  # measured gap: about 10 logits


def test_gpu_the_right_passage_ranks_first_wherever_it_sits_in_the_list(model):
    fillers = [UNRELATED, "The Lords will sit at 3 pm tomorrow.", "Rail fares rose by 4 per cent in March."]
    for position in range(4):
        texts = fillers[:position] + [NHS_RELEVANT] + fillers[position:]
        assert int(np.argmax(model.score(NHS_QUERY, texts))) == position


def test_gpu_scores_are_finite_float32_one_per_text(model):
    out = model.score(NHS_QUERY, [NHS_RELEVANT, UNRELATED, "Yes."])
    assert out.dtype == np.float32 and out.shape == (3,) and np.isfinite(out).all()


def test_gpu_more_texts_than_one_batch_are_scored_in_order(model):
    texts = [UNRELATED] * 40 + [NHS_RELEVANT] + [UNRELATED] * 40  # 81 pairs: three batches of 32
    out = model.score(NHS_QUERY, texts)
    assert out.shape == (81,) and int(np.argmax(out)) == 40 and np.isfinite(out).all()


def test_gpu_a_text_scores_the_same_alone_as_inside_a_batch(model):
    alone = model.score(NHS_QUERY, [NHS_RELEVANT])[0]
    inside = model.score(NHS_QUERY, [UNRELATED, NHS_RELEVANT, "Short."] * 10)[1]
    assert abs(alone - inside) < 0.1  # fp16 padding noise only


def test_gpu_the_same_call_twice_gives_the_same_scores(model):
    texts = [NHS_RELEVANT, UNRELATED, "Waiting lists are too long."]
    assert np.array_equal(model.score(NHS_QUERY, texts), model.score(NHS_QUERY, texts))


@pytest.mark.parametrize(
    "query,text",
    [
        (NHS_QUERY, "waiting lists " * 5_000),  # a passage far beyond 512 tokens
        (NHS_QUERY, "x" * 200_000),  # one enormous word
        ("waiting lists " * 2_000, NHS_RELEVANT),  # a query beyond 512 tokens
        ("waiting lists " * 2_000, "hospital treatment " * 2_000),  # both too long
    ],
    ids=["long-passage", "one-huge-word", "long-query", "both-long"],
)
def test_gpu_over_length_pairs_are_truncated_not_rejected(model, query, text):
    out = model.score(query, [text])
    assert out.shape == (1,) and np.isfinite(out).all()


def test_gpu_a_full_batch_of_maximum_length_pairs_stays_small(model):
    import torch

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model.score("short query", ["parliament debate " * 2_000] * DEFAULT_RERANK_BATCH)
    assert (torch.cuda.max_memory_allocated() - before) / 2**20 < 600  # measured +264 MiB


def test_gpu_an_out_of_memory_becomes_gpu_out_of_memory_and_the_model_still_works(model, monkeypatch):
    import torch

    real_forward = model.model.forward

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(model.model, "forward", oom)
    with pytest.raises(reranker.GpuOutOfMemory):
        model.score(NHS_QUERY, [NHS_RELEVANT])
    monkeypatch.setattr(model.model, "forward", real_forward)
    assert np.isfinite(model.score(NHS_QUERY, [NHS_RELEVANT])).all()


def test_gpu_the_loaded_reranker_is_the_pinned_revision_in_fp16(model):
    assert model.revision == RERANKER_REVISION and model.dtype == "float16"
    assert next(model.model.parameters()).dtype == __import__("torch").float16
