"""
Unit tests for pure helpers in train.py (standardize, collate, tokenizer hook, config).
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

# Heavy training-stack imports are not needed for these unit tests; stub before `import train`.
for _name in ("wandb", "datasets", "underthesea", "transformers"):
    sys.modules.setdefault(_name, MagicMock())

# tqdm is imported by train.py; return the iterable unchanged from ``tqdm(...)``.
_tqdm_mod = types.ModuleType("tqdm")
_tqdm_mod.tqdm = lambda x, **kwargs: x  # type: ignore[misc, assignment]
sys.modules["tqdm"] = _tqdm_mod

_sklearn_metrics = types.ModuleType("sklearn.metrics")


def _f1_stub(y_true, y_pred, average=None, zero_division=0):
    y = np.asarray(y_true)
    if y.size == 0:
        return np.array([1.0], dtype=float) if average is None else 1.0
    n = len(np.unique(y))
    if average is None:
        return np.ones(n, dtype=float)
    return 1.0


def _cm_stub(y_true, y_pred, labels=None):
    labs = labels if labels is not None else list(range(int(np.max(np.asarray(y_true)) + 1)))
    k = len(labs)
    return np.zeros((k, k), dtype=int)


_sklearn_metrics.f1_score = _f1_stub
_sklearn_metrics.confusion_matrix = _cm_stub
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
sys.modules["sklearn.metrics"] = _sklearn_metrics

if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")
if "src.models" not in sys.modules:
    sys.modules["src.models"] = types.ModuleType("src.models")
_model_mod = types.ModuleType("src.models.model")
_model_mod.MambaClassifier = MagicMock()
sys.modules["src.models.model"] = _model_mod

import torch

from train import (
    _standardize_one,
    collate_fn,
    get_default_config,
    standardize_data,
    tokenize_function,
)


class FakeTokenizer:
    """Minimal tokenizer stand-in: records arguments; returns HF-like output dicts."""

    vocab_size = 128

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding == "max_length"
        assert truncation is True
        assert max_length == 8
        if isinstance(texts, str):
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[101, 102, 0, 0, 0, 0, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]]),
            }
        assert return_tensors is None
        return {
            "input_ids": [
                [101, 102, 0, 0, 0, 0, 0, 0],
                [201, 202, 203, 0, 0, 0, 0, 0],
            ],
            "attention_mask": [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
            ],
        }


class TestStandardizeOne:
    """AAA: core punctuation removal, casing, and spacing for a single string."""

    def test_lowercases_and_replaces_punctuation_with_spaces(self):
        text = 'Hello, "World"!'
        out = _standardize_one(text)
        assert out == "hello world"

    def test_strips_trailing_sentence_punctuation_then_normalizes(self):
        # Trailing . , ? removed by regex before other replaces
        assert _standardize_one("  OK.  ") == "ok"
        assert _standardize_one("Really?") == "really"

    def test_collapses_internal_whitespace(self):
        assert _standardize_one("a   b\t\nc") == "a b c"

    def test_empty_string_returns_empty(self):
        assert _standardize_one("") == ""


class TestStandardizeData:
    def test_single_string_example_mutates_text_key(self):
        examples = {"text": "Foo, BAR.", "label": 1}
        out = standardize_data(examples)
        assert out is examples
        assert out["text"] == "foo bar"
        assert out["label"] == 1

    def test_batched_list_standardizes_each_row(self):
        examples = {
            "text": ["One, Two.", "THREE"],
            "label": [0, 1],
        }
        out = standardize_data(examples)
        assert out["text"] == ["one two", "three"]
        assert out["label"] == [0, 1]


class TestCollateFn:
    def test_pads_variable_length_input_ids_and_stacks_labels(self):
        # Short and long sequences to exercise pad_sequence + unsqueeze
        batch = [
            {"input_ids": [1, 2, 3], "labels": 0},
            {"input_ids": [9, 8, 7, 6, 5], "labels": 1},
        ]
        out = collate_fn(batch)
        assert out["labels"].shape == (2,)
        assert out["labels"].tolist() == [0, 1]
        # (batch, max_len, 1)
        assert out["input_ids"].dim() == 3
        assert out["input_ids"].shape == (2, 5, 1)
        # First row padded with zeros after three tokens
        row0 = out["input_ids"][0, :, 0].tolist()
        assert row0[:3] == [1, 2, 3]
        assert row0[3:] == [0, 0]

    def test_single_item_batch_shape(self):
        batch = [{"input_ids": [7, 7], "labels": 3}]
        out = collate_fn(batch)
        assert out["labels"].tolist() == [3]
        assert out["input_ids"].shape == (1, 2, 1)


class TestGetDefaultConfig:
    def test_returns_expected_keys_and_types(self):
        cfg = get_default_config()
        assert cfg["d_model"] == 256
        assert cfg["n_layers"] == 4
        assert cfg["num_classes"] == 3
        assert cfg["batch_size"] == 32
        assert cfg["max_length"] == 512
        assert cfg["seed"] == 42
        assert cfg["dataset_name"] == "uitnlp/vietnamese_students_feedback"
        assert cfg["tokenizer_name"] == "vinai/phobert-base"
        assert cfg["max_patience"] == 5


class TestTokenizeFunction:
    def test_single_string_uses_pt_tensors_and_appends_labels(self):
        tok = FakeTokenizer()
        examples = {"text": "dummy", "label": torch.tensor(1)}
        out = tokenize_function(examples, tok, max_length=8)
        assert "input_ids" in out and "attention_mask" in out
        assert out["labels"] is examples["label"]
        assert isinstance(out["input_ids"], torch.Tensor)

    def test_batched_strings_returns_lists_and_appends_labels(self):
        tok = FakeTokenizer()
        examples = {"text": ["a", "b b"], "label": [0, 1]}
        out = tokenize_function(examples, tok, max_length=8)
        assert isinstance(out["input_ids"], list)
        assert len(out["input_ids"]) == 2
        assert out["labels"] == [0, 1]


class TestStandardizeInputValidation:
    """Behavior for invalid types: function is not defensive; document failure mode."""

    def test_non_string_raises_typeError(self):
        with pytest.raises(TypeError):
            _standardize_one(123)  # replace/re.sub expect str
