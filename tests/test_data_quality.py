"""Regression tests for full-corpus quality evaluation; no model downloads."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _shared
import evaluate_alignment as alignment
import evaluate_diversity as diversity
import prepare_synthetic as prepare


class CachedEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode(self, texts, **kwargs):
        assert len(texts) == len(self.embeddings)
        return self.embeddings


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.enterContext(patch.object(_shared, "SYNTHETIC_DIR", self.root))
        self.enterContext(patch.object(prepare, "SYNTHETIC_DIR", self.root))

    def pool(self, task, label, texts, config="c3", run=2, lower=False):
        path = self.root / task / config / f"run{run}" / label.lower() / "data.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "Text": texts, "Label": [label] * len(texts),
            "Domain": ["Healthcare"] * len(texts),
        })
        if lower:
            df = df.rename(columns={"Text": "text", "Label": "label"})
        df.to_csv(path, index=False)
        return path

    def test_t2_matches_training_assembly_and_keeps_generation_hypotheses(self):
        for config in ("c3", "c2_claude-haiku-4.5"):
            with self.subTest(config=config):
                self.pool("t2", "Functional", ["shared repeated text", "functional only text"], config)
                self.pool("t1", "Security", ["shared repeated text"] * 2, config, lower=True)
                self.pool("t1", "Availability", ["availability only text"], config)
                self.pool("t1", "Security", ["wrong run text"], config, run=1)
                samples = alignment.load_generated_data("t2", config, 2)
                texts = diversity.load_texts("t2", config, 2)
                training = pd.concat([
                    prepare._load_raw("t2", config, 2),
                    prepare._load_raw("t1", config, 2).assign(label="Non-Functional"),
                ], ignore_index=True)
                self.assertEqual(texts, training["text"].tolist())
                self.assertEqual(texts, [s["text"] for s in samples])
                self.assertEqual(texts.count("shared repeated text"), 3)
                self.assertEqual([s["features"]["label"] for s in samples],
                                 ["Functional", "Functional", "Availability", "Security", "Security"])
                hypotheses = [alignment.verbalize_features(s["features"]) for s in samples]
                self.assertIn("label is Security", hypotheses[-1])
                self.assertNotIn("Non-Functional", hypotheses[-1])

    def test_missing_reused_pool_is_an_error_for_both_evaluators(self):
        self.pool("t2", "Functional", ["functional only text"])
        self.pool("t1", "Security", ["wrong run text"], run=1)
        for loader in (alignment.load_generated_data, diversity.load_texts):
            with self.subTest(loader=loader.__name__):
                with self.assertRaises(FileNotFoundError):
                    loader("t2", "c3", 2)

    def test_reused_directory_without_csv_is_an_error(self):
        self.pool("t2", "Functional", ["functional only text"])
        (self.root / "t1/c3/run2").mkdir(parents=True)
        for loader in (alignment.load_generated_data, diversity.load_texts):
            with self.assertRaises(FileNotFoundError):
                loader("t2", "c3", 2)

    def test_empty_reused_pool_is_an_error(self):
        self.pool("t2", "Functional", ["functional only text"])
        self.pool("t1", "Security", [])
        for loader in (alignment.load_generated_data, diversity.load_texts):
            with self.assertRaises(ValueError):
                loader("t2", "c3", 2)

    def test_invalid_text_is_not_silently_dropped(self):
        self.pool("t2", "Functional", ["functional only text"])
        for invalid in (None, "   "):
            self.pool("t1", "Security", [invalid])
            for loader in (alignment.load_generated_data, diversity.load_texts):
                with self.assertRaises(ValueError):
                    loader("t2", "c3", 2)

    def test_other_tasks_load_only_their_own_pool(self):
        for task in ("t1", "t3", "t4"):
            expected = [f"{task} first text", f"{task} repeated text", f"{task} repeated text"]
            self.pool(task, "Security", expected)
            self.assertEqual(diversity.load_texts(task, "c3", 2), expected)
            self.assertEqual([s["text"] for s in alignment.load_generated_data(task, "c3", 2)], expected)

    def test_alignment_scores_every_row_with_its_original_hypothesis(self):
        self.pool("t2", "Functional", ["functional one text", "functional two text"])
        self.pool("t1", "Security", ["security only text"])
        samples = alignment.load_generated_data("t2", "c3", 2)
        seen = []

        def nli(pairs, **kwargs):
            seen.extend(pairs)
            return [[{"label": "entailment", "score": 0.8 if "label is Functional" in p["text_pair"] else 0.2}]
                    for p in pairs]

        result = alignment.compute_alignment(samples, nli)
        self.assertEqual(result["count"], 3)
        self.assertEqual([p["text"] for p in seen], [s["text"] for s in samples])
        self.assertAlmostEqual(result["mean"], 0.6)  # Count-weighted, not the two-class mean (0.5).

    def test_missing_hypothesis_is_not_silently_skipped(self):
        path = self.pool("t1", "Security", ["security only text"])
        pd.DataFrame({"Text": ["security only text"]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            alignment.load_generated_data("t1", "c3", 2)
        with self.assertRaises(ValueError):
            alignment.compute_alignment([{"text": "text", "features": {}}], None)


class DiversityTests(unittest.TestCase):
    def test_ingf_matches_document_frequency_definition(self):
        corpora = [[], ["two words"], ["A b c a b c", "a b c", "a b c.", "x y z"],
                   ["same repeated text"] * 3]
        for texts in corpora:
            with self.subTest(texts=texts):
                grams = [diversity.compute_trigrams(t) for t in texts]
                vocabulary = set().union(*grams) if grams else set()
                expected = (sum(sum(g in s for s in grams) / len(texts) for g in vocabulary)
                            / len(vocabulary)) if vocabulary else 0.0
                self.assertAlmostEqual(diversity.compute_ingf(texts), expected)
        self.assertEqual(diversity.compute_trigrams("A B C."), {"a b c."})

    def test_small_aps_keeps_cross_pool_pairs_duplicates_and_negative_cosines(self):
        # Two FR rows and one NFR row; the FR-NFR pairs have cosine -1.
        embeddings = np.array([[2., 0.], [2., 0.], [-3., 0.]], dtype=np.float32)
        result = diversity.compute_aps(["fr", "fr", "nfr"], CachedEmbeddings(embeddings))
        self.assertAlmostEqual(result, -1 / 3, places=7)

    def test_large_aps_preserves_original_draws_and_reductions(self):
        n = 10001
        embeddings = np.random.default_rng(123).normal(size=(n, 3)).astype(np.float32)
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        rng = np.random.default_rng(42)
        total, count = 0.0, 0
        for _ in range(10):
            left = rng.integers(0, n, size=500000)
            right = rng.integers(0, n, size=500000)
            keep = left != right
            scores = np.sum(normalized[left[keep]] * normalized[right[keep]], axis=1)
            total += float(np.sum(scores))
            count += len(scores)
        with contextlib.redirect_stdout(io.StringIO()):
            actual = diversity.compute_aps(["text"] * n, CachedEmbeddings(embeddings))
        self.assertEqual(actual, total / count)


if __name__ == "__main__":
    unittest.main()
