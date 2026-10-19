# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tokenizer"""

# pylint: disable=protected-access

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import tokenizer
from maxtext.trainers.tokenizer import train_tokenizer
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT

from tests.utils.test_helpers import ensure_tokenizer_downloaded

TEST_REVISION = "01234567" * 5


class PinnedTokenizerRevisionTest(unittest.TestCase):
  """Tests revision plumbing without network access."""

  def test_hf_tokenizer_passes_revision(self):
    loaded_tokenizer = mock.MagicMock(
        pad_token_id=0,
        unk_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
    )
    with mock.patch.object(
        tokenizer.transformers.AutoTokenizer,
        "from_pretrained",
        return_value=loaded_tokenizer,
    ) as mock_load:
      tokenizer.HFTokenizer(
          "example-org/example-model",
          add_bos=False,
          add_eos=True,
          hf_access_token="test-token",
          tokenizer_revision=TEST_REVISION,
      )

    mock_load.assert_called_once_with(
        "example-org/example-model",
        add_bos_token=False,
        add_eos_token=True,
        token="test-token",
        revision=TEST_REVISION,
    )

  def test_hf_tokenizer_normalizes_empty_revision(self):
    loaded_tokenizer = mock.MagicMock(
        pad_token_id=0,
        unk_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
    )
    with mock.patch.object(
        tokenizer.transformers.AutoTokenizer,
        "from_pretrained",
        return_value=loaded_tokenizer,
    ) as mock_load:
      tokenizer.HFTokenizer("example-org/example-model", False, False, None, "")

    self.assertIsNone(mock_load.call_args.kwargs["revision"])

  def test_tokenizer_cache_key_includes_revision(self):
    config = SimpleNamespace(
        tokenizer_path="example-org/example-model",
        tokenizer_type="huggingface",
        add_bos=False,
        add_eos=False,
        hf_access_token="test-token",
        tokenizer_revision="a" * 40,
    )
    built = SimpleNamespace(pad_id=0, unk_id=1)
    data_processing_utils._build_tokenizer_cached.cache_clear()
    self.addCleanup(data_processing_utils._build_tokenizer_cached.cache_clear)

    with mock.patch.object(tokenizer, "build_tokenizer", return_value=built) as mock_build:
      data_processing_utils.get_tokenizer_and_pad_id(config)
      data_processing_utils.get_tokenizer_and_pad_id(config)
      config.tokenizer_revision = "b" * 40
      data_processing_utils.get_tokenizer_and_pad_id(config)

    self.assertEqual(mock_build.call_count, 2)
    self.assertEqual(mock_build.call_args_list[0].args[-1], "a" * 40)
    self.assertEqual(mock_build.call_args_list[1].args[-1], "b" * 40)


@unittest.skipIf(is_decoupled(), "Bypassed in offline decoupled runs (no GCS/internet)")
class TrainTokenizerTest(unittest.TestCase):
  """Tests for train_tokenizer.py using data from Parquet files"""

  @classmethod
  def setUpClass(cls):
    # the test only use ~10Mb of data, one file is enough, more files cause slow down
    grain_train_files = "gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet"
    cls.vocab_size = 32_768
    cls.max_corpus_chars = 10_000_000
    assets_path = "tests"
    vocab_model_name = "test_tokenizer"
    cls.tokenizer_path = os.path.join(assets_path, vocab_model_name)
    cls.source_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
        "sentencepiece",
        add_bos=False,
        add_eos=False,
    )
    dataset_iter = train_tokenizer.build_grain_iterator(grain_train_files, "parquet")
    train_tokenizer.train_tokenizer(
        dataset_iter,
        vocab_path=cls.tokenizer_path,
        vocab_size=cls.vocab_size,
        max_corpus_chars=cls.max_corpus_chars,
    )
    cls.test_tokenizer = input_pipeline_utils.get_tokenizer(
        cls.tokenizer_path, "sentencepiece", add_bos=False, add_eos=False
    )

  @classmethod
  def tearDownClass(cls):
    os.remove(cls.tokenizer_path)

  def test_tokenize(self):
    text = "This is a test"
    self.assertTrue(np.array_equal(self.source_tokenizer.encode(text), self.test_tokenizer.encode(text)))

  def test_detokenize(self):
    tokens = [66, 12, 10, 702]
    self.assertEqual(np.asarray(self.source_tokenizer.decode(tokens)), np.asarray(self.test_tokenizer.decode(tokens)))


@unittest.skipIf(is_decoupled(), "Bypassed in offline decoupled runs (no GCS/internet)")
class TikTokenTest(unittest.TestCase):
  """Tests for TikToken"""

  @classmethod
  def setUpClass(cls):
    grain_train_files = "gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet"
    cls.source_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
        "tiktoken",
        add_bos=False,
        add_eos=False,
    )
    cls.dataset = train_tokenizer.build_grain_iterator(grain_train_files, "parquet")

  def test_tokenize(self):
    text = "This is a test"
    tokens = [2028, 374, 264, 1296]
    self.assertTrue(np.array_equal(self.source_tokenizer.encode(text), tokens))

  def test_detokenize(self):
    tokens = [2028, 374, 264, 1296]
    text = "This is a test"
    self.assertEqual(np.asarray(self.source_tokenizer.decode(tokens)), np.asarray(text))


@unittest.skipIf(is_decoupled(), "Bypassed in offline decoupled runs (no GCS/internet)")
class HFTokenizerTest(unittest.TestCase):
  """Tests for HFTokenizer"""

  @classmethod
  def setUpClass(cls):
    gemma2_path = ensure_tokenizer_downloaded("gemma2-2b", skip_test_on_failure=False)
    cls.hf_tokenizer = input_pipeline_utils.get_tokenizer(gemma2_path, "huggingface", add_bos=False, add_eos=False)
    cls.sp_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.gemma"), "sentencepiece", add_bos=False, add_eos=False
    )

  def test_tokenize(self):
    text = "This is a test"
    self.assertTrue(np.array_equal(self.hf_tokenizer.encode(text), self.sp_tokenizer.encode(text)))


if __name__ == "__main__":
  unittest.main()
