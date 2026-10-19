# Copyright 2023–2026 Google LLC
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

"""Tests for decoded training-text logging."""

# pylint: disable=missing-function-docstring,protected-access

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from maxtext.common.metric_logger import MetricLogger
from maxtext.input_pipeline import data_processing_utils


class _Tokenizer:

  def decode(self, tokens):
    return "<" + ",".join(str(token) for token in tokens) + ">"


class MetricLoggerTextTest(unittest.TestCase):

  def _logger(self, **config_overrides):
    logger = MetricLogger.__new__(MetricLogger)
    config = {
        "log_text_period": 1,
        "log_text_num_samples": 1,
        "log_text_num_docs": 1,
        "log_text_num_tokens": -1,
        "enable_tensorboard": True,
    }
    config.update(config_overrides)
    logger.config = SimpleNamespace(**config)
    logger.writer = mock.MagicMock()
    logger._text_tokenizer = _Tokenizer()
    return logger

  @staticmethod
  def _packed_batch():
    return {
        "inputs": np.array([[101, 102, 201, 202, 0, 0]]),
        "targets": np.array([[102, 103, 202, 203, 0, 0]]),
        "inputs_segmentation": np.array([[1, 1, 2, 2, 0, 0]]),
        "targets_segmentation": np.array([[1, 1, 2, 2, 0, 0]]),
    }

  @mock.patch("jax.process_index", return_value=0)
  def test_logs_selected_packed_document_to_console_and_tensorboard(self, _):
    logger = self._logger(log_text_num_docs=1)

    with mock.patch("maxtext.common.metric_logger.max_logging.log") as log:
      logger.maybe_log_text_samples(self._packed_batch(), step=4)

    console = "\n".join(call.args[0] for call in log.call_args_list)
    self.assertIn("doc 1/2", console)
    self.assertIn("<101,102>", console)
    self.assertNotIn("<201,202>", console)
    logger.writer.add_text.assert_called_once()
    tag, tensorboard_text, logged_step = logger.writer.add_text.call_args.args
    self.assertEqual(tag, "text_samples")
    self.assertEqual(logged_step, 4)
    self.assertIn("Doc 1/2", tensorboard_text)
    self.assertIn("<101,102>", tensorboard_text)
    self.assertNotIn("<201,202>", tensorboard_text)

  @mock.patch("jax.process_index", return_value=0)
  def test_period_gate_avoids_tokenizer_and_writer_work(self, _):
    logger = self._logger(log_text_period=10)
    del logger._text_tokenizer

    with (
        mock.patch.object(logger, "_get_tokenizer") as get_tokenizer,
        mock.patch("maxtext.common.metric_logger.max_logging.log") as log,
    ):
      logger.maybe_log_text_samples(self._packed_batch(), step=3)

    get_tokenizer.assert_not_called()
    log.assert_not_called()
    logger.writer.add_text.assert_not_called()

  def test_head_tail_token_and_text_views(self):
    tokens = [1, 2, 3, 4, 5, 6]

    token_view, description = MetricLogger._format_token_view(tokens, num_tokens=2)
    text_view = MetricLogger._decode_trimmed(_Tokenizer(), tokens, num_tokens=2)

    self.assertEqual(token_view, "[1, 2, ..., 5, 6]")
    self.assertEqual(description, "6, first 2 + last 2")
    self.assertEqual(text_view, "<1,2> ... <5,6>")

  def test_zero_token_view_hides_content(self):
    tokens = [1, 2, 3]

    token_view, description = MetricLogger._format_token_view(tokens, num_tokens=0)
    text_view = MetricLogger._decode_trimmed(_Tokenizer(), tokens, num_tokens=0)

    self.assertEqual(token_view, "[]")
    self.assertEqual(description, "3, hidden")
    self.assertEqual(text_view, "")

  @mock.patch("jax.process_index", return_value=0)
  def test_logging_failure_warns_without_raising(self, _):
    logger = self._logger()
    del logger._text_tokenizer

    with (
        mock.patch.object(logger, "_get_tokenizer", side_effect=RuntimeError("tokenizer unavailable")),
        mock.patch("maxtext.common.metric_logger.max_logging.log") as log,
    ):
      logger.maybe_log_text_samples(self._packed_batch(), step=0)

    self.assertIn("WARNING", log.call_args.args[0])
    self.assertIn("tokenizer unavailable", log.call_args.args[0])
    logger.writer.add_text.assert_not_called()

  @mock.patch("jax.process_index", return_value=0)
  def test_sequence_shard_is_labeled_as_a_local_fragment(self, _):
    class _RemoteArray:

      def __init__(self, local):
        self.shape = (2, 12)
        self.addressable_shards = [SimpleNamespace(data=local)]

      def __array__(self, *args, **kwargs):
        raise RuntimeError("array spans hosts")

    logger = self._logger()
    batch = {name: _RemoteArray(value) for name, value in self._packed_batch().items()}
    with mock.patch("maxtext.common.metric_logger.max_logging.log") as log:
      logger.maybe_log_text_samples(batch, 0)
    console = "\n".join(call.args[0] for call in log.call_args_list)
    self.assertIn("local sequence fragment (6 of 12 sequence positions)", console)
    self.assertIn("local sequence fragment (6 of 12 sequence positions)", logger.writer.add_text.call_args.args[1])

  def test_hf_decoder_uses_the_pipeline_backend_and_revision(self):
    logger = self._logger(
        dataset_type="hf",
        tokenizer_type="sentencepiece",
        tokenizer_path="example/model",
        tokenizer_revision="revision",
        use_sft=True,
        add_bos=True,
        add_eos=True,
        hf_access_token=None,
    )
    del logger._text_tokenizer
    loaded = SimpleNamespace(pad_id=0)
    with mock.patch.object(data_processing_utils, "_build_tokenizer_cached", return_value=loaded) as build:
      self.assertIs(logger._get_tokenizer(), loaded)
      self.assertIs(logger._get_tokenizer(), loaded)
    build.assert_called_once_with("example/model", "huggingface", False, False, None, "revision")

  @mock.patch("jax.process_index", return_value=0)
  def test_tokenizer_failure_is_cached_and_warned_once(self, _):
    logger = self._logger()
    del logger._text_tokenizer
    with (
        mock.patch.object(
            data_processing_utils, "get_tokenizer_and_pad_id", side_effect=RuntimeError("unavailable")
        ) as build,
        mock.patch("maxtext.common.metric_logger.max_logging.log") as log,
    ):
      logger.maybe_log_text_samples(self._packed_batch(), 0)
      logger.maybe_log_text_samples(self._packed_batch(), 1)
    build.assert_called_once()
    self.assertEqual(sum("WARNING" in call.args[0] for call in log.call_args_list), 1)
    logger.writer.add_text.assert_not_called()


if __name__ == "__main__":
  unittest.main()
