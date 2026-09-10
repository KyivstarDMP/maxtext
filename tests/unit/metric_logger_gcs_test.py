# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ensure GCS exports retain final evaluation independently of training flushes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maxtext.common import metric_logger


def make_logger(monkeypatch, tmp_path, log_period=1):
  """Capture uploads in memory while preserving the writer's naming behavior."""
  logger = metric_logger.MetricLogger.__new__(metric_logger.MetricLogger)
  logger.config = SimpleNamespace(run_name="test", steps=80, log_period=log_period, metrics_dir="gs://unused/test")
  logger.running_gcs_metrics = []
  objects = {}

  def upload(destination, source):
    objects[destination] = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines()]

  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(metric_logger.gcs_utils, "upload_blob", upload)
  return logger, objects


@pytest.mark.parametrize("order", [("eval", "train"), ("train", "eval")])
def test_final_evaluation_and_training_both_survive(monkeypatch, tmp_path, order):
  logger, objects = make_logger(monkeypatch, tmp_path)
  metrics = {"train": {"scalar": {"learning/lm_loss": 0.5}}, "eval": {"scalar": {"eval/avg_loss": 0.75}}}
  for kind in order:
    logger.write_metrics_for_gcs(metrics[kind], 79, kind)
  rows = [row for records in objects.values() for row in records]
  assert sum("learning/lm_loss" in row for row in rows) == 1
  assert sum("eval/avg_loss" in row for row in rows) == 1
  assert all(row["step"] == 79 for row in rows)
  assert logger.running_gcs_metrics == []


def test_eval_does_not_flush_or_erase_pending_training(monkeypatch, tmp_path):
  logger, objects = make_logger(monkeypatch, tmp_path, log_period=10)
  logger.write_metrics_for_gcs({"scalar": {"learning/lm_loss": 1}}, 78, "train")
  logger.write_metrics_for_gcs({"scalar": {"eval/avg_loss": 2}}, 79, "eval")
  assert len(logger.running_gcs_metrics) == 1
  logger.write_metrics_for_gcs({"scalar": {"learning/lm_loss": 3}}, 79, "train")
  rows = [row for records in objects.values() for row in records]
  assert sorted(row["learning/lm_loss"] for row in rows if "learning/lm_loss" in row) == [1, 3]
  assert [row["eval/avg_loss"] for row in rows if "eval/avg_loss" in row] == [2]


def test_distinct_eval_producers_at_same_step_both_survive(monkeypatch, tmp_path):
  logger, objects = make_logger(monkeypatch, tmp_path)
  logger.write_metrics_for_gcs({"scalar": {"eval/avg_loss": 2}}, 79, "eval")
  logger.write_metrics_for_gcs({"scalar": {"per_dataset_eval_loss/a": 3}}, 79, "eval")
  rows = [row for records in objects.values() for row in records]
  assert len(rows) == 2
  assert {k for row in rows for k in row if "loss" in k} == {"eval/avg_loss", "per_dataset_eval_loss/a"}


def test_midrun_eval_is_durable_without_following_train(monkeypatch, tmp_path):
  logger, objects = make_logger(monkeypatch, tmp_path, log_period=10)
  logger.write_metrics_for_gcs({"scalar": {"eval/avg_loss": 2}}, 25, "eval")
  rows = [row for records in objects.values() for row in records]
  assert len(rows) == 1 and rows[0]["step"] == 25 and rows[0]["eval/avg_loss"] == 2
