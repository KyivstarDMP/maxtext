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

"""Instruction data processing test."""

import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import datasets

from maxtext.input_pipeline import instruction_data_processing
from maxtext.input_pipeline import data_processing_utils


class InstructionDataProcessingTest(unittest.TestCase):
  """Test instruction data processing."""

  def _run_math_qa_test(self, template_config, example_input, expected_user, expected_assistant):
    """Helper to run math_qa_formatting tests."""
    result = instruction_data_processing.math_qa_formatting(example_input, template_config=template_config)

    messages = {msg["role"]: msg["content"] for msg in result["messages"]}
    self.assertEqual(messages.get("user"), expected_user)
    self.assertEqual(messages.get("assistant"), expected_assistant)

  def test_load_data_template_from_file(self):
    template_config = instruction_data_processing.load_data_template_from_file(
        "maxtext/examples/chat_templates/gsm8k_rl.json"
    )
    self.assertEqual(
        template_config,
        {
            "SYSTEM_PROMPT": (
                "You are given a problem. Think about the problem and provide"
                " your reasoning. Place it between {reasoning_start_token} and"
                " {reasoning_end_token}. Then, provide the final answer (i.e.,"
                " just one numerical value) between {solution_start_token} and"
                " {solution_end_token}."
            ),
            "TEMPLATE": ("<start_of_turn>user\n{system_prompt}\n\n{question}<end_of_turn>\n<start_of_turn>model"),
        },
    )

  def test_math_qa_formatting_with_prompt_completion_template(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={"question": "What is 2 + 2?", "answer": "Because 2 and 2 make 4.\n ## 4"},
        expected_user="This is a question: What is 2 + 2?",
        expected_assistant="<reasoning>\nBecause 2 and 2 make 4.\n</reasoning>\n<answer>\n4\n</answer>",
    )

  def test_math_qa_formatting_with_no_reasoning_template(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "The answer is: {answer}",
        },
        example_input={"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        expected_user="This is a question: What is the capital of France?",
        expected_assistant="The answer is: The capital of France is Paris.",
    )

  def test_math_qa_formatting_with_missing_reasoning_placeholder(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "The answer is: {answer}",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={"question": "What is 2 + 2?", "answer": "Because 2 and 2 make 4.\n ## The answer is: 4"},
        expected_user="This is a question: What is 2 + 2?",
        expected_assistant="Because 2 and 2 make 4.\n ## The answer is: 4",
    )

  def test_math_qa_formatting_with_missing_answer_placeholder(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "The answer is: {reply}",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={"question": "What is 2 + 2?", "answer": "Because 2 and 2 make 4.\n ## The answer is: 4"},
        expected_user="This is a question: What is 2 + 2?",
        expected_assistant="Because 2 and 2 make 4.\n ## The answer is: 4",
    )

  def test_math_qa_formatting_with_missing_question_placeholder(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {user_question}",
            "COMPLETION_TEMPLATE": "The answer is: {reply}",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={
            "question": "What is 2 + 2?",
            "answer": "The answer is: Because 2 and 2 make 4.\n ## The answer is: 4",
        },
        expected_user="What is 2 + 2?",
        expected_assistant="The answer is: Because 2 and 2 make 4.\n ## The answer is: 4",
    )

  def test_math_qa_formatting_with_no_templates(self):
    self._run_math_qa_test(
        template_config=None,
        example_input={"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
        expected_user="What is the capital of Germany?",
        expected_assistant="The capital of Germany is Berlin.",
    )

  def test_load_chat_template_from_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Test .jinja file
      jinja_path = os.path.join(tmpdir, "test.jinja")
      with open(jinja_path, "w", encoding="utf-8") as f:
        f.write("test jinja template")
      self.assertEqual(
          instruction_data_processing.load_chat_template_from_file(jinja_path),
          "test jinja template",
      )

      # Test .json file with chat_template
      json_path = os.path.join(tmpdir, "test.json")
      with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"chat_template": "test json template"}, f)
      self.assertEqual(
          instruction_data_processing.load_chat_template_from_file(json_path),
          "test json template",
      )

      # Test .json file without chat_template
      json_no_key_path = os.path.join(tmpdir, "no_key.json")
      with open(json_no_key_path, "w", encoding="utf-8") as f:
        json.dump({"other_key": "other_value"}, f)
      self.assertIsNone(instruction_data_processing.load_chat_template_from_file(json_no_key_path))

      # Test non-existent file
      self.assertIsNone(instruction_data_processing.load_chat_template_from_file("non_existent.jinja"))

  def test_load_chat_template_from_hub_with_revision_token_and_sha(self):
    template_bytes = b"{% generation %}answer{% endgeneration %}"
    expected_sha256 = hashlib.sha256(template_bytes).hexdigest()
    with tempfile.TemporaryDirectory() as tmpdir:
      downloaded_path = os.path.join(tmpdir, "chat_template.jinja")
      with open(downloaded_path, "wb") as template_file:
        template_file.write(template_bytes)

      with patch.object(instruction_data_processing, "hf_hub_download", return_value=downloaded_path) as mock_download:
        template = instruction_data_processing.load_chat_template_from_file(
            "hf://example-org/example-model/templates/train.jinja",
            hf_access_token="test-token",
            revision="0123456789abcdef0123456789abcdef01234567",
            expected_sha256=expected_sha256,
        )

    self.assertEqual(template, template_bytes.decode("utf-8"))
    mock_download.assert_called_once_with(
        repo_id="example-org/example-model",
        filename="templates/train.jinja",
        revision="0123456789abcdef0123456789abcdef01234567",
        token="test-token",
    )

  def test_local_and_hub_jinja2_templates_use_original_suffix_and_verify_sha(self):
    template_bytes = b"{% generation %}answer{% endgeneration %}"
    digest = hashlib.sha256(template_bytes).hexdigest()
    with tempfile.TemporaryDirectory() as tmpdir:
      # Hub downloads can resolve to an extensionless blob path.
      for hub in (False, True):
        with self.subTest(hub=hub):
          downloaded_path = os.path.join(tmpdir, "blob" if hub else "template.jinja2")
          with open(downloaded_path, "wb") as template_file:
            template_file.write(template_bytes)
          path = "hf://example-org/example-model/template.jinja2" if hub else downloaded_path
          with patch.object(instruction_data_processing, "hf_hub_download", return_value=downloaded_path):
            self.assertEqual(
                instruction_data_processing.load_chat_template_from_file(path, expected_sha256=digest),
                template_bytes.decode("utf-8"),
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
              instruction_data_processing.load_chat_template_from_file(path, expected_sha256="0" * 64)

  def test_existing_template_with_unsupported_suffix_has_specific_error(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      for hub in (False, True):
        with self.subTest(hub=hub):
          downloaded_path = os.path.join(tmpdir, "blob" if hub else "template.unsupported")
          with open(downloaded_path, "w", encoding="utf-8") as template_file:
            template_file.write("template")
          path = "hf://example-org/example-model/template.unsupported" if hub else downloaded_path
          with patch.object(instruction_data_processing, "hf_hub_download", return_value=downloaded_path):
            with self.assertRaisesRegex(ValueError, "Unsupported chat template file extension '.unsupported'"):
              instruction_data_processing.load_chat_template_from_file(path)
      self.assertIsNone(
          instruction_data_processing.load_chat_template_from_file(os.path.join(tmpdir, "missing.unsupported"))
      )

  def test_load_chat_template_normalizes_empty_hub_options_to_none(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      downloaded_path = os.path.join(tmpdir, "chat_template.jinja")
      with open(downloaded_path, "w", encoding="utf-8") as template_file:
        template_file.write("template")

      with patch.object(instruction_data_processing, "hf_hub_download", return_value=downloaded_path) as mock_download:
        self.assertEqual(
            instruction_data_processing.load_chat_template_from_file(
                "hf://example-org/example-model/chat_template.jinja",
                hf_access_token="",
                revision="",
            ),
            "template",
        )

    self.assertIsNone(mock_download.call_args.kwargs["revision"])
    self.assertIsNone(mock_download.call_args.kwargs["token"])

  def test_load_chat_template_rejects_sha_mismatch(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      downloaded_path = os.path.join(tmpdir, "chat_template.jinja")
      with open(downloaded_path, "w", encoding="utf-8") as template_file:
        template_file.write("template")

      with patch.object(instruction_data_processing, "hf_hub_download", return_value=downloaded_path):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
          instruction_data_processing.load_chat_template_from_file(
              "hf://example-org/example-model/chat_template.jinja",
              revision="0123456789abcdef0123456789abcdef01234567",
              expected_sha256="0" * 64,
          )

  def test_load_local_chat_template_checks_sha(self):
    template_bytes = b"local template"
    with tempfile.TemporaryDirectory() as tmpdir:
      template_path = os.path.join(tmpdir, "chat_template.jinja")
      with open(template_path, "wb") as template_file:
        template_file.write(template_bytes)

      self.assertEqual(
          instruction_data_processing.load_chat_template_from_file(
              template_path,
              expected_sha256=hashlib.sha256(template_bytes).hexdigest(),
          ),
          "local template",
      )

  def test_load_chat_template_rejects_revision_inside_uri(self):
    with patch.object(instruction_data_processing, "hf_hub_download") as mock_download:
      with self.assertRaisesRegex(ValueError, "set chat_template_revision separately"):
        instruction_data_processing.load_chat_template_from_file(
            "hf://example-org/example-model@feature/template/chat_template.jinja"
        )
    mock_download.assert_not_called()

  def test_load_chat_template_rejects_percent_encoded_path(self):
    with patch.object(instruction_data_processing, "hf_hub_download") as mock_download:
      with self.assertRaisesRegex(ValueError, "Percent-encoded"):
        instruction_data_processing.load_chat_template_from_file(
            "hf://example-org/example-model/templates%2Fchat_template.jinja"
        )
    mock_download.assert_not_called()

  def test_load_chat_template_rejects_missing_hub_path(self):
    with patch.object(instruction_data_processing, "hf_hub_download") as mock_download:
      with self.assertRaisesRegex(ValueError, "include a file path"):
        instruction_data_processing.load_chat_template_from_file("hf://example-org/example-model")
    mock_download.assert_not_called()

  def test_load_chat_template_rejects_missing_hub_repository(self):
    with patch.object(instruction_data_processing, "hf_hub_download") as mock_download:
      with self.assertRaises(ValueError):
        instruction_data_processing.load_chat_template_from_file("hf://example-org")
    mock_download.assert_not_called()

  def test_load_chat_template_rejects_non_model_hub_uri(self):
    with patch.object(instruction_data_processing, "hf_hub_download") as mock_download:
      with self.assertRaisesRegex(ValueError, "model repository"):
        instruction_data_processing.load_chat_template_from_file(
            "hf://datasets/example-org/example-dataset/chat_template.jinja"
        )
    mock_download.assert_not_called()

  @unittest.skipUnless(os.environ.get("MAXTEXT_RUN_NETWORK_TESTS") == "1", "network integration is opt-in")
  def test_load_public_pinned_hub_chat_template(self):
    template = instruction_data_processing.load_chat_template_from_file(
        "hf://HuggingFaceTB/SmolLM3-3B/chat_template.jinja",
        revision="a07cc9a04f16550a088caea529712d1d335b0ac1",
    )

    self.assertIn("{% generation", template)


class TestCustomDataFormatting(unittest.TestCase):
  """Test custom data formatting."""

  def setUp(self):
    super().setUp()
    self.columns = ["question", "answer"]
    self.dataset = datasets.Dataset.from_dict({col: [f"val_{col}_{i}" for i in range(3)] for col in self.columns})
    self.dataset_features = datasets.Features(
        {"messages": [{"content": datasets.Value("string"), "role": datasets.Value("string")}]}
    )

  def test_data_formatter_without_formatting_func_path(self):
    returned_dataset, returned_columns = instruction_data_processing.convert_to_conversational_format(
        self.dataset,
        self.columns,
    )

    expected_dataset = datasets.Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": f"val_question_{i}"}, {"role": "assistant", "content": f"val_answer_{i}"}]
                for i in range(3)
            ]
        },
        features=self.dataset_features,
    )

    self.assertEqual(returned_columns, ["messages"])
    assert list(returned_dataset) == list(expected_dataset)
    assert returned_dataset["messages"] == expected_dataset["messages"]

  def test_data_formatter_with_formatting_func_path_and_kwargs(self):
    expected_dataset = datasets.Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] for _ in range(3)]},
        features=self.dataset_features,
    )
    mock_formatter = MagicMock(return_value=expected_dataset)

    with patch.object(instruction_data_processing, "load_formatter", return_value=mock_formatter) as mock_load:
      returned_dataset, returned_columns = instruction_data_processing.convert_to_conversational_format(
          self.dataset,
          self.columns,
          formatting_func_path="some.module.my_func",
          formatting_func_kwargs={"template_path": "/tmp/tmpl.json"},
      )

    args, kwargs = mock_load.call_args
    self.assertEqual(args[0], "some.module.my_func")
    self.assertEqual(kwargs["remove_columns"], self.columns)
    self.assertTrue("template_config" in kwargs)
    mock_formatter.assert_called_once_with(self.dataset, self.dataset_features)
    self.assertIs(returned_dataset, expected_dataset)
    self.assertEqual(returned_columns, ["messages"])

  def test_data_formatter_with_formatting_func_path_without_kwargs(self):
    expected_dataset = datasets.Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] for _ in range(3)]},
        features=self.dataset_features,
    )
    mock_formatter = MagicMock(return_value=expected_dataset)

    with patch.object(instruction_data_processing, "load_formatter", return_value=mock_formatter) as mock_load:
      returned_dataset, returned_columns = instruction_data_processing.convert_to_conversational_format(
          self.dataset,
          self.columns,
          formatting_func_path="some.module.my_func",
          formatting_func_kwargs={},
      )

    args, kwargs = mock_load.call_args
    self.assertEqual(args[0], "some.module.my_func")
    self.assertEqual(kwargs["remove_columns"], self.columns)
    mock_formatter.assert_called_once_with(self.dataset, self.dataset_features)
    self.assertIs(returned_dataset, expected_dataset)
    self.assertEqual(returned_columns, ["messages"])


class TestDataProcessingUtils(unittest.TestCase):
  """Unit tests for dataset column validation (Scenario B)."""

  def test_validate_sft_columns_valid(self):
    """Verifies that valid SFT columns do not raise any error."""
    # These should pass without raising any exception
    data_processing_utils.validate_and_configure_sft_columns(["messages"], None)
    data_processing_utils.validate_and_configure_sft_columns(["prompt", "completion"], None)
    data_processing_utils.validate_and_configure_sft_columns(["question", "answer"], None)

  def test_validate_sft_columns_invalid_raises_helpful_error(self):
    """Verifies that invalid SFT columns raise a helpful AssertionError."""
    with self.assertRaises(AssertionError) as ctx:
      data_processing_utils.validate_and_configure_sft_columns(["some_invalid_column"], None)

    # Verify that the error message is helpful and contains the expected guidance
    self.assertIn("Dataset column names mismatch", str(ctx.exception))
    self.assertIn("Expected columns to match one of", str(ctx.exception))
    self.assertIn("prompt", str(ctx.exception))
    self.assertIn("completion", str(ctx.exception))
    self.assertIn("messages", str(ctx.exception))
    self.assertIn("question", str(ctx.exception))
    self.assertIn("answer", str(ctx.exception))
    self.assertIn("some_invalid_column", str(ctx.exception))


if __name__ == "__main__":
  unittest.main()
