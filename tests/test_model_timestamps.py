"""Exercise production helpers without downloading checkpoint dependencies."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch
from utils.ctc_alignment import ctc_forced_align


SOURCE = Path(__file__).resolve().parents[1] / "model.py"


def load_class(name):
    node = next(
        node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def load_model_methods():
    node = next(
        node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SenseVoiceSmall"
    )
    methods = [item for item in node.body if isinstance(item, ast.FunctionDef)
               and item.name in ("inference", "post")]
    namespace = {
        "torch": torch,
        "ctc_forced_align": ctc_forced_align,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class PositionEncoderTest(unittest.TestCase):
    def test_module_supports_checkpoint_state_and_device_transfer(self):
        encoder = load_class("SinusoidalPositionEncoder")(d_model=8)
        self.assertEqual(dict(encoder.state_dict()), {})
        encoder.load_state_dict({})
        encoder.to("cpu")

    def test_call_matches_forward_without_resizing_the_input(self):
        encoder = load_class("SinusoidalPositionEncoder")(d_model=8)
        speech = torch.zeros(2, 5, 8)
        output = encoder(speech)
        self.assertEqual(output.shape, speech.shape)
        self.assertTrue(torch.equal(output, encoder.forward(speech)))
        self.assertTrue(torch.equal(speech, torch.zeros_like(speech)))


class TimestampContractTest(unittest.TestCase):
    def test_multiple_ids_keep_their_original_token_time_range(self):
        methods = load_model_methods()
        logits = torch.full((1, 9, 8), -10.0)
        for frame, token in enumerate([1, 2, 3, 4, 5, 6, 0, 7, 0]):
            logits[0, frame, token] = 10
        model = SimpleNamespace(
            lid_dict={"zh": 0}, textnorm_dict={"woitn": 0},
            embed=lambda ids: torch.zeros(*ids.shape, 8),
            encoder=lambda speech, lengths: (torch.zeros(1, 9, 8), torch.tensor([9])),
            ctc=SimpleNamespace(log_softmax=lambda x: logits, softmax=lambda x: logits.softmax(-1)),
            blank_id=0, ignore_id=-1,
        )
        model.post = lambda timestamps: methods["post"](model, timestamps)
        tokenizer = SimpleNamespace(
            decode=lambda ids: "A B",
            text2tokens=lambda text: ["zh", "neutral", "speech", "woitn", "▁A", "▁B"],
            tokens2ids=lambda tokens: [[5, 6], [7]],
        )
        results, _ = methods["inference"](
            model, torch.zeros(1, 5, 8), torch.tensor([5]), key=["sample"],
            tokenizer=tokenizer, data_type="fbank", device="cpu",
            language="zh", output_timestamp=True,
        )
        self.assertEqual(results[0]["timestamp"], [[0, 90], [150, 210]])
        self.assertEqual(results[0]["words"], ["A", "B"])

        tokenizer.decode = lambda ids: ""
        tokenizer.text2tokens = lambda text: ["zh", "neutral", "speech", "woitn"]
        tokenizer.tokens2ids = lambda tokens: []
        results, _ = methods["inference"](
            model, torch.zeros(1, 5, 8), torch.tensor([5]), key=["empty"],
            tokenizer=tokenizer, data_type="fbank", device="cpu",
            language="zh", output_timestamp=True,
        )
        self.assertEqual(results, [{"key": "empty", "text": "", "timestamp": [], "words": []}])

    def test_batch_alignment_uses_each_items_length(self):
        methods = load_model_methods()
        logits = torch.full((2, 12, 6), -10.0)
        for item in range(2):
            for frame, token in enumerate([1, 2, 3, 4, 5, 0, 5, 0, 5, 0, 5, 0]):
                logits[item, frame, token] = 10
        model = SimpleNamespace(
            lid_dict={"zh": 0}, textnorm_dict={"woitn": 0},
            embed=lambda ids: torch.zeros(*ids.shape, 8),
            encoder=lambda speech, lengths: (torch.zeros(2, 12, 8), torch.tensor([6, 12])),
            ctc=SimpleNamespace(log_softmax=lambda x: logits, softmax=lambda x: logits.softmax(-1)),
            blank_id=0, ignore_id=-1,
        )
        model.post = lambda timestamps: methods["post"](model, timestamps)
        tokenizer = SimpleNamespace(
            decode=lambda ids: "你" * (len(ids) - 4),
            text2tokens=lambda text: ["zh", "neutral", "speech", "woitn"] + list(text),
            tokens2ids=lambda tokens: [[5] for token in tokens],
        )
        results, _ = methods["inference"](
            model, torch.zeros(2, 8, 8), torch.tensor([2, 8]), key=["short", "long"],
            tokenizer=tokenizer, data_type="fbank", device="cpu",
            language="zh", output_timestamp=True,
        )
        self.assertEqual([r["key"] for r in results], ["short", "long"])
        self.assertEqual(results[0]["timestamp"], [[0, 30]])
        self.assertEqual(results[1]["timestamp"], [[0, 30], [90, 150], [210, 270], [330, 390]])
        self.assertEqual(results[1]["words"], ["你"] * 4)

    def test_inference_returns_word_aligned_millisecond_pairs(self):
        methods = load_model_methods()
        logits = torch.full((1, 8, 6), -10.0)
        for frame, token in enumerate([1, 2, 3, 4, 5, 0, 5, 0]):
            logits[0, frame, token] = 10
        model = SimpleNamespace(
            lid_dict={"zh": 0}, textnorm_dict={"woitn": 0},
            embed=lambda ids: torch.zeros(*ids.shape, 8),
            encoder=lambda speech, lengths: (torch.zeros(1, 8, 8), torch.tensor([8])),
            ctc=SimpleNamespace(log_softmax=lambda x: logits, softmax=lambda x: logits.softmax(-1)),
            blank_id=0, ignore_id=-1,
        )
        if "post" in methods:
            model.post = lambda timestamps: methods["post"](model, timestamps)
        tokenizer = SimpleNamespace(
            decode=lambda ids: "你好",
            text2tokens=lambda text: ["zh", "neutral", "speech", "woitn", "你", "好"],
            tokens2ids=lambda tokens: [[5] for token in tokens],
        )
        results, _ = methods["inference"](
            model, torch.zeros(1, 4, 8), torch.tensor([4]), key=["sample"],
            tokenizer=tokenizer, data_type="fbank", device="cpu",
            language="zh", output_timestamp=True,
        )
        self.assertEqual(results[0]["timestamp"], [[0, 30], [90, 150]])
        self.assertEqual(results[0]["words"], ["你", "好"])

    def test_word_boundaries_empty_input_and_nonmutation(self):
        methods = load_model_methods()
        self.assertIn("post", methods, "The timestamp normalization must be part of the model")
        convert = lambda values: methods["post"](None, values)
        self.assertEqual(convert([]), ([], []))
        tokens = [
            ["▁Hel", 0.0, 0.06], ["lo", 0.06, 0.12],
            ["▁", 0.12, 0.15], ["world", 0.15, 0.24],
            ["你", 0.24, 0.30], ["好", 0.30, 0.36], ["!", 0.36, 0.42],
        ]
        original = [token[:] for token in tokens]
        self.assertEqual(
            convert(tokens),
            ([[0, 120], [150, 240], [240, 300], [300, 360], [360, 420]],
             ["Hello", "world", "你", "好", "!"]),
        )
        self.assertEqual(tokens, original)


if __name__ == "__main__":
    unittest.main()
