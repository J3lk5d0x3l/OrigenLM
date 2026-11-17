import unittest
import torch
from origen.model import ModelConfig, OrigenLM
from origen.tokenizer import encode, decode, training_example, conversation_prefix, VOCAB_SIZE


class OrigenTests(unittest.TestCase):
    def test_utf8_and_chat_tokens(self):
        self.assertEqual(decode(encode("¡Hola, mundo! 🐍")), "¡Hola, mundo! 🐍")
        self.assertTrue(len(training_example({"user": "Hola", "assistant": "¡Buenas!"})) > 20)
        self.assertEqual(conversation_prefix("sistema", [], "¿qué tal?")[-1], 260)

    def test_malformed_terminal_input_does_not_crash(self):
        # An invalid UTF-8 byte read with surrogateescape can become U+DCC2.
        # Accented characters and emoji must remain intact.
        broken = "Hola, ¿en qué puedo ayudarte? 🐍\udcc2"
        self.assertEqual(decode(encode(broken)), "Hola, ¿en qué puedo ayudarte? 🐍?")
        self.assertEqual(decode(encode("Español café 🐍")), "Español café 🐍")
        self.assertEqual(decode(encode("a\ud800b")), "a?b")
        self.assertEqual(conversation_prefix("sistema", [("Hola", "¡Buenas!")], broken)[-1], 260)

    def test_forward_and_backprop(self):
        model = OrigenLM(ModelConfig(dim=32, n_layers=2, n_heads=4, hidden_dim=64, max_seq_len=16))
        tokens = torch.randint(0, VOCAB_SIZE, (2, 12))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, VOCAB_SIZE))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.tokens.weight.grad)

    def test_generation(self):
        torch.manual_seed(7)
        model = OrigenLM(ModelConfig(dim=32, n_layers=1, n_heads=4, hidden_dim=64, max_seq_len=16))
        model.eval()
        output = model.generate([256, 259, 72, 260], max_new_tokens=5, top_k=10)
        self.assertLessEqual(len(output), 5)


if __name__ == "__main__":
    unittest.main()
