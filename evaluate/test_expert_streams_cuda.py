"""Run on the GPU server: python -m unittest discover -s evaluate -p test_expert_streams_cuda.py"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA PyTorch required')
class StreamTests(unittest.TestCase):
    def test_same_math_on_nondefault_caller_stream_and_reused_allocator(self):
        from expert_streams import install

        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])

            def pre_attn(self, block, x, aux, layer):
                z = block(x)
                return z, z, z, x

            def post_attn(self, block, state, mixed, aux, layer):
                return state + torch.tanh(mixed)

        class Mot(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_order = ['video', 'code', 'action']
                self.experts = torch.nn.ModuleDict({n: Expert() for n in self.expert_order})
                self.num_layers = 3

            def _mixed_attention(self, q, k, v, mask):
                return torch.softmax((q @ k.transpose(-1, -2)) / 8**.5, -1) @ v

            def forward(self, tokens, aux, mask):
                tokens = dict(tokens)
                for layer in range(self.num_layers):
                    values = {n: self.experts[n].pre_attn(
                        self.experts[n].blocks[layer], tokens[n], aux[n], layer)
                        for n in self.expert_order}
                    q, k, v = [torch.cat([values[n][i] for n in self.expert_order], 1)
                               for i in range(3)]
                    mixed = self._mixed_attention(q, k, v, mask)
                    offset = 0
                    for n in self.expert_order:
                        size = values[n][0].shape[1]
                        tokens[n] = self.experts[n].post_attn(
                            self.experts[n].blocks[layer], values[n][3],
                            mixed[:, offset:offset+size], aux[n], layer)
                        offset += size
                return tokens

        torch.manual_seed(7)
        model = Mot().cuda().eval()
        aux = {n: {} for n in model.expert_order}
        caller = torch.cuda.Stream()
        for size in (3, 7, 3):
            inputs = {n: torch.randn(1, size+i, 8, device='cuda')
                      for i, n in enumerate(model.expert_order)}
            mask = torch.ones(sum(x.shape[1] for x in inputs.values()), device='cuda', dtype=torch.bool)
            caller.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(caller):
                expected = model(inputs, aux, mask)
                restore = install(model)
                try:
                    for _ in range(5):
                        actual = model(inputs, aux, mask)
                        for n in model.expert_order:
                            torch.testing.assert_close(actual[n], expected[n])
                finally:
                    restore()


if __name__ == '__main__':
    unittest.main()
