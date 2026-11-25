"""Opt-in, inference-only GenericMoT scheduling experiment on one CUDA device.

One stream per expert for independent pre/post work; joint attention stays on the
caller stream. No mathematical changes. Not a training or torch.compile backend.
"""
import types


def install(mot):
    import torch
    if mot.training:
        raise ValueError('Multi-stream experiment is inference-only')
    params = list(mot.parameters())
    devices = {p.device for p in params}
    if len(devices) != 1 or next(iter(devices)).type != 'cuda':
        raise ValueError('Expected experts on exactly one CUDA device')
    device = next(iter(devices))
    streams = {name: torch.cuda.Stream(device=device) for name in mot.expert_order}
    had_local, local = 'forward' in vars(mot), vars(mot).get('forward')

    def record(value, stream):
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value.record_stream(stream)
        elif isinstance(value, dict):
            for item in value.values():
                record(item, stream)
        elif isinstance(value, (tuple, list)):
            for item in value:
                record(item, stream)

    def forward(self, tokens, aux, attention_mask):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Use multi-stream only under no_grad/inference_mode and eval')
        if any(name not in tokens for name in self.expert_order):
            raise ValueError('Missing expert tokens')
        caller = torch.cuda.current_stream(device)
        tokens = dict(tokens)
        for layer in range(self.num_layers):
            prepared = {}
            for name in self.expert_order:
                stream = streams[name]
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    record((tokens[name], aux[name]), stream)
                    expert = self.experts[name]
                    prepared[name] = expert.pre_attn(
                        expert.blocks[layer], tokens[name], aux[name], layer)
            for stream in streams.values():
                caller.wait_stream(stream)
            # Allocator ownership follows tensors crossing stream boundaries.
            record(prepared, caller)
            q = torch.cat([prepared[n][0] for n in self.expert_order], dim=1)
            k = torch.cat([prepared[n][1] for n in self.expert_order], dim=1)
            v = torch.cat([prepared[n][2] for n in self.expert_order], dim=1)
            if attention_mask.shape[-1] != q.shape[1]:
                raise ValueError('Joint attention mask/token length mismatch')
            mixed = self._mixed_attention(q, k, v, attention_mask)
            offset = 0
            for name in self.expert_order:
                stream = streams[name]
                stream.wait_stream(caller)
                size = prepared[name][0].shape[1]
                with torch.cuda.stream(stream):
                    record((mixed, prepared[name][3], aux[name]), stream)
                    expert = self.experts[name]
                    tokens[name] = expert.post_attn(
                        expert.blocks[layer], prepared[name][3],
                        mixed[:, offset:offset + size, :], aux[name], layer)
                offset += size
            for stream in streams.values():
                caller.wait_stream(stream)
            record(tokens, caller)
        return tokens

    mot.forward = types.MethodType(forward, mot)

    def restore():
        torch.cuda.synchronize(device)
        if had_local:
            mot.forward = local
        else:
            delattr(mot, 'forward')

    return restore
