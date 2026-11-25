"""CPU annotation ranges for attributing trace kernels without timing the clean pass."""


class TraceRanges:
    def __init__(self):
        self.restores = []

    def wrap(self, obj, attr, name):
        from torch.profiler import record_function
        original = getattr(obj, attr)
        had_local, local = attr in vars(obj), vars(obj).get(attr)

        def annotated(*args, **kwargs):
            with record_function(name):
                return original(*args, **kwargs)

        setattr(obj, attr, annotated)
        self.restores.append((obj, attr, had_local, local))

    def install(self, model):
        for name in model.mot.expert_order:
            expert = model.mot.experts[name]
            for attr in ('pre_attn', 'post_attn', 'pre_dit', 'post_dit'):
                if hasattr(expert, attr):
                    self.wrap(expert, attr, f'expert.{name}.{attr}')
            if name == 'action':
                for i, block in enumerate(expert.blocks):
                    for attr in ('cross_attn', 'ffn'):
                        if hasattr(block, attr):
                            self.wrap(getattr(block, attr), 'forward',
                                      f'expert.action.block_{i}.{attr}')
                    self.wrap(block.self_attn.o, 'forward', f'expert.action.block_{i}.out_projection')
        self.wrap(model.mot, '_mixed_attention', 'expert.shared.attention')

    def restore(self):
        for obj, attr, had_local, local in reversed(self.restores):
            if had_local:
                setattr(obj, attr, local)
            else:
                delattr(obj, attr)
