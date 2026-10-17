"""CUDA events outside the complete DiT __call__, including FSDP/SP hooks."""


class TimedDiT:
    def __init__(self, inner, torch, forwards):
        self.inner = inner
        self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(forwards)]
        self.count = 0
        # First record allocates events: do this OUTSIDE every request timer.
        for begin, end in self.events:
            begin.record()
            end.record()
        torch.cuda.synchronize()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        begin, end = self.events[self.count]
        begin.record()  # DiT START, before inner module's pre-hooks.
        output = self.inner(*args, **kwargs)
        end.record()    # DiT END, after inner module's post-hooks. No synchronization.
        self.count += 1
        return output

    def elapsed_ms(self):
        # Only read after E2E ends and the caller synchronizes.
        if self.count != len(self.events):
            raise ValueError(f'Expected {len(self.events)} forwards, got {self.count}')
        return [begin.elapsed_time(end) for begin, end in self.events]
