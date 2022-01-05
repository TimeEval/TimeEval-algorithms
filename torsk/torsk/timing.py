from time import time
from torsk.numpy_accelerate import bh_flush


class Timer(object):

    def __init__(self, timing_depth=3, flush=True, root_context="root"):
        self.depth = 0
        self.times = {}
        self.time_stack = []
        self.context_stack = [f"/{root_context}"]
        self.flush = flush
        self.timing_depth = timing_depth

    def begin(self, ctx):
        self.depth += 1
        if self.depth <= self.timing_depth:
            self.context_stack.append(ctx)  # append = push
            if self.flush:
                bh_flush()
            self.time_stack.append(time())

    def end(self):
        self.depth -= 1
        if self.depth < self.timing_depth:
            if self.flush:
                bh_flush()
            path = '/'.join(self.context_stack)
            t = time() - self.time_stack.pop()

            if path in self.times.keys():
                self.times[path] += t
            else:
                self.times[path] = t

            self.context_stack.pop()

    def minus(self, times0):
        time_difference = {}
        for k in self.times.keys():
            if k in times0.keys():
                time_difference[k] = self.times[k] - times0[k]
            else:
                time_difference[k] = self.times[k]

        return time_difference

    def reset(self):
        self.times = {}

    def pretty_print(self):
        s = "Accumulated timing:\n"
        keys_r = list(self.times.keys())
        keys_r.reverse()
        for k in keys_r:
            prefix = k.rfind("/") + 1
            pad = ' ' * 4 * (k.count('/') - 1)
            pretty_k = k[prefix:]
            s += f"{pad}{self.times[k]:9.3f} {pretty_k}\n"
        return s


def start_timer(timer, context):
    if timer is not None:
        timer.begin(context)


def end_timer(timer):
    if timer is not None:
        timer.end()
