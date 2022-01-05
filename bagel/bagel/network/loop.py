from collections import Iterable
import time
from contextlib import contextmanager
from typing import List, Dict, Union, Any

import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Loop(object):
    __EPOCH_TIME_KEY = "epoch_time(s)"
    __STEP_TIME_KEY = "step_time(s)"

    def __init__(self, max_epochs: int = 32, max_steps: int = None, disp_epoch_freq=1, print_fn=print, use_cuda=False):
        def assert_positive_integer(v, name, can_be_none=False):
            type_tuple = (int, np.integer, type(None)) if can_be_none else (int, np.integer)
            assert isinstance(v, type_tuple) and (v is None or v > 0), "{} should be positive integer: {}".format(name,
                                                                                                                  v)

        assert max_epochs is not None or max_steps is not None, "At least one of max_epochs and max_steps should not be None"
        assert_positive_integer(max_epochs, "max_epochs", True)
        assert_positive_integer(max_steps, "max_steps", True)
        assert_positive_integer(disp_epoch_freq, "disp_epoch_freq")
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._print_fn = print_fn
        self._disp_epoch_freq = disp_epoch_freq
        self._use_cuda = use_cuda

        self._epoch_cnt = 0  # type: int
        self._step_cnt = 0  # type: int
        self._displayed_at_epoch = 0  # type: int

        self._metrics = []  # type: List[Dict[str, Any]]
        self._data = []  # type: List[Dict[str, Any]]

        self._within_epochs = False
        self._within_steps = False

    @contextmanager
    def with_context(self):
        yield self

    def _eta(self, epoch_time_estimate, step_time_estimate):
        estimate = float("inf")
        if self._max_epochs is not None:
            estimate = min(estimate, (self._max_epochs - self._epoch_cnt) * epoch_time_estimate)
        if self._max_steps is not None:
            estimate = min(estimate, (self._max_steps - self._step_cnt) * step_time_estimate)
        return estimate

    def iter_epochs(self):
        def loop_condition():
            return (self._max_epochs is None or self._epoch_cnt < self._max_epochs) and (
                    self._max_steps is None or self._step_cnt < self._max_steps)

        def disp_condition():
            return self._epoch_cnt % self._disp_epoch_freq == 0

        self._within_epochs = True

        try:
            while loop_condition():
                self._epoch_cnt += 1
                self._metrics.append({})
                self._data.append({})
                tic = time.time()
                yield self._epoch_cnt
                toc = time.time()
                self.submit_metric(self.__EPOCH_TIME_KEY, toc - tic)
                if disp_condition():
                    self._print_log()
                    self._displayed_at_epoch = self._epoch_cnt
            if not disp_condition():
                self._print_log()
        finally:
            self._within_epochs = False

    def iter_steps(self, dataloader: DataLoader):
        assert self._within_epochs, "iter_steps must be called in an iter_epoch."
        self._within_steps = True
        try:
            for data in dataloader:
                self._step_cnt += 1
                tic = time.time()
                yield self._step_cnt, self.__make_variables(data)
                toc = time.time()
                self.submit_metric(self.__STEP_TIME_KEY, toc - tic)
                if self._max_steps is not None and self._step_cnt >= self._max_steps:
                    break
        finally:
            self._within_steps = False

    @property
    def metrics(self):
        return self._metrics

    @property
    def data(self):
        return self._data

    def get_metric_by_epoch(self, epoch: int):
        assert isinstance(epoch, (int, np.integer)) and epoch > 0, "{} should be positive integer: {}".format("epoch",
                                                                                                              epoch)
        return self._metrics[epoch - 1]

    def get_metric_by_name(self, name: str):
        metric_list = []
        epoch_list = []
        for idx, data in enumerate(self.metrics):
            if name in data:
                metric_list.append(data[name])
                epoch_list.append(idx + 1)
        return epoch_list, metric_list

    def get_data_by_epoch(self, epoch: int):
        assert isinstance(epoch, (int, np.integer)) and epoch > 0, "{} should be positive integer: {}".format("epoch",
                                                                                                              epoch)
        return self._data[epoch - 1]

    def get_data_by_name(self, name: str):
        data_list = []
        epoch_list = []
        for idx, data in enumerate(self.data):
            if name in data:
                data_list.append(data[name])
                epoch_list.append(idx + 1)
        return epoch_list, data_list

    def submit_metric(self, name, value):
        if self._within_steps:
            d = self._metrics[-1]  # type: dict
            if name not in d:
                d[name] = []
            d[name].append(value)
        elif self._within_epochs:
            d = self._metrics[-1]  # type: dict
            d[name] = value
        else:
            raise RuntimeError("Can't submit metric outside epoch or step")

    def submit_data(self, name, value):
        if self._within_steps:
            d = self._data[-1]  # type: dict
            if name not in d:
                d[name] = []
            d[name].append(value)
        elif self._within_epochs:
            d = self._data[-1]  # type: dict
            d[name] = value
        else:
            raise RuntimeError("Can't submit data outside epoch or step")

    def _print_log(self):
        metrics_dict = {}
        for metrics in self._metrics[self._displayed_at_epoch:self._epoch_cnt]:
            for name, data in metrics.items():
                if name not in metrics_dict:
                    metrics_dict[name] = []
                if isinstance(data, Iterable):
                    metrics_dict[name].extend(list(data))
                else:
                    metrics_dict[name].append(data)
        metric_str_list = []
        estimate_epoch_time, estimate_step_time = float("inf"), float("inf")
        for name, data in metrics_dict.items():
            mean = np.mean(data)
            if name == self.__EPOCH_TIME_KEY:
                estimate_epoch_time = mean
            elif name == self.__STEP_TIME_KEY:
                estimate_step_time = mean
            if len(data) > 1:
                std = np.std(data)
                metric_str_list.append("{}: {:.6f}(±{:.6f})".format(name, mean, std))
            else:
                metric_str_list.append("{}: {:.6f}".format(name, mean))
        metric_str = " ".join(metric_str_list)

        if self._max_epochs is None:
            epoch_str = "{}".format(self._epoch_cnt)
        else:
            epoch_str = "{}/{}".format(self._epoch_cnt, self._max_epochs)
        if self._max_steps is None:
            step_str = "{}".format(self._step_cnt)
        else:
            step_str = "{}/{}".format(self._step_cnt, self._max_steps)
        process_str = "[epoch:{} step:{} ETA:{:.3f}s]".format(epoch_str, step_str,
                                                              self._eta(estimate_epoch_time, estimate_step_time))
        self._print_fn("{} {}".format(process_str, metric_str))

    def __make_variables(self, data):
        if isinstance(data, Iterable):
            ret = []
            for x in data:
                ret.append(self.__make_single_variable(x))
            return tuple(ret)
        else:
            return self.__make_single_variable(data)

    def __make_single_variable(self, data):
        ret = Variable(data)
        if self._use_cuda:
            ret = ret.cuda()
        return ret


TrainLoop = Loop


class TestLoop(Loop):
    def __init__(self, print_fn=print, use_cuda=False):
        super(TestLoop, self).__init__(max_epochs=1, max_steps=None, disp_epoch_freq=1, print_fn=print_fn, use_cuda=use_cuda)

    def iter_epochs(self):
        raise RuntimeError("TestLoop don't need to iterate epochs.")

    @contextmanager
    def with_context(self):
        for _ in super(TestLoop, self).iter_epochs():
            yield self

    def get_metric_by_name(self, name: str):
        return self.metrics[0][name]

    def get_data_by_name(self, name: str):
        return self.data[0][name]


def __test():
    with Loop(max_epochs=32) as loop:
        for epoch in loop.iter_epochs():
            for step, data in loop.iter_steps([1, 2, 3, 4]):
                loop.submit_metric("step_loss", 1. / step)
            for step, data in loop.iter_steps([2, 3, 4, 5]):
                loop.submit_metric("valid_loss", 2 / step)
    print("")
    with Loop(max_epochs=32, max_steps=10, disp_epoch_freq=3) as loop:
        for epoch in loop.iter_epochs():
            for step, data in loop.iter_steps([1, 2, 3, 4]):
                loop.submit_metric("step_loss", 1. / step)
    print("")
    with Loop(max_epochs=32, disp_epoch_freq=4) as loop:
        for epoch in loop.iter_epochs():
            for step, data in loop.iter_steps([1, 2, 3, 4]):
                loop.submit_metric("step_loss", 1. / step)


if __name__ == '__main__':
    __test()
