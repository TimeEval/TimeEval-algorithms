import numpy as np
from typing import Iterator, Tuple
from sksequitur import parse, Grammar
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
from joblib import Parallel, delayed


class EnsembleGI:
    def __init__(self, anomaly_window_size: int, n_estimators: int, max_paa_transform_size: int, max_alphabet_size: int, selectivity: float, random_state: int, n_jobs: int):
        self.window_size = anomaly_window_size
        self.ensemble_size = n_estimators
        self.w_max = max_paa_transform_size
        self.a_max = max_alphabet_size
        self.selectivity = selectivity
        self.n_jobs = n_jobs

        np.random.seed(random_state)
        import random
        random.seed(random_state)

    def _random_params(self) -> Iterator[Tuple[int, int]]:
        n_combinations = (self.w_max - 2) * (self.a_max - 2)
        comb_idx = np.random.choice(n_combinations, size=self.ensemble_size, replace=False)
        w = (comb_idx // (self.w_max - 2)) + 2
        a = (comb_idx % (self.a_max - 2)) + 2
        return zip(w, a)

    def _density_curve(self, grammar: Grammar) -> np.ndarray:
        rules = grammar[0]
        density_curve = []
        i = 0
        depth = 0
        depths = []
        while i < len(rules):
            value_at_i = rules[i]
            rule = grammar.get(value_at_i, value_at_i)
            if type(rule) == str:
                i += 1
                density_curve.append(depth)
                if len(depths) > 0:
                    depths[-1] -= 1
                    while len(depths) > 0 and depths[-1] < 1:
                        del depths[-1]
                        depth -= 1
                        if len(depths) > 0:
                            depths[-1] -= 1
            else:
                rules[i] = rule
                depth += 1
                depths.append(len(rule))
            rules = flatten(rules)
        return np.array(density_curve)

    def sax(self, X: np.ndarray, w: int, a: int, method: str = "orig") -> str:
        saxed = []
        if method == "sliding":
            iter = range(len(X) - self.window_size + 1)
        elif method == "tumbling":
            iter = range(0, len(X), self.window_size)
        else:  # method == "orig"
            iter = range(len(X) // self.window_size)

        for i in iter:
            _paa = paa(znorm(X[i:i+self.window_size]), w)
            saxed.append(_paa)

        if method == "orig":
            rest = len(X) % self.window_size
            rest_segments = int(w * (rest / self.window_size)) or 1
            if rest > 0:
                _paa = paa(znorm(X[-rest:]), rest_segments)
                saxed.append(_paa)

        saxed = np.concatenate(saxed)
        return ts_to_string(saxed, cuts_for_asize(a))

    def _stretch_density_curve(self, density_curve: np.ndarray, w: int, l: int):
        result = np.interp(np.arange(l), np.linspace(0, l, len(density_curve)), density_curve)
        return result

    def _grammar_induction(self, X: np.ndarray, w: int, a: int, window_method: str = "orig") -> np.ndarray:
        saxed = self.sax(X, w, a, method=window_method)
        parsed = parse(saxed)
        density_curve = self._density_curve(parsed)
        density_curve = self._stretch_density_curve(density_curve, w, len(X))
        return density_curve

    def detect(self, X: np.ndarray, window_method: str = "orig"):
        def discretize(i, w, a):
            print(f"\nModel {i} with w={w} and a={a}")
            return self._grammar_induction(X, w, a, window_method)

        density_curves = Parallel(n_jobs=self.n_jobs)(
            delayed(discretize)(i, w, a)
            for i, (w, a) in enumerate(self._random_params())
        )

        density_curves = np.stack(density_curves)
        indices = np.argsort(density_curves.std(axis=1))[::-1]
        indices = indices[:int(self.ensemble_size*self.selectivity)]
        density_curves = density_curves[indices]
        density_curves = density_curves / np.max(density_curves, axis=0)
        density_overall = 1 - np.median(density_curves, axis=0)
        return density_overall


def flatten(x: list) -> list:
    _x = []
    for dd in x:
        if type(dd) == list:
            for d in dd:
                _x.append(d)
        else:
            _x.append(dd)
    return _x
