from . import pcw_regrs_py as _rs

from dataclasses import dataclass
from enum import Enum, auto
import scipy.optimize as opt
from typing import Callable, Iterable, List, NamedTuple, Optional
import numpy as np
import numpy.typing as npt
from numbers import Integral, Real
from bisect import bisect
import pycw_fn


def continuity_opt_jumps(polys: List[np.polynomial.Polynomial], jump_idxs, sample_times) -> np.ndarray:
    """Calculate the optimal jump positions to make the resulting model 'as continuous as possible'."""
    boundaries = np.vstack((sample_times[:-1], sample_times[1:])).T
    jumps = []
    for left_poly, right_poly, jump_idx in zip(polys[:-1], polys[1:], jump_idxs):
        interval = boundaries[jump_idx]
        if left_poly.degree() == 0 and right_poly.degree() == 0:
            jumps.append(np.mean(interval))
        else:
            # compute root of `|left_poly(x) - right_poly(x)|²` over the `interval` containing the jump
            p = (left_poly - right_poly)**2
            jumps.append(opt.minimize_scalar(
                p, bounds=interval, method="bounded").x)
    return jumps


def model_params_to_poly(ts: np.ndarray, ys: np.ndarray, model_params: _rs.PolyModelSpec, weights: Optional[np.ndarray] = None) -> np.polynomial.Polynomial:
    """Turn the basic model parameters for a segment of data into a numpy polynomial."""
    deg = model_params.degrees_of_freedom - 1
    seg_weights = None if weights is None else weights[
        model_params.start_idx: model_params.stop_idx + 1]
    if deg == 0:
        # np.mean(ys[model_params.start_idx: model_params.stop_idx + 1])
        weighted_mean = np.ma.average(
            ys[model_params.start_idx: model_params.stop_idx + 1], weights=seg_weights)
        return np.polynomial.Polynomial((weighted_mean, ), window=np.array([-1., 1.]), domain=np.array([-1., 1.]))
    else:
        return np.polynomial.Polynomial.fit(
            ts[model_params.start_idx: model_params.stop_idx + 1],
            ys[model_params.start_idx: model_params.stop_idx + 1],
            deg=deg, domain=np.array([-1., 1.]), w=seg_weights)


class TimeSeriesSample(NamedTuple):
    sample_times: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

    def __len__(self):
        return len(self.values)


class JumpLocator(Enum):
    MIDPOINT = auto()
    CONTINUITY_OPTIMIZED = auto()


@dataclass(frozen=True)
class PcwFn():
    """A piecewise function"""
    funcs: Iterable[Callable]
    jumps: Iterable[Real]  # must be sorted ascendingly

    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluate the function at some point (or an array of points)"""
        if isinstance(x, np.ndarray):
            xs = x.flatten()
            jumps = self.jumps
            funcs = self.funcs
            return np.array([funcs[bisect(jumps, x)](x) for x in xs]).reshape(x.shape)
        else:
            return self.funcs[bisect(self.jumps, x)](x)


class PcwConstFn(pycw_fn.PcwFn):
    values: np.ndarray
    jump_points: np.ndarray

    def __init__(self, values, jump_points): pass

    @staticmethod
    def _constant_func(x: float) -> np.polynomial.Polynomial:
        """Construct a constant function"""
        return lambda _: x

    @classmethod
    def _from_rs(Self, func: _rs.PcwConstFn):
        """Converts a piecewise function from its native representation into a callable python function"""
        return Self(func.values, func.jump_points)

    def __new__(Self, values, jump_points):
        native = pycw_fn.PcwFn.from_funcs_and_jumps(
            [PcwConstFn._constant_func(y) for y in values], jump_points)._native
        self = super().__new__(Self)
        super().__init__(self, native)
        self.values = values
        self.jump_points = jump_points
        return self


class Solution():
    _sol: _rs.Solution
    sample: TimeSeriesSample
    weights: Optional[np.ndarray]

    def __init__(self, _sol, sample, weights):
        self._sol = _sol
        self.sample = sample
        self.weights = weights

    @classmethod
    def fit(
            Self,
            sample: TimeSeriesSample,
            max_total_dof: Optional[Integral] = None,
            max_segment_degree: Optional[Integral] = 15,
            weights=None,) -> "Solution":
        """Construct a solution to piecewise regression problem on the given data"""
        return Self(
            _rs.fit_pcw_poly(
                sample.sample_times,
                sample.values,
                max_total_dof,
                max_segment_degree + 1 if max_segment_degree is not None else None,
                weights,
            ),
            sample, weights
        )

    def n_best(self,
               jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
               n_best: Integral = 1,
               ) -> "PcwPolynomial":
        """Get the n models with the n lowest CV scores."""
        models = self._sol.n_cv_minimizers(n_best)
        return [PcwPolynomial.from_data_and_model(self.sample, model, jump_locator, self.weights) for model in models]

    def ose(self, jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,) -> "PcwPolynomial":
        """Get the best model with respect to the one standard error rule."""
        model = self._sol.ose_best()
        return PcwPolynomial.from_data_and_model(self.sample, model, jump_locator, self.weights)

    def xse(
        self,
        xse_factor: Optional[np.float64] = 1.0,
        jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
    ) -> "PcwPolynomial":
        """Get the best model with respect to the se_factor standard error rule."""
        model = self._sol.xse_best(xse_factor)
        return PcwPolynomial.from_data_and_model(self.sample, model, jump_locator, self.weights)

    def with_penalty(
        self,
        penalty: np.float64,
        jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
    ) -> "PcwPolynomial":
        """Get the best model with the given penalty (gamma)."""
        model = self._sol.model_for_penalty(penalty)
        return PcwPolynomial.from_data_and_model(self.sample, model, jump_locator, self.weights)

    @property
    def cv_func(self) -> PcwConstFn:
        return PcwConstFn._from_rs(self._sol.cv_func())

    @property
    def downsampled_cv_func(self) -> PcwConstFn:
        return PcwConstFn._from_rs(self._sol.downsampled_cv_func())

    @property
    def cv_se_func(self) -> PcwConstFn:
        return PcwConstFn._from_rs(self._sol.cv_se_func())

    @property
    def downsampled_cv_se_func(self) -> PcwConstFn:
        return PcwConstFn._from_rs(self._sol.downsampled_cv_se_func())

    def model_func(self) -> PcwConstFn:
        mf = self._sol.model_func()
        return PcwConstFn(
            values=[PcwPolynomial.from_data_and_model(
                self.sample, m, weights=self.weights) for m in mf.values],
            jump_points=mf.jump_points,
        )


@dataclass(frozen=True)
class PcwPolynomial(PcwFn):
    funcs: List[np.polynomial.Polynomial]
    jumps: npt.NDArray[np.float64]
    _cut_idxs: Optional[npt.NDArray[np.int64]] = None

    @classmethod
    def from_data_and_model(
        Self,
        timeseries: TimeSeriesSample,
        model: _rs.ScoredPolyModel,
        jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
        weights: npt.NDArray[np.float64] = None,
    ) -> "PcwPolynomial":
        funcs = [model_params_to_poly(
            timeseries.sample_times, timeseries.values, seg_model, weights) for seg_model in model.model_params]
        match jump_locator:
            case JumpLocator.CONTINUITY_OPTIMIZED:
                jumps = continuity_opt_jumps(
                    funcs, model.cut_idxs, timeseries.sample_times)
            case JumpLocator.MIDPOINT:
                ts = timeseries.sample_times
                jumps = np.mean(np.vstack((ts[:-1], ts[1:])).T, axis=1)
        return Self(funcs, jumps, model.cut_idxs)

    @classmethod
    def fit_n_best(
        Self,
        timeseries: TimeSeriesSample,
        jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
        max_segment_degree: Optional[Integral] = 15,
        max_total_dof: Optional[Integral] = None,
        n_best: Integral = 1,
        weights: npt.NDArray[np.float64] = None,
    ) -> List["PcwPolynomial"]:
        """Fit the n models with the n lowest CV scores."""
        models = _rs.fit_pcw_poly(
            timeseries.sample_times,
            timeseries.values,
            max_total_dof,
            max_segment_degree + 1 if max_segment_degree is not None else None,
            weights,
        ).n_cv_minimizers(n_best)
        return [Self.from_data_and_model(timeseries, model, jump_locator, weights) for model in models]

    @classmethod
    def fit_ose(
        Self,
        timeseries: TimeSeriesSample,
        jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
        max_segment_degree: Optional[Integral] = None,
        max_total_dof: Optional[Integral] = None,
        weights: npt.NDArray[np.float64] = None,
    ) -> "PcwPolynomial":
        """Fit the best model with respect to the one standard error rule."""
        model = _rs.fit_pcw_poly(
            timeseries.sample_times,
            timeseries.values,
            max_total_dof,
            max_segment_degree + 1 if max_segment_degree is not None else None,
            weights,
        ).ose_best()
        return Self.from_data_and_model(timeseries, model, jump_locator, weights)

    @classmethod
    def fit_xse(
        Self,
        timeseries: TimeSeriesSample,
        xse_factor: Optional[np.float64] = 1.0,
        jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
        max_segment_degree: Optional[Integral] = None,
        max_total_dof: Optional[Integral] = None,
        weights: npt.NDArray[np.float64] = None,
    ) -> "PcwPolynomial":
        """Fit the best model with respect to the se_factor standard error rule."""
        model = _rs.fit_pcw_poly(
            timeseries.sample_times,
            timeseries.values,
            max_total_dof,
            max_segment_degree + 1 if max_segment_degree is not None else None,
            weights,
        ).xse_best(xse_factor)
        return Self.from_data_and_model(timeseries, model, jump_locator, weights)

    @classmethod
    def fit_with_penalty(
        Self,
        timeseries: TimeSeriesSample,
        penalty: np.float64,
        jump_locator=JumpLocator.CONTINUITY_OPTIMIZED,
        max_segment_degree: Optional[Integral] = None,
        max_total_dof: Optional[Integral] = None,
        weights: npt.NDArray[np.float64] = None,
    ) -> "PcwPolynomial":
        """Fit the best model with the given penalty (gamma)."""
        model = _rs.fit_pcw_poly(
            timeseries.sample_times,
            timeseries.values,
            max_total_dof,
            max_segment_degree + 1 if max_segment_degree is not None else None,
            weights,
        ).model_for_penalty(penalty)
        return Self.from_data_and_model(timeseries, model, jump_locator, weights)

    def __str__(self):
        body = (r") \\" "\n    ").join(f"{str(poly)} & x \\in [{poly.domain[0]}, {poly.domain[1]}"
                                       for poly in self.funcs
                                       )
        return r"\begin{cases}" f"\n{body}]\n" r"\end{cases}"

    def __format__(self, format_spec):
        body = (r") \\" "\n    ").join(f"{poly:{format_spec}} & x \\in [{poly.domain[0]}, {poly.domain[1]}"
                                       for poly in self.funcs
                                       )
        return r"\begin{cases}" f"\n{body}]\n" r"\end{cases}"


if __name__ == "__main__":
    # a simple example
    import numpy as np
    times = np.linspace(0, 10)
    values = np.hstack([5*times[:10]**2, times[10:30] - 10, (-times[30:] **
                       4 + times[30:]**3 + 10)/np.amax(-times[30:]**4 + times[30:]**3 + 10)])
    p1 = PcwPolynomial.fit_ose(TimeSeriesSample(times, values))
    p2 = PcwPolynomial.fit_n_best(TimeSeriesSample(times, values))[0]

    import matplotlib.pyplot as plt
    ts = np.linspace(0, 10, 3000)
    plt.scatter(times, values)
    plt.scatter(ts, p2(ts), label="Best", marker=".", alpha=0.3)
    plt.scatter(ts, p1(ts), label="OSE", marker=".")
    plt.legend()
    plt.show()
