//! This library provides an interface to fit piecewise polynomial models to timeseries
//! data such that the resulting models are in some sense optimal.
//!
//! (Note that it really solves a more general problem; however the only full implementation
//! we provide is the piecewise polynomial case. To learn more about the more general problem
//! please have a look at the associated paper / thesis.
//! To use this more general interface you'll wanna implement [PcwApproximator].)
//!
//! A Python interface to this crate is available as `pcw_regrs_py`.
//!
//! ## Mathematical background
//!
//! The central optimization problem being solved for all γ ≥ 0 is
//! ```text
//! minₚ total_training_error(p) + γ complexity(p)
//! ```
//! where minimization is over possible piecewise polynomial models p for the timeseries,
//! the training error is measured using the (squared) L2-norm such that the produced
//! models are least-squares models and the complexity is measured as the sum of the "local
//! degrees of freedom" of the model.
//!
//! The returned model is the solution of this problem that additionally minimizes a
//! forward cross validation score.
//!
//! In the future a detailed description of the mathematical background
//! will become available in a paper. When this happens we'll link to it from here.
//!
//! ## Complexity
//!
//! The time complexity of the polynomial case is O(n³m) where n is the length of the timeseries
//! and m a bound on the local degree of the model.
//! For a more general bound please consider the accompanying thesis / paper.
//!
//! ## Examples
//!
//! Fitting a model to some data is simple:
//! We want to fit polynomial models so we import the corresponding types
//! and function.
//! We define a timeseries from some raw values. The sample times are in `ts`
//! and the corresponding values in `ys`.
//!
//! ```
//! use pcw_regrs::{fit_pcw_poly_primitive, ScoredPolyModel, SegmentModelSpec};
//! use pcw_fn::{VecPcwFn, PcwFn};
//! use std::num::NonZeroUsize;
//! let ts = [1., 2., 3., 3.25, 3.5, 3.75_f64];
//! let ys = [1.03771907, 0.96036854, 4.1921268, 6.03147004, 6.50596585, 6.10847852];
//! # let solution = fit_pcw_poly_primitive(&ts, &ys, None, Some(10)).unwrap();
//! # let cv_sol: ScoredPolyModel<_> = solution.cv_minimizer().unwrap();
//! # assert_eq!(
//! #     cv_sol.model,
//! #     VecPcwFn::try_from_iters(
//! #         [1],
//! #         [
//! #             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//! #             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(3).unwrap() },
//! #         ],
//! #     ).unwrap()
//! # );
//! # let ose_sol: ScoredPolyModel<_> = solution.ose_best().unwrap();
//! # assert_eq!(
//! #     ose_sol.model,
//! #     VecPcwFn::try_from_iters(
//! #         [1],
//! #         [
//! #             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//! #             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(1).unwrap() },
//! #         ],
//! #     ).unwrap()
//! # );
//! ```
//!
//! We fit a polynomial model with locally no more than 10 degrees of freedom to our data.
//!
//! ```
//! # use pcw_regrs::{fit_pcw_poly_primitive, ScoredPolyModel, SegmentModelSpec};
//! # use pcw_fn::{VecPcwFn, PcwFn};
//! # use std::num::NonZeroUsize;
//! # let ts = [1., 2., 3., 3.25, 3.5, 3.75_f64];
//! # let ys = [1.03771907, 0.96036854, 4.1921268, 6.03147004, 6.50596585, 6.10847852];
//! let solution = fit_pcw_poly_primitive(&ts, &ys, None, Some(10)).unwrap();
//! # let cv_sol: ScoredPolyModel<_> = solution.cv_minimizer().unwrap();
//! # assert_eq!(
//! #     cv_sol.model,
//! #     VecPcwFn::try_from_iters(
//! #         [1],
//! #         [
//! #             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//! #             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(3).unwrap() },
//! #         ],
//! #     ).unwrap()
//! # );
//! # let ose_sol: ScoredPolyModel<_> = solution.ose_best().unwrap();
//! # assert_eq!(
//! #     ose_sol.model,
//! #     VecPcwFn::try_from_iters(
//! #         [1],
//! #         [
//! #             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//! #             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(1).unwrap() },
//! #         ],
//! #     ).unwrap()
//! # );
//! ```
//!
//! and get the model corresponding to the absolute minimum of the CV score
//!
//! ```
//! # use pcw_regrs::{fit_pcw_poly_primitive, ScoredPolyModel, SegmentModelSpec};
//! # use pcw_fn::{VecPcwFn, PcwFn};
//! # use std::num::NonZeroUsize;
//! # let ts = [1., 2., 3., 3.25, 3.5, 3.75_f64];
//! # let ys = [1.03771907, 0.96036854, 4.1921268, 6.03147004, 6.50596585, 6.10847852];
//! # let solution = fit_pcw_poly_primitive(&ts, &ys, None, Some(10)).unwrap();
//! let cv_sol: ScoredPolyModel<_> = solution.cv_minimizer().unwrap();
//! assert_eq!(
//!     cv_sol.model,
//!     VecPcwFn::try_from_iters(
//!         [1],
//!         [
//!             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//!             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(3).unwrap() },
//!         ],
//!     ).unwrap()
//! );
//! # let ose_sol: ScoredPolyModel<_> = solution.ose_best().unwrap();
//! # assert_eq!(
//! #     ose_sol.model,
//! #     VecPcwFn::try_from_iters(
//! #         [1],
//! #         [
//! #             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//! #             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(1).unwrap() },
//! #         ],
//! #     ).unwrap()
//! # );
//! ```
//!
//! Finally we also compute the model corresponding to the one standard error rule.
//!
//! ```
//! # use pcw_regrs::{fit_pcw_poly_primitive, ScoredPolyModel, SegmentModelSpec};
//! # use pcw_fn::{VecPcwFn, PcwFn};
//! # use std::num::NonZeroUsize;
//! # let ts = [1., 2., 3., 3.25, 3.5, 3.75_f64];
//! # let ys = [1.03771907, 0.96036854, 4.1921268, 6.03147004, 6.50596585, 6.10847852];
//! # let solution = fit_pcw_poly_primitive(&ts, &ys, None, Some(10)).unwrap();
//! # let cv_sol: ScoredPolyModel<_> = solution.cv_minimizer().unwrap();
//! # assert_eq!(
//! #     cv_sol.model,
//! #     VecPcwFn::try_from_iters(
//! #         [1],
//! #         [
//! #             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//! #             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(3).unwrap() },
//! #         ],
//! #     ).unwrap()
//! # );
//! let ose_sol: ScoredPolyModel<_> = solution.ose_best().unwrap();
//! assert_eq!(
//!     ose_sol.model,
//!     VecPcwFn::try_from_iters(
//!         [1],
//!         [
//!             SegmentModelSpec { start_idx: 0, stop_idx: 1, model: NonZeroUsize::new(1).unwrap() },
//!             SegmentModelSpec { start_idx: 2, stop_idx: 5, model: NonZeroUsize::new(1).unwrap() },
//!         ],
//!     ).unwrap()
//! );
//! ```
//!
//! ## User parameter selection
//!
//! The model we'd usually recommend without further knowledge about the application is one with
//! locally no more than 10 degrees of freedom (so `max_total_dof = None`, `max_seg_dof = Some(10)`
//! when using [fit_pcw_poly_primitive]).
//! This yields good runtimes and models with most datasets we've tested.
//!
//! The CV criterion we recommend as default is the one standard error rule (available via
//! [Solution::ose_best]).
//!

mod affine_min;
mod annotate;
mod approximators;
pub mod solve_jump;
mod stack;

use approximators::PolynomialApproximator;
pub use approximators::SegmentModelSpec;
pub use approximators::{
    PcwApproximator, PcwPolynomialApproximator, PcwPolynomialArgs, PolynomialArgs,
    TimeSeries, /* PcwConstantApproximator */
};
use itertools::Itertools;
use ndarray::Array1;
use num_traits::{real::Real, Bounded, Float, FromPrimitive, Signed};
use ordered_float::{NotNan, OrderedFloat};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use solve_jump::dof::{ScoredModel, Solution, SolutionCore};
use std::{iter::Sum, num::NonZeroUsize, ops::AddAssign};
use thiserror::Error;

/// How many "steps into the future" we predict during cross validation
const CV_PREDICTION_COUNT: usize = 1;

/// A piecewise polynomial model for a timeseries and its cv score.
pub type ScoredPolyModel<'a, R> = ScoredModel<'a, R, R, R, PolynomialApproximator<R, R>>;

/// Squared euclidean metric d(x,y)=‖x-y‖₂².
#[inline]
pub fn euclid_sq_metric<T: Real>(x: &T, y: &T) -> T {
    let d = *x - *y;
    // d * d
    d.powi(2)
}

/// Fit a piecewise polynomial to the timeseries given by some sample `times` and corresponding
/// `response`s.
///
/// The full model should locally spend no more than `max_seg_dof` degrees of freedom
/// and globally no more than `max_total_dof` for the full model.
pub fn fit_pcw_poly<'a, R>(
    times: &[R],
    response: &[R],
    max_total_dof: Option<NonZeroUsize>,
    max_seg_dof: Option<NonZeroUsize>,
    weights: Option<&[R]>,
) -> Option<Solution<'a, R, R>>
where
    R: Real
        + Ord
        + Signed
        + Send
        + Sync
        + Sum<R>
        + 'static
        + Bounded
        + FromPrimitive
        + Default
        + AddAssign,
{
    Solution::try_new(
        max_total_dof,
        &PcwPolynomialApproximator::fit_metric_data_from_model(
            PcwPolynomialArgs {
                max_seg_dof,
                weights: weights.map(|w| Array1::from(w.to_owned())),
            },
            euclid_sq_metric,
            TimeSeries::new(&times, &response),
        ),
        euclid_sq_metric,
    )
}

/// Marker trait for primitive floating point types without [Ord] implementations.
pub trait PrimitiveFloat: Real + Float {}
impl PrimitiveFloat for f32 {}
impl PrimitiveFloat for f64 {}

/// The various kinds of errors that can happen during the polynomial fitting process.
#[derive(Error, Debug, PartialEq, Eq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PolyFitError {
    #[error("encountered a floating point NaN in the sequence of sample times")]
    NanInTimes,
    #[error("encountered a floating point NaN in the sequence of timeseries values / responses")]
    NanInResponses,
    #[error("can't fit a polynomial to an empty timeseries")]
    EmptyData,
}

/// Convenience wrapper for [fit_pcw_poly] that wraps floating point types into [OrderedFloat]s.
pub fn fit_pcw_poly_primitive<'a, R>(
    times: &[R],
    response: &[R],
    max_total_dof: Option<usize>,
    max_seg_dof: Option<usize>,
    weights: Option<&[R]>,
) -> Result<Solution<'a, OrderedFloat<R>, OrderedFloat<R>>, PolyFitError>
where
    R: PrimitiveFloat
        + Signed
        + Send
        + Sync
        + Sum
        + Bounded
        + FromPrimitive
        + Default
        + AddAssign
        + 'static,
{
    let times_nn = times
        .iter()
        .map(|x| -> Option<_> { NotNan::new(*x).ok() })
        .collect::<Option<Vec<_>>>()
        .ok_or(PolyFitError::NanInTimes)?
        .into_iter()
        .map(|x| OrderedFloat(x.into_inner()))
        .collect_vec();
    let response_nn = response
        .iter()
        .map(|x| -> Option<_> { NotNan::new(*x).ok() })
        .collect::<Option<Vec<_>>>()
        .ok_or(PolyFitError::NanInResponses)?
        .into_iter()
        .map(|x| OrderedFloat(x.into_inner()))
        .collect_vec();
    let weights_nn = weights
        .map(|w| {
            Ok(w.iter()
                .map(|x| -> Option<_> { NotNan::new(*x).ok() })
                .collect::<Option<Vec<_>>>()
                .ok_or(PolyFitError::NanInResponses)?
                .into_iter()
                .map(|x| OrderedFloat(x.into_inner()))
                .collect_vec())
        })
        .transpose()?;
    fit_pcw_poly(
        &times_nn,
        &response_nn,
        max_total_dof.map(|n| NonZeroUsize::new(n).unwrap()),
        max_seg_dof.map(|dof| NonZeroUsize::new(dof).unwrap()),
        weights_nn.as_deref(),
    )
    // TODO: propagate the specific error from a lower level rather than using options along the way.
    .ok_or(PolyFitError::EmptyData)
}

/// Turns a partial order on a type into a new one that forgets about the equality of objects;
/// so two objects x,y are comparable iff x < y or x > y in the original order.
///
/// # Example
/// Please see the test `test_strict_partial_cmp` for an example of how this function is intended
/// to be used.
pub(crate) fn strict_partial_cmp<T: PartialOrd>(x: &T, y: &T) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering::*;
    match x.partial_cmp(y) {
        Some(Equal) => None,
        x => x,
    }
}

#[test]
fn test_strict_partial_cmp() {
    use crate::strict_partial_cmp;
    use is_sorted::IsSorted;

    let v = vec![0, 1, 2];
    assert!(IsSorted::is_sorted(&mut v.iter()));
    assert!(IsSorted::is_sorted_by(&mut v.iter(), strict_partial_cmp));

    let v = vec![0, 1, 1, 2];
    assert!(IsSorted::is_sorted(&mut v.iter()));
    assert!(!IsSorted::is_sorted_by(&mut v.iter(), strict_partial_cmp));

    let v = vec![2, 1, 9, 2];
    assert!(!IsSorted::is_sorted(&mut v.iter()));
    assert!(!IsSorted::is_sorted_by(&mut v.iter(), strict_partial_cmp));
}
