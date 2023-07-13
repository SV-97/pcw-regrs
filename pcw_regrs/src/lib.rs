#![allow(dead_code, unused_variables)]
#![feature(let_chains)]

mod prelude;
mod stack;

use ndarray::Array2;
use polyfit_residuals as pr;
pub use prelude::*;
use solve_dp::solve_dp;
mod affine_min;
mod annotate;
mod approx;
mod cv;
mod solve_dp;

/// How many "steps into the future" we predict during cross validation
const CV_PREDICTION_COUNT: usize = 1;

/// Solution to the general regression problem
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    model_func: ModelFunc,
    cv_func: CvFunc,
    /// CV function downsampled to the jumps of the model function.
    down_cv_func: CvFunc,
}

fn all_residuals(
    timeseries_sample: &ValidTimeSeriesSample,
    user_params: &MatchedUserParams,
) -> Vec<Array2<OFloat>> {
    let max_degree = user_params.max_seg_dof.to_deg();
    let xs = timeseries_sample.times();
    let ys = timeseries_sample.response();
    match timeseries_sample.weights() {
        Some(weights) => {
            if cfg!(feature = "parallel_rayon") {
                pr::weighted::all_residuals_par(xs, ys, max_degree, weights)
            } else {
                pr::weighted::all_residuals(xs, ys, max_degree, weights)
            }
        }
        None => {
            if cfg!(feature = "parallel_rayon") {
                pr::all_residuals_par(xs, ys, max_degree)
            } else {
                pr::all_residuals(xs, ys, max_degree)
            }
        }
    }
}

/// Fit a piecewise polynomial to the timeseries sample.
///
/// The full model should locally spend no more than `max_seg_dof` degrees of freedom
/// and globally no more than `max_total_dof` for the full model.
pub fn fit_pcw_poly(
    timeseries_sample: &TimeSeriesSample,
    user_params: &UserParams,
) -> Result<Solution, FitError> {
    // Check the input timeseries for any potential problems
    let ts = ValidTimeSeriesSample::try_from(timeseries_sample)?;
    // Resolve optional parameters
    let matched_params = user_params.match_to_timeseries(&ts);
    // Calculate all residual errors of the non-pcw fits
    let residuals = all_residuals(&ts, &matched_params);
    Solution::try_new(
        &ts,
        &matched_params,
        move |segment_start_idx, segment_stop_idx, dof| {
            residuals[segment_start_idx]
                [[segment_stop_idx - segment_start_idx, usize::from(dof) - 1]]
        },
    )
}

impl Solution {
    pub fn try_new(
        timeseries_sample: &ValidTimeSeriesSample,
        user_params: &MatchedUserParams,
        training_err: impl Fn(usize, usize, DegreeOfFreedom) -> OFloat,
    ) -> Result<Solution, FitError> {
        let opt = solve_dp(timeseries_sample, user_params, training_err);
        todo!()
    }
}
