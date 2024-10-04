#![feature(let_chains)]
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod prelude;
mod stack;
use annotate::Annotated;
use cv::cv_scores_and_models;
use derive_new::new;
use itertools::Itertools;
use ndarray::Array2;
use pcw_fn::{FunctorRef, PcwFn, VecPcwFn};
use polyfit_residuals as pr;
pub use prelude::*;
use solve_dp::solve_dp;
pub mod affine_min;
mod annotate;
mod approx;
mod cv;
pub mod solve_dp;

/// How many "steps into the future" we predict during cross validation
const CV_PREDICTION_COUNT: usize = 1;

/// A solution of the optimization problem providing an interface to find the globally
/// minimizing model of the CV score, the OSE optimal model and to investigate the CV and
/// model functions.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Solution<T> {
    model_func: ModelFunc<T>,
    cv_func: CvFunc<T>,
    /// CV function downsampled to the jumps of the model function.
    down_cv_func: CvFunc<T>,
}

/// Calculate all residual errors of the non-pcw fits
fn all_residuals<T>(
    ts: &ValidTimeSeriesSample<T>,
    up: &MatchedUserParams,
) -> impl Fn(usize, usize, DegreeOfFreedom) -> T
where
    T: OrdFloat,
{
    /// Calculate all residual errors of the non-pcw fits. Returns the raw residual data
    fn all_residuals_raw<S>(
        timeseries_sample: &ValidTimeSeriesSample<S>,
        user_params: &MatchedUserParams,
    ) -> Vec<Array2<S>>
    where
        S: OrdFloat,
    {
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
    let res = all_residuals_raw(ts, up);
    #[inline(always)]
    move |segment_start_idx, segment_stop_idx, dof| {
        // Safety: DegreeOfFreedom is guaranteed to be >= 1
        res[segment_start_idx][[segment_stop_idx - segment_start_idx, unsafe {
            usize::from(dof).unchecked_sub(1)
        }]]
    }
}

/// Fit a piecewise polynomial to the timeseries sample.
///
/// The full model should locally spend no more than `max_seg_dof` degrees of freedom
/// and globally no more than `max_total_dof` for the full model.
pub fn try_fit_pcw_poly<T>(
    timeseries_sample: &TimeSeriesSample<T::Base>,
    user_params: &UserParams,
) -> Result<Solution<T>, FitError>
where
    T: OrdFloat,
{
    // Check the input timeseries for any potential problems
    let ts = ValidTimeSeriesSample::try_from(timeseries_sample)?;
    // Resolve optional parameters
    let matched_params = user_params.match_to_timeseries(&ts);
    // Calculate all residual errors of the non-pcw fits
    let res = all_residuals(&ts, &matched_params);
    Solution::try_new(&ts, &matched_params, res)
}

/// A model for a timeseries and its CV score.
#[derive(new, Debug, Eq, PartialEq, Clone)]
pub struct ScoredModel<T> {
    /// A piecewise function where the domain are jump indices and the codomain models (so
    /// elements of Ω).
    pub model: VecPcwFn<usize, SegmentModelSpec>,
    /// The cross validation score of the full model.
    pub score: T,
}

/// An optimal model with respect to the one standard error rule.
pub type OseBestModel<T> = ScoredModel<T>;

/// An optimal model minimizing the CV score.
pub type CvMinimizerModel<T> = ScoredModel<T>;

/// Maps each penalty γ to the corresponding optimal model.
pub type ScoredModelFunc<T> = VecPcwFn<T, ScoredModel<T>>;

impl<T> Solution<T>
where
    T: OrdFloat,
{
    pub fn try_new(
        timeseries_sample: &ValidTimeSeriesSample<T>,
        user_params: &MatchedUserParams,
        training_err: impl Fn(usize, usize, DegreeOfFreedom) -> T,
    ) -> Result<Self, FitError> {
        // Solve dynamic program
        let dp_solution = solve_dp(timeseries_sample, user_params, &training_err);
        // Determine crossvalidation and model functions; so the functions mapping hyperparameters to the
        // corresponding optimal CV values and models
        let (cv_func, model_func) =
            cv_scores_and_models(timeseries_sample, user_params, &dp_solution, training_err);
        // resample the cv score function to the model function; folding intervals with a minimum
        // so if there's an interval on which the models are constant the CV score we associate to
        // this interval is the minimal one of any penalty on it.
        let cv_down =
            cv_func
                .clone()
                .resample_to::<VecPcwFn<_, _>, _>(model_func.clone(), |a, b| {
                    if a.data <= b.data {
                        a
                    } else {
                        b
                    }
                });
        Ok(Self {
            model_func,
            cv_func,
            down_cv_func: cv_down,
        })
    }

    /// Return the best model w.r.t. the "one standard error" rule.
    pub fn ose_best(&self) -> Option<OseBestModel<T>> {
        let Annotated {
            metadata: se_min,
            data: cv_min,
        } = *self
            .down_cv_func
            .funcs()
            .iter()
            // compare by cv score
            .min_by(|cv1, cv2| cv1.data.cmp(&cv2.data))?;

        let (selected_cv, selected_model) = self
            .down_cv_func
            .funcs()
            .iter()
            .zip(self.model_func.funcs().iter())
            // reverse since we want the highest gamma possible
            .rev()
            // find first model within one se of cv_min
            .find(|(cv, _model)| (cv.data - cv_min).abs() <= se_min)
            .unwrap();
        Some(OseBestModel::new(selected_model.clone(), selected_cv.data))
    }

    /// Return the best model w.r.t. the "x-times standard error" rule.
    pub fn xse_best(&self, x: T) -> Option<OseBestModel<T>> {
        let Annotated {
            metadata: se_min,
            data: cv_min,
        } = *self
            .down_cv_func
            .funcs()
            .iter()
            // compare by cv score
            .min_by(|cv1, cv2| cv1.data.cmp(&cv2.data))?;

        let (selected_cv, selected_model) = self
            .down_cv_func
            .funcs()
            .iter()
            .zip(self.model_func.funcs().iter())
            // reverse since we want the highest gamma possible
            .rev()
            // find first model within one se of cv_min
            .find(|(cv, _model)| (cv.data - cv_min).abs() <= x * se_min)
            .unwrap();
        Some(OseBestModel::new(selected_model.clone(), selected_cv.data))
    }

    /// Return the global minimizer of the CV score.
    pub fn cv_minimizer(&self) -> Option<CvMinimizerModel<T>> {
        self.n_cv_minimizers(1).and_then(|mut vec| vec.pop())
    }

    /// Return the models corresponding to the `n_best` lowest CV scores.
    pub fn n_cv_minimizers(&self, n_best: usize) -> Option<Vec<CvMinimizerModel<T>>> {
        // Sort models in ascending order and pick out the ones with the lowest CV score
        let best_models = self
            .down_cv_func
            .fmap_ref(|cv_and_se| cv_and_se.data) // we only need CV scores without the standard errors
            .into_funcs()
            .zip(self.model_func.funcs())
            .map(|(score, model)| ScoredModel::new(model.clone(), score))
            .sorted_by(|m1, m2| m1.score.cmp(&m2.score))
            .take(n_best)
            .collect();
        Some(best_models)
    }

    /// Returns the optimal model corresponding to the given penalty γ
    pub fn model_for_penalty(&self, penalty: T) -> ScoredModel<T> {
        let Annotated { data: cv_score, .. } = self.down_cv_func.func_at(&penalty);
        let model = self.model_func.func_at(&penalty);
        ScoredModel::new(model.clone(), *cv_score)
    }

    /// The cross validation function mapping hyperparameters γ to CV scores (and their
    /// approximate standard errors).
    pub fn cv_func(&self) -> &CvFunc<T> {
        &self.cv_func
    }

    /// The cross validation function downsampled to the jumps of the model function.
    pub fn downsampled_cv_func(&self) -> &CvFunc<T> {
        &self.down_cv_func
    }

    /// The model function mapping hyperparameters γ to the corresponding solutions of
    /// the penalized partition problem.
    pub fn model_func(&self) -> &ModelFunc<T> {
        &self.model_func
    }

    /// The model function mapping hyperparameters γ to the corresponding solutions of
    /// the penalized partition problem.
    pub fn scored_model_func(&self) -> ScoredModelFunc<T> {
        ScoredModelFunc::try_from_iters(
            self.model_func.jumps().iter().cloned(),
            self.model_func
                .funcs()
                .iter()
                .zip(
                    self.down_cv_func
                        .funcs()
                        .iter()
                        .map(|Annotated { data: cv_score, .. }| cv_score),
                )
                .map(|(model, cv_score)| ScoredModel::new(model.clone(), *cv_score)),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod solve_dp {

        use super::*;
        use crate::solve_dp::{BellmanTable, CutPath, OptimalJumpData, RefDofPartition};
        use ndarray::arr2;

        fn fit<T>(raw_data: Vec<T::Base>, up: UserParams) -> OptimalJumpData<T>
        where
            T: OrdFloat,
        {
            use num_traits::FromPrimitive;
            let times = (0..raw_data.len())
                .into_iter()
                .map(T::Base::from_usize)
                .map(Option::unwrap)
                .collect_vec();
            let timeseries_sample = TimeSeriesSample::try_new(&times, &raw_data, None).unwrap();
            let ts = ValidTimeSeriesSample::try_from(&timeseries_sample).unwrap();
            // Resolve optional parameters
            let up = up.match_to_timeseries(&ts);
            solve_dp(&ts, &up, all_residuals(&ts, &up))
        }

        #[test]
        #[cfg_attr(
            feature = "dofs-sub-one",
            ignore = "This test is only implemented for the case where the `dofs-sub-one` feature is disabled"
        )]
        fn optimal_jump_data() {
            use ordered_float::OrderedFloat;
            let raw_data = vec![8., 9., 10., 1., 4., 9., 16.];
            let opt = fit(raw_data.clone(), UserParams::default());

            let es = arr2(&[
                [
                    Some(OrderedFloat(0.0)),
                    Some(OrderedFloat(0.5000000000000006)),
                    Some(OrderedFloat(1.9999999999999991)),
                    Some(OrderedFloat(50.00000000000001)),
                    Some(OrderedFloat(57.20000000000001)),
                    Some(OrderedFloat(62.83333333333334)),
                    Some(OrderedFloat(134.8571428571429)),
                ],
                [
                    None,
                    Some(OrderedFloat(0.0)),
                    Some(OrderedFloat(3.0814879110195774e-31)),
                    Some(OrderedFloat(1.9999999999999991)),
                    Some(OrderedFloat(6.499999999999998)),
                    Some(OrderedFloat(34.666666666666664)),
                    Some(OrderedFloat(62.83333333333334)),
                ],
                [
                    None,
                    None,
                    Some(OrderedFloat(0.0)),
                    Some(OrderedFloat(3.0814879110195774e-31)),
                    Some(OrderedFloat(1.9999999999999991)),
                    Some(OrderedFloat(2.6666666666666656)),
                    Some(OrderedFloat(6.0)),
                ],
                [
                    None,
                    None,
                    None,
                    Some(OrderedFloat(0.0)),
                    Some(OrderedFloat(3.0814879110195774e-31)),
                    Some(OrderedFloat(0.6666666666666666)),
                    Some(OrderedFloat(1.9999999999999991)),
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    Some(OrderedFloat(0.0)),
                    Some(OrderedFloat(3.0814879110195774e-31)),
                    Some(OrderedFloat(3.5745259767827097e-31)),
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(OrderedFloat(0.0)),
                    Some(OrderedFloat(4.930380657631324e-32)),
                ],
                [None, None, None, None, None, None, Some(OrderedFloat(0.0))],
            ]);

            let pc = arr2(&[
                [
                    None,
                    Some(CutPath::NoCuts),
                    Some(CutPath::NoCuts),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(1),
                        elem_after_prev_cut: 3,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(1),
                        elem_after_prev_cut: 3,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(1),
                        elem_after_prev_cut: 3,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(1),
                        elem_after_prev_cut: 6,
                    }),
                ],
                [
                    None,
                    None,
                    Some(CutPath::NoCuts),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(2),
                        elem_after_prev_cut: 3,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(2),
                        elem_after_prev_cut: 4,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(1),
                        elem_after_prev_cut: 3,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(1),
                        elem_after_prev_cut: 3,
                    }),
                ],
                [
                    None,
                    None,
                    None,
                    Some(CutPath::NoCuts),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(3),
                        elem_after_prev_cut: 4,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(2),
                        elem_after_prev_cut: 3,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(1),
                        elem_after_prev_cut: 3,
                    }),
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    Some(CutPath::NoCuts),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(4),
                        elem_after_prev_cut: 5,
                    }),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(2),
                        elem_after_prev_cut: 3,
                    }),
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(CutPath::NoCuts),
                    Some(CutPath::SomeCuts {
                        remaining_dofs: dof!(3),
                        elem_after_prev_cut: 3,
                    }),
                ],
                [None, None, None, None, None, None, Some(CutPath::NoCuts)],
            ]);

            // One of these is computed using the old code. We verify against this
            assert_eq!(
                opt,
                OptimalJumpData {
                    energies: BellmanTable { energies: es },
                    prev_cuts: pc,
                    max_seg_dof: DegreeOfFreedom::try_from(7).unwrap(),
                }
            );
            let n = |v: Vec<usize>| -> Vec<DegreeOfFreedom> {
                v.into_iter()
                    .map(DegreeOfFreedom::try_from)
                    .map(Result::unwrap)
                    .collect()
            };
            // println!("{:?}", opt.energies);

            assert_eq!(
                RefDofPartition {
                    cut_indices: &vec![2],
                    segment_dofs: &n(vec![2, 3]),
                },
                RefDofPartition::from(&opt.optimal_cuts(dof!(5)).unwrap())
            );
            assert_eq!(
                RefDofPartition {
                    cut_indices: &vec![2],
                    segment_dofs: &n(vec![1, 3]),
                },
                RefDofPartition::from(&opt.optimal_cuts(dof!(4)).unwrap())
            );
            assert_eq!(
                RefDofPartition {
                    cut_indices: &vec![2],
                    segment_dofs: &n(vec![1, 2]),
                },
                RefDofPartition::from(&opt.optimal_cuts(dof!(3)).unwrap())
            );
        }
    }
}
