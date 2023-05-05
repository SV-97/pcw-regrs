//! The part of the solution revolving around the cross validation.
use super::OptimalJumpData;
use crate::affine_min::{pointwise_minimum_of_affines, AffineFunction};
use crate::annotate::Annotated;
use crate::approximators::{
    DegreeOfFreedom, ErrorApproximator, PcwApproximator, PcwPolynomialArgs, PolynomialArgs,
    SegmentModelSpec,
};

use crate::stack::Stack;
use ndarray::{s, Array1};
use pcw_fn::{Functor, FunctorRef, PcwFn, VecPcwFn};

use derive_new::new;
use itertools::Itertools;
use num_traits::real::Real;
use num_traits::{Bounded, FromPrimitive, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::iter;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{AddAssign, Deref};

/// Calculate the cross validation scores for all models of the approximation given the
/// corresponding optimal jump data. Each score is annotated with its standard error.
pub fn calc_cv_scores<T, D, E, S, A>(
    approx: &A,
    opt: &OptimalJumpData<E>,
    mut metric: impl FnMut(&D, &D) -> E,
    max_total_dof: Option<DegreeOfFreedom>,
) -> Option<CvFunc<E>>
where
    D: Clone,
    S: for<'a> ErrorApproximator<T, D, E, Model<'a> = PolynomialArgs<'a, D>>,
    A: PcwApproximator<S, T, D, E, Model = PcwPolynomialArgs<D>>,
    E: Bounded + Real + Ord + FromPrimitive + Default + AddAssign + std::iter::Sum,
{
    /// We'll use this type to annotate our CV scores with some metadata prior to finding the
    /// pointwise min of our affine functions so we can easily recover the n_dofs from the
    /// resulting function later on.
    #[derive(Clone, PartialEq, Eq, Debug)]
    struct MetaData<E> {
        cv_score: E,
        n_dofs: NonZeroUsize,
    }

    impl<E: Default> Default for MetaData<E> {
        fn default() -> Self {
            MetaData {
                cv_score: E::default(),
                n_dofs: unsafe { NonZeroUsize::new_unchecked(1) },
            }
        }
    }

    if approx.data_len().is_zero() {
        None
    } else {
        let data_len = approx.data_len();
        let mut cv_func = VecPcwFn::zero(); // this is CV : γ ↦ ∑ᵣ (CVᵣᵒᵖᵗ(γ))
        let mut cv_func_sq: VecPcwFn<E, E> = VecPcwFn::zero(); // this is  γ ↦ ∑ᵣ (CVᵣᵒᵖᵗ(γ))²

        // allocate two buffers that are used throughout the cut tracing to avoid allocations in the hot loop
        let mut cut_buffer = Stack::with_capacity(data_len - 1);
        let mut dof_buffer = Stack::with_capacity(data_len);

        // maybe_debug::dbg!(&opt);

        // `data_len - 1` without `..=` since we wanna calculate cross validation scores for
        // prediction into the future which isn't possible for the last data point.
        for rb in 0..data_len - 1 {
            let max_total_seg_dofs = {
                // Safety: rb starts at 0 such that rb+1 is nonzero. Furthermore we're only going up to
                // data_len - 1 <= usize::MAX such that there can be no overflows in the rb+1.
                let absolute_bound = unsafe { NonZeroUsize::new_unchecked(rb + 1) };
                match max_total_dof {
                    Some(k) => std::cmp::min(absolute_bound, k),
                    None => absolute_bound,
                }
            };
            // opt.max_total_seg_dofs(0, rb);
            // let max_seg_dofs = opt.max_seg_dofs(0, rb);
            // Allocating and consuming a new vec each iteration is faster than allocating outside
            // of the loop and draining it later on.
            let mut affines = Vec::with_capacity(usize::from(max_total_seg_dofs) + 1);

            // reversing iteration so that we generate the jumps in the correct order for the
            // pointwise minimization later on
            for n_dofs in (1..=max_total_seg_dofs.into()).rev() {
                let n_dofs_nonzero = unsafe { NonZeroUsize::new_unchecked(n_dofs) };
                let (training_error, cv_score) = match opt.optimal_cuts_on_subinterval_with_buf(
                    n_dofs_nonzero,
                    rb,
                    &mut cut_buffer,
                    &mut dof_buffer,
                ) {
                    Some(dof_partition) => {
                        let training_error = approx.total_training_error(
                            rb,
                            dof_partition.cut_indices,
                            dof_partition
                                .segment_dofs
                                .iter()
                                .cloned()
                                .map(PolynomialArgs::from),
                        );
                        // get the prediction error from the last cut to the end of the segment
                        let prediction_dof = *dof_partition.segment_dofs.last().unwrap();
                        let cv_score = approx.prediction_error(
                            unsafe { dof_partition.cut_indices.last().unwrap_unchecked() } + 1,
                            rb,
                            PolynomialArgs::from(prediction_dof),
                            &mut metric,
                        );
                        (training_error, cv_score)
                    }
                    None => {
                        // handle the 0 cut case where there is no "last optimal cut" (there isn't even a first one)
                        // but we still want a model
                        let cv_score = approx.prediction_error(
                            0,
                            rb,
                            PolynomialArgs::from(n_dofs_nonzero),
                            &mut metric,
                        );
                        let training_error = approx.total_training_error::<usize>(
                            rb,
                            None,
                            iter::once(PolynomialArgs::from(n_dofs_nonzero)),
                        );
                        (training_error, cv_score)
                    }
                };
                // maybe_debug::dbg!((&n_dofs, &training_error, &cv_score));

                affines.push(Annotated::new(
                    AffineFunction::new(
                        E::from_usize(n_dofs) // TODO: is this correct or do we want n_dofs?
                            .expect("Failed to create error slope from usize"),
                        training_error,
                    ),
                    MetaData { cv_score, n_dofs: n_dofs_nonzero },
                ));
            }
            // maybe_debug::dbg!(&affines);

            if !affines.is_empty() {
                let m1: VecPcwFn<_, _> =
                    pointwise_minimum_of_affines(affines.into_iter() /*.drain(..) */)
                        .expect("Failed to find pointwise minimum");
                // create piecewise constant function with CV scores on each segment
                let cv_r_opt = m1.fmap(|aff| aff.metadata.cv_score); // this is CVᵣᵒᵖᵗ = CVᵣ^*
                let cv_r_opt_sq = cv_r_opt.clone().fmap(|cv| cv.powi(2));
                // and add it to the total: we sum up the minimal piecewise cv scores over all possible
                // right boundaries
                cv_func = cv_func + cv_r_opt;
                cv_func_sq = cv_func_sq + cv_r_opt_sq;
            }
        }
        // maybe_debug::dbg!(&cv_func);
        let n = E::from_usize(data_len).unwrap();
        let k = E::from_usize(data_len - 1).unwrap();
        // standard error function
        let se_func = ((cv_func_sq - cv_func.fmap_ref(|x| x.powi(2) / n)).fmap(|x| x.sqrt() / n))
            .fmap(|x| x / k.sqrt());
        // match cv_func scaling to se_func
        cv_func = cv_func.fmap(|x| x / k);
        let (jumps, cvs) = cv_func.into_jumps_and_funcs();
        let ses = se_func.into_funcs();
        Some(
            PcwFn::try_from_iters(
                jumps,
                cvs.into_iter()
                    .zip(ses.into_iter())
                    .map(|(cv, se)| Annotated::new(cv, se)),
            )
            .unwrap(),
        )
    }
}

/// Calculate the functions mapping hyperparameter values to cross validation scores (including
/// standard errors) and model params.
pub fn cv_scores_and_models<'a, T, D, E, S, A>(
    max_total_dof: Option<DegreeOfFreedom>,
    approx: &A,
    mut metric: impl FnMut(&D, &D) -> E,
) -> Option<(CvFunc<E>, ModelFunc<'a, D, E>)>
where
    D: Clone,
    S: for<'b> ErrorApproximator<T, D, E, Model<'b> = PolynomialArgs<'b, D>>,
    A: Sync + PcwApproximator<S, T, D, E, Model = PcwPolynomialArgs<D>>,
    E: Bounded + Send + Sync + Real + Ord + FromPrimitive + Default + AddAssign + std::iter::Sum,
{
    let data_len = approx.data_len();
    if data_len.is_zero() {
        None
    } else {
        let opt = OptimalJumpData::from_approximations(max_total_dof, approx);
        let max_total_dof_u = max_total_dof
            .map(|max| std::cmp::min(data_len, usize::from(max)))
            .unwrap_or(data_len);

        // allocate two buffers that are used throughout the cut tracing to avoid allocations in the hot loop
        // there's at most one cut between all pairs of successive data points - so `data_len - 1`.
        let mut cut_buffer = Stack::with_capacity(data_len - 1);
        // In that case we gotta assign a dof to each data point.
        let mut dof_buffer = Stack::with_capacity(data_len);

        // reverse interval to have correct order for pointwise min
        let affines = (1..max_total_dof_u + 1).rev().map(|n_dofs| {
            let n_dofs_nonzero = unsafe { NonZeroUsize::new_unchecked(n_dofs) };
            let model: VecPcwFn<_, _> =
                match opt.optimal_cuts_with_buf(n_dofs_nonzero, &mut cut_buffer, &mut dof_buffer) {
                    Some(dof_partition) => crate::approximators::full_modelspec_on_seg(
                        approx,
                        0,
                        data_len - 1,
                        dof_partition
                            .segment_dofs
                            .as_ref()
                            .iter()
                            .cloned()
                            .map(PolynomialArgs::from),
                        dof_partition.cut_indices,
                    ),
                    None => crate::approximators::full_modelspec_on_seg(
                        approx,
                        0,
                        data_len - 1,
                        iter::once(PolynomialArgs::from(n_dofs_nonzero)),
                        None::<usize>,
                    ),
                };

            let model_error = model
                .funcs()
                .iter()
                .map(
                    |SegmentModelSpec {
                         start_idx,
                         stop_idx,
                         model: segment_model,
                     }| {
                        approx.training_error(
                            *start_idx,
                            *stop_idx,
                            segment_model.clone(),
                            &mut metric,
                        )
                    },
                )
                .sum();
            // let stats = partial_cvs.try_simple_statistics().unwrap();
            // let model_error = stats.sum;
            Annotated::new(
                AffineFunction::new(E::from_usize(n_dofs).unwrap(), model_error), // TODO: is this correct or do we want n_dofs?
                // TODO: I don't think the model here doesn't actually have to be an `Option`.
                // Find out why it is and remove it if it's really not needed. It might be due to the pointwise minimization later on requiring a Default.
                Some(model),
            )
        });

        let model_func: VecPcwFn<E, VecPcwFn<usize, SegmentModelSpec<PolynomialArgs<D>>>> =
            pointwise_minimum_of_affines::<_, _, _, VecPcwFn<_, _>>(affines)
                .expect("Failed to find pointwise min")
                .fmap(|aff| aff.metadata.expect("Encountered model-free γ-segment"));
        // .fmap(|aff| {
        //     let f = aff.metadata.expect("Encountered model-free γ-segment");
        //     f.fmap(
        //         |SegmentModelSpec {
        //              start_idx,
        //              stop_idx,
        //              model,
        //          }| SegmentModelSpec {
        //             start_idx,
        //             stop_idx,
        //             model: PolynomialArgs { dof: model.dof, weights: () },
        //         },
        //     )
        // });
        let score_func = calc_cv_scores(approx, &opt, &mut metric, max_total_dof)?;
        Some((score_func, ModelFunc(model_func)))
    }
}

/// Combines the CV function mapping hyperparamters (γ) to the corresponding CV scores
/// with the model function mapping hyperparamters to the corresponding optimal models.
pub struct ScoresAndModels<T, D, E: Ord, S>
where
    S: ErrorApproximator<T, D, E>,
{
    /// Maps hyperparameter values to CV-scores
    pub score_func: VecPcwFn<E, E>,
    /// Maps hyperparameter values to models
    pub model_func: VecPcwFn<E, VecPcwFn<usize, S>>,
    _phantomdata: PhantomData<(T, D)>,
}

impl<T, D, E: Ord, S> ScoresAndModels<T, D, E, S>
where
    S: ErrorApproximator<T, D, E>,
{
    pub fn new(score_func: VecPcwFn<E, E>, model_func: VecPcwFn<E, VecPcwFn<usize, S>>) -> Self {
        ScoresAndModels {
            score_func,
            model_func,
            _phantomdata: PhantomData,
        }
    }
}

/// Calculates the cross validation scores and jump points for all possible parameter choices
/// together with a function mapping parameter values to data models.
pub fn models_with_cv_scores<T, D, E, S, A>(
    max_total_dof: Option<DegreeOfFreedom>,
    approx: &A,
    mut metric: impl FnMut(&D, &D) -> E,
) -> Option<ScoresAndModels<T, D, E, S>>
where
    D: Clone,
    S: for<'a> ErrorApproximator<T, D, E, Model<'a> = PolynomialArgs<'a, D>>,
    A: Sync + PcwApproximator<S, T, D, E, Model = PcwPolynomialArgs<D>>,
    E: Bounded
        + Send
        + Sync
        + Real
        + Ord
        + Clone
        + FromPrimitive
        + Default
        + AddAssign
        + std::iter::Sum,
{
    eprintln!("Starting calculation of total CV scores");
    let data_len = approx.data_len();
    if data_len.is_zero() {
        None
    } else {
        let opt = OptimalJumpData::from_approximations(max_total_dof, approx);

        // allocate two buffers that are used throughout the cut tracing to avoid allocations in the hot loop
        // there's at most one cut between all pairs of successive data points - so `data_len - 1`.
        let mut cut_buffer = Stack::with_capacity(data_len - 1);
        // In that case we gotta assign a dof to each data point.
        let mut dof_buffer = Stack::with_capacity(data_len);

        // let cut_buffer = Stack::with_capacity(approx.data_len() - 1);
        // reverse interval to have correct order for pointwise min
        let affines = (1..data_len + 1).rev().map(|n_dofs| {
            let n_dofs_nonzero = unsafe { NonZeroUsize::new_unchecked(n_dofs) };
            let model: VecPcwFn<_, _> =
                match opt.optimal_cuts_with_buf(n_dofs_nonzero, &mut cut_buffer, &mut dof_buffer) {
                    Some(dof_partition) => crate::approximators::full_approximation_on_seg(
                        approx,
                        0,
                        data_len - 1,
                        dof_partition
                            .segment_dofs
                            .as_ref()
                            .iter()
                            .cloned()
                            .map(PolynomialArgs::from),
                        dof_partition.cut_indices,
                    ),
                    None => crate::approximators::full_approximation_on_seg(
                        approx,
                        0,
                        data_len - 1,
                        iter::once(PolynomialArgs::from(n_dofs_nonzero)),
                        None::<usize>,
                    ),
                };
            let model_error = model.funcs().iter().map(|seg| seg.training_error()).sum();
            // let model_error = approx.total_training_error(&optimal_cuts, data_len - 1);
            Annotated::new(
                AffineFunction::new(E::from_usize(n_dofs).unwrap(), model_error), // TODO: is this correct or do we want n_dofs?
                Some(model),
            )
        });
        let model_func = pointwise_minimum_of_affines::<_, _, _, VecPcwFn<_, _>>(affines)
            .expect("Failed to find pointwise min")
            .fmap(|aff| aff.metadata.expect("Encountered model-free γ-segment"));
        eprintln!("Finished calculation of total CV scores");
        eprintln!("Starting calculation of remaining CV scores");
        let cv_func = calc_cv_scores(approx, &opt, &mut metric, max_total_dof)?;
        eprintln!("Finished calculation of remaining CV scores");
        Some(ScoresAndModels::new(
            cv_func.fmap(|annotated_score| annotated_score.data),
            model_func,
        ))
    }
}

/// The result of running the algorithm finding the absolute minimum of the CV score.
#[derive(new)]
pub struct CrossValidationResult<'a, T, D, E, S>
where
    E: Ord,
    S: ErrorApproximator<T, D, E>,
{
    /// The best models ordered by CV score (ascending - lower is better)
    pub best_models: Vec<ScoredModel<'a, T, D, E, S>>,
    /// The full cross validation function: it maps γ to the CV score of the solution
    /// to the γ-penalized partition problem.
    pub full_cv_func: VecPcwFn<E, E>,
    /// The cross validation function min-downsampled such that on any segment the
    /// resulting best model is constant.
    pub downsampled_cv_func: VecPcwFn<E, E>,
}

/// An optimal model with respect to the one standard error rule.
pub type OseBestModel<'a, T, D, E, S> = ScoredModel<'a, T, D, E, S>;

/// An optimal model minimizing the CV score.
pub type CvMinimizerModel<'a, T, D, E, S> = ScoredModel<'a, T, D, E, S>;

/// Maps each penalty γ to the corresponding optimal models given as piecewise model
/// specifications: the arguments of the "inner" piecewise function are jump indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFunc<'a, D, E>(
    VecPcwFn<E, VecPcwFn<usize, SegmentModelSpec<PolynomialArgs<'a, D>>>>,
);

impl<'a, D, E> Deref for ModelFunc<'a, D, E> {
    type Target = VecPcwFn<E, VecPcwFn<usize, SegmentModelSpec<PolynomialArgs<'a, D>>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// pub type ModelFunc<E> = VecPcwFn<E, VecPcwFn<usize, SegmentModelSpec<NonZeroUsize>>>;

/// A function mapping hyperparameter values to CV scores; each score being annotated with
/// its standard error.
pub type CvFunc<E> = VecPcwFn<E, Annotated<E, E>>;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFuncCore<E>(VecPcwFn<E, VecPcwFn<usize, SegmentModelSpec<NonZeroUsize>>>);

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionCore<D, E> {
    model_func: ModelFuncCore<E>,
    cv_func: CvFunc<E>,
    /// CV function downsampled to the jumps of the model function.
    down_cv_func: CvFunc<E>,
    /// Weights used in weighted regression
    weights: Option<Array1<D>>,
}

impl<'a, D, E> From<ModelFunc<'a, D, E>> for ModelFuncCore<E> {
    fn from(ModelFunc(f): ModelFunc<'a, D, E>) -> Self {
        ModelFuncCore(f.fmap(|g| {
            g.fmap(
                |SegmentModelSpec {
                     start_idx,
                     stop_idx,
                     model,
                 }| SegmentModelSpec {
                    start_idx,
                    stop_idx,
                    model: model.dof,
                },
            )
        }))
    }
}

impl<'a, D, E> From<Solution<'a, D, E>> for SolutionCore<D, E> {
    fn from(
        Solution {
            model_func,
            cv_func,
            down_cv_func,
            weights,
        }: Solution<'a, D, E>,
    ) -> Self {
        SolutionCore {
            model_func: ModelFuncCore::from(model_func),
            cv_func,
            down_cv_func,
            weights,
        }
    }
}

impl<'a, D, E> From<&'a SolutionCore<D, E>> for Solution<'a, D, E>
where
    D: Clone,
    E: Clone,
{
    fn from(
        SolutionCore {
            model_func,
            cv_func,
            down_cv_func,
            weights,
        }: &'a SolutionCore<D, E>,
    ) -> Self {
        Solution {
            model_func: ModelFunc(model_func.0.fmap_ref(|idxs_to_models| {
                idxs_to_models.fmap_ref(
                    |SegmentModelSpec {
                         start_idx,
                         stop_idx,
                         model,
                     }| SegmentModelSpec {
                        start_idx: *start_idx,
                        stop_idx: *stop_idx,
                        model: PolynomialArgs {
                            dof: *model,
                            weights: weights
                                .as_ref()
                                .map(|w| w.slice(s![*start_idx..=*stop_idx])),
                        },
                    },
                )
            })),
            cv_func: cv_func.clone(),
            down_cv_func: down_cv_func.clone(),
            weights: weights.clone(),
        }
    }
}

/// A model for a timeseries and its CV score.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ScoredModel<'a, T, D, E, S>
where
    E: Ord,
    S: ErrorApproximator<T, D, E>,
{
    /// A piecewise function where the domain are jump indices and the codomain models (so
    /// elements of Ω).
    pub model: VecPcwFn<usize, SegmentModelSpec<S::Model<'a>>>,
    /// The cross validation score of the full model.
    pub score: E,
    _phantom: PhantomData<(T, D)>,
}

impl<'a, T, D, E: Ord, S> ScoredModel<'a, T, D, E, S>
where
    S: ErrorApproximator<T, D, E>,
{
    pub fn new(model: VecPcwFn<usize, SegmentModelSpec<S::Model<'a>>>, score: E) -> Self {
        Self {
            model,
            score,
            _phantom: PhantomData,
        }
    }
}

/// A solution of the optimization problem providing an interface to find the globally
/// minimizing model of the CV score, the OSE optimal model and to investigate the CV and
/// model functions.
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution<'a, D, E> {
    model_func: ModelFunc<'a, D, E>,
    cv_func: CvFunc<E>,
    /// CV function downsampled to the jumps of the model function.
    down_cv_func: CvFunc<E>,
    /// Weights used in weighted regression
    weights: Option<Array1<D>>,
}

impl<'a, D, E> Solution<'a, D, E>
where
    E: Bounded + Send + Sync + Real + Ord + FromPrimitive + Default + AddAssign + std::iter::Sum,
{
    /// Try to calculate the solution for a given approximator using some metric and at most
    /// `max_tota_dof` degrees of freedom for the full model.
    pub fn try_new<T, S, A>(
        max_total_dof: Option<NonZeroUsize>,
        approx: &A,
        metric: impl FnMut(&D, &D) -> E,
    ) -> Option<Self>
    where
        D: Clone,
        S: for<'b> ErrorApproximator<T, D, E, Model<'b> = PolynomialArgs<'b, D>>,
        A: Sync + PcwApproximator<S, T, D, E, Model = PcwPolynomialArgs<D>>,
    {
        let (cv_func, model_func) = cv_scores_and_models(max_total_dof, approx, metric)?;
        // resample the cv score function to the model function; folding intervals with a minimum
        // so if there's an interval on which the models are constant the CV score we associate to
        // this interval is the minimal one of any penalty on it.
        let cv_down =
            cv_func
                .clone()
                .resample_to::<VecPcwFn<_, _>, _>(model_func.0.clone(), |a, b| {
                    if a.data <= b.data {
                        a
                    } else {
                        b
                    }
                });

        Some(Self {
            model_func,
            cv_func,
            down_cv_func: cv_down,
            weights: approx.model().weights.clone(),
        })
    }

    /// Return the best model w.r.t. the "one standard error" rule.
    pub fn ose_best<T, S>(&'a self) -> Option<OseBestModel<T, D, E, S>>
    where
        D: Clone,
        S: for<'b> ErrorApproximator<T, D, E, Model<'b> = PolynomialArgs<'b, D>>,
    {
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
            .find(|(cv, _model)| (cv.data - cv_min).abs() <= E::from_usize(1).unwrap() * se_min)
            .unwrap();
        Some(OseBestModel::new(selected_model.clone(), selected_cv.data))
    }

    /// Return the global minimizer of the CV score.
    pub fn cv_minimizer<T, S>(&'a self) -> Option<CvMinimizerModel<T, D, E, S>>
    where
        D: Clone,
        S: for<'b> ErrorApproximator<T, D, E, Model<'b> = PolynomialArgs<'b, D>>,
    {
        self.n_cv_minimizers(1).and_then(|mut vec| vec.pop())
    }

    /// Return the models corresponding to the `n_best` lowest CV scores.
    pub fn n_cv_minimizers<T, S>(
        &'a self,
        n_best: usize,
    ) -> Option<Vec<CvMinimizerModel<T, D, E, S>>>
    where
        D: Clone,
        S: for<'b> ErrorApproximator<T, D, E, Model<'b> = PolynomialArgs<'b, D>>,
        E: Bounded
            + Send
            + Sync
            + Real
            + Ord
            + FromPrimitive
            + Default
            + AddAssign
            + std::iter::Sum,
    {
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

    /// The cross validation function mapping hyperparameters γ to CV scores (and their
    /// approximate standard errors).
    pub fn cv_func(&self) -> &CvFunc<E> {
        &self.cv_func
    }

    /// The cross validation function downsampled to the jumps of the model function.
    pub fn downsampled_cv_func(&self) -> &CvFunc<E> {
        &self.down_cv_func
    }

    /// The model function mapping hyperparameters γ to the corresponding solutions of
    /// the penalized partition problem.
    pub fn model_func(&self) -> &ModelFunc<'a, D, E> {
        &self.model_func
    }
}
