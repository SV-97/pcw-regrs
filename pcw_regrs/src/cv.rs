use num_traits::real::Real;
use num_traits::Zero;
use pcw_fn::{Functor, FunctorRef, PcwFn, VecPcwFn};

use crate::affine_min::{pointwise_minimum_of_affines, AffineFunction};
use crate::annotate::Annotated;
use crate::solve_dp::OptimalJumpData;
use crate::stack::Stack;
use crate::{approx, prelude::*};
use std::iter;

/// Calculate the functions mapping hyperparameter values to cross validation scores (including
/// standard errors) and model params.
pub fn cv_scores_and_models<T>(
    timeseries_sample: &ValidTimeSeriesSample<T>,
    user_params: &MatchedUserParams,
    dp_solution: &OptimalJumpData<T>,
    training_err: impl Fn(usize, usize, DegreeOfFreedom) -> T,
) -> (CvFunc<T>, ModelFunc<T>)
where
    T: OrdFloat,
{
    let data_len = usize::from(timeseries_sample.len());

    let opt = dp_solution;
    // allocate two buffers that are used throughout the cut tracing to avoid allocations in the hot loop
    // there's at most one cut between all pairs of successive data points - so `data_len - 1`.
    let mut cut_buffer = Stack::with_capacity(data_len - 1);
    // In that case we gotta assign a dof to each data point => capacity is length of input data
    let mut dof_buffer = Stack::with_capacity(data_len);

    // reverse interval to have correct order for pointwise min
    // TODO: Use Range<NonZeroUsize> once implemented.
    // See https://internals.rust-lang.org/t/impl-range-and-rangeinclusive-for-nonzerou/18721/5
    let affines = (1..usize::from(user_params.max_total_dof) + 1)
        .rev()
        .map(|n_dofs| {
            // Safety: the range starts at 1
            let n_dofs_nonzero = DegreeOfFreedom::try_from(n_dofs).unwrap();
            let model: VecPcwFn<_, _> =
                match opt.optimal_cuts_with_buf(n_dofs_nonzero, &mut cut_buffer, &mut dof_buffer) {
                    Some(dof_partition) => crate::approx::full_modelspec_on_seg(
                        0,
                        data_len - 1,
                        dof_partition.segment_dofs.as_ref().iter().cloned(),
                        dof_partition.cut_indices.iter().copied(),
                    ),
                    None => crate::approx::full_modelspec_on_seg(
                        0,
                        data_len - 1,
                        iter::once(n_dofs_nonzero),
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
                         seg_dof: segment_model,
                     }| { training_err(*start_idx, *stop_idx, *segment_model) },
                )
                .sum();
            // let stats = partial_cvs.try_simple_statistics().unwrap();
            // let model_error = stats.sum;
            Annotated::new(
                AffineFunction::new(T::from_usize(n_dofs).unwrap(), model_error),
                // TODO: I don't think the model here doesn't actually have to be an `Option`.
                // Find out why it is and remove it if it's really not needed. It might be due to the pointwise minimization later on requiring a Default.
                Some(model),
            )
        });

    let model_func = pointwise_minimum_of_affines::<_, _, _, VecPcwFn<_, _>>(affines)
        .expect("Failed to find pointwise min")
        .fmap(|aff| aff.metadata.expect("Encountered model-free γ-segment"));
    let score_func = calc_cv_scores(timeseries_sample, user_params, opt);
    (score_func, model_func)
}

/// We'll use this type to annotate our CV scores with some metadata prior to finding the
/// pointwise min of our affine functions so we can easily recover the n_dofs from the
/// resulting function later on.
#[derive(Clone, PartialEq, Eq, Debug)]
struct MetaData<E> {
    cv_score: E,
    n_dofs: DegreeOfFreedom,
}

impl<E: Default> Default for MetaData<E> {
    fn default() -> Self {
        MetaData {
            cv_score: E::default(),
            n_dofs: DegreeOfFreedom::one(),
        }
    }
}

/// Calculate the cross validation scores for all models of the approximation given the
/// corresponding optimal jump data. Each score is annotated with its standard error.
// TODO: Refactor this, it's quite long
pub fn calc_cv_scores<T>(
    timeseries_sample: &ValidTimeSeriesSample<T>,
    user_params: &MatchedUserParams,
    opt: &OptimalJumpData<T>,
) -> CvFunc<T>
where
    T: OrdFloat,
{
    let data_len = usize::from(timeseries_sample.len());
    let mut cv_func = VecPcwFn::zero(); // this is CV : γ ↦ ∑ᵣ (CVᵣᵒᵖᵗ(γ))
    let mut cv_func_sq: VecPcwFn<T, T> = VecPcwFn::zero(); // this is  γ ↦ ∑ᵣ (CVᵣᵒᵖᵗ(γ))²

    // allocate two buffers that are used throughout the cut tracing to avoid allocations in the hot loop
    let mut cut_buffer = Stack::with_capacity(data_len - 1);
    let mut dof_buffer = Stack::with_capacity(data_len);

    // maybe_debug::dbg!(&opt);

    // `data_len - 1` without `..=` since we wanna calculate cross validation scores for
    // prediction into the future which isn't possible for the last data point.
    for rb in 0..data_len - crate::CV_PREDICTION_COUNT {
        let max_total_seg_dofs = {
            // Safety: rb starts at 0 such that rb+1 is nonzero. Furthermore we're only going up to
            // data_len - 1 <= usize::MAX such that there can be no overflows in the rb+1.
            let upper_bound_due_to_segment_len = if cfg!(feature = "dofs-sub-one") {
                if rb == 0 {
                    // there can be no model on data[0] if we don't allow a constant model in this case
                    continue;
                } else {
                    DegreeOfFreedom::try_from(rb).unwrap()
                }
            } else {
                DegreeOfFreedom::try_from(rb + 1).unwrap()
            };
            std::cmp::min(upper_bound_due_to_segment_len, user_params.max_total_dof)
        };
        // opt.max_total_seg_dofs(0, rb);
        // let max_seg_dofs = opt.max_seg_dofs(0, rb);
        // Allocating and consuming a new vec each iteration is faster than allocating outside
        // of the loop and draining it later on.
        let mut affines = Vec::with_capacity(usize::from(max_total_seg_dofs) + 1);

        // reversing iteration so that we generate the jumps in the correct order for the
        // pointwise minimization later on
        for n_dofs in (1..=max_total_seg_dofs.into()).rev() {
            let n_dofs_nonzero = DegreeOfFreedom::try_from(n_dofs).unwrap();
            let (training_error, cv_score) = match opt.optimal_cuts_on_subinterval_with_buf(
                n_dofs_nonzero,
                rb,
                &mut cut_buffer,
                &mut dof_buffer,
            ) {
                Some(dof_partition) => {
                    let training_error = approx::total_training_error(
                        timeseries_sample,
                        rb,
                        dof_partition.cut_indices.iter().copied(),
                        dof_partition.segment_dofs.iter().copied(),
                    );
                    // get the prediction error from the last cut to the end of the segment
                    let prediction_dof = *dof_partition.segment_dofs.last().unwrap();
                    let cv_score = approx::prediction_error(
                        timeseries_sample,
                        unsafe { dof_partition.cut_indices.last().unwrap_unchecked() } + 1,
                        rb,
                        prediction_dof,
                    );
                    (training_error, cv_score)
                }
                None => {
                    // handle the 0 cut case where there is no "last optimal cut" (there isn't even a first one)
                    // but we still want a model
                    let cv_score =
                        approx::prediction_error(timeseries_sample, 0, rb, n_dofs_nonzero);
                    let training_error = approx::total_training_error(
                        timeseries_sample,
                        rb,
                        None,
                        iter::once(n_dofs_nonzero),
                    );
                    (training_error, cv_score)
                }
            };

            //  struct for getting debug info about partial cv scores
            // #[derive(Debug)]
            // struct Dbg<'a, T> {
            //     right_boundary: usize,
            //     n_dofs: usize,
            //     cv_score: &'a T,
            // }
            // maybe_debug::dbg!(Dbg {
            //     right_boundary: rb,
            //     n_dofs: n_dofs,
            //     cv_score: &cv_score
            // });

            affines.push(Annotated::new(
                AffineFunction::new(
                    T::from_usize(n_dofs) // TODO: is this correct or do we want n_dofs?
                        .expect("Failed to create error slope from usize"),
                    training_error,
                ),
                MetaData {
                    cv_score,
                    n_dofs: n_dofs_nonzero,
                },
            ));
        }
        // maybe_debug::dbg!(&affines);

        if !affines.is_empty() {
            let m1: VecPcwFn<_, _> =
                pointwise_minimum_of_affines(affines.into_iter() /*.drain(..) */)
                    .expect("Failed to find pointwise minimum");
            // create piecewise constant function with CV scores on each segment
            let cv_r_opt = m1.fmap(|aff| aff.metadata.cv_score); // this is CVᵣᵒᵖᵗ = CVᵣ^*
            let cv_r_opt_sq = cv_r_opt.clone().fmap(|cv| Real::powi(cv, 2));
            // and add it to the total: we sum up the minimal piecewise cv scores over all possible
            // right boundaries
            cv_func = cv_func + cv_r_opt;
            cv_func_sq = cv_func_sq + cv_r_opt_sq;
        }
    }
    // maybe_debug::dbg!(&cv_func);
    let n = T::from_usize(data_len).unwrap();
    let k = T::from_usize(data_len - 1).unwrap();
    // standard error function
    let se_func = ((cv_func_sq - cv_func.fmap_ref(|&x| Real::powi(x, 2) / n))
        .fmap(|x| Real::sqrt(x) / n))
    .fmap(|x| x / k.sqrt());
    // match cv_func scaling to se_func
    cv_func = cv_func.fmap(|x| x / k);
    let (jumps, cvs) = cv_func.into_jumps_and_funcs();
    let ses = se_func.into_funcs();

    PcwFn::try_from_iters(
        jumps,
        cvs.into_iter()
            .zip(ses)
            .map(|(cv, se)| Annotated::new(cv, se)),
    )
    .unwrap()
}
