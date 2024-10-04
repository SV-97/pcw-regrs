//! The part of the solution concerned with the optimal partitions and general implementation
//! of the dynamic program.

use std::{cmp, mem::MaybeUninit};

use crate::{
    max,
    prelude::*,
    stack::Stack,
    {dof, min},
};

use ndarray::{s, Array2};

/// A single "cut" of the partition given by some reference data telling us which cut should come before it.
///
/// See also the [try_cut_path_to_cut] function to see how this this type relates to an actual cut.
/// Note that this essentially represents a linked list through some indirection given by the DP solution.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CutPath {
    SomeCuts {
        /// How many degrees of freedom may still be expended
        remaining_dofs: DegreeOfFreedom,
        /// The index of the data point after the cut
        elem_after_prev_cut: usize,
    },
    NoCuts,
}

/// Represents a solution of the key dynamic program.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct OptimalJumpData<T>
where
    T: OrdFloat,
{
    /// `energies[[k,n]]` contains the optimal energy when approximating `data[0..=n]` with exactly
    /// `k + 1` degrees of freedom.
    // TODO: optimize by exploiting triangle structure
    pub energies: BellmanTable<T>,
    /// Index `[[k,r]]` will tell us the index of the element after the previous cut,
    /// and how many degrees of freedom may still be expended on the left segment if
    /// we start with `data[0..=r]` and `k + 1` degrees of freedom.
    // A prev value of 0 means there's a jump in front of the first element - which is
    // really no jump at all.
    // TODO: This may actually be `k + 2` according to some comment. Please verify
    // TODO: optimize by exploiting triangle structure
    // TODO: remove first column since it's always full of `None`s
    pub prev_cuts: Array2<Option<CutPath>>,
    /// The maximal number of degrees of freedom to be spent on a single segment
    pub(crate) max_seg_dof: DegreeOfFreedom,
}

/// Solves the core dynamic program
pub fn solve_dp<T>(
    timeseries_sample: &ValidTimeSeriesSample<T>,
    user_params: &MatchedUserParams,
    // Calculates the residual error / training error of a (non-pcw) model fit given a
    // segment_start_idx, segment_stop_idx (into the timeseries sample; inclusive) and
    // the number of degrees of freedom of the approximation.
    training_error: impl Fn(usize, usize, DegreeOfFreedom) -> T,
) -> OptimalJumpData<T>
where
    T: OrdFloat,
{
    OptimalJumpData::from_errors(timeseries_sample, user_params, training_error)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BellmanBuilder<F, T> {
    energies: Array2<Option<T>>,
    /// Maps segment start and end indices (both inclusive) of the timeseries
    /// and a number of degrees of freedom into the residual error of the
    /// model with the given number of dofs on that segment.
    training_error: F,
}

impl<F, T> BellmanBuilder<F, T>
where
    T: OrdFloat,
    F: Fn(usize, usize, DegreeOfFreedom) -> T,
{
    pub fn new(max_total_dof: DegreeOfFreedom, data_len: usize, training_error: F) -> Self {
        let mut energies = Array2::from_elem((usize::from(max_total_dof), data_len), None);
        // apply initial conditions
        for (segment_end_idx, energy) in energies.slice_mut(s![0, ..]).iter_mut().enumerate() {
            *energy = Some(training_error(0, segment_end_idx, DegreeOfFreedom::one()));
        }
        Self {
            energies,
            training_error,
        }
    }

    /// Immutable core algorithm of `step`
    #[inline(always)]
    fn step_core<M>(
        &self,
        right_boundary: usize,
        k_dof_plus: usize,
        max_seg_dof: usize,
        minimizer: &mut M,
    ) -> MinimizationSolution<CutPath, T>
    where
        M: Minimizer<MinimizationSolution<CutPath, T>>,
    {
        minimizer.minimize(move |mut minh| {
            // "for all partitions of data[0..=right_boundary] into data[0..l], data[l..=right_boundary]"
            for l in 0..=right_boundary {
                let p_r_min = max!(
                    1,
                    // k_dof_plus.saturating_sub(l + 1), // this is implied by a later condition
                    // we have to have l > p_l so that there's enough datapoints in the segment to "spend" all the dofs
                    // which is equivalent to this condition on p_r
                    k_dof_plus.saturating_sub(l),
                );
                let p_r_max = min!(
                    right_boundary + 1 - l, // p_r can't be more larger than the number of data point in the segment
                    max_seg_dof, // it's also restricted by the maximal dof we allow on any segment
                    k_dof_plus - 1, // and of course the total dof count minus 1 since 1 dof has to be spent on p_l
                );
                // "for all valid (subject to the extra constraints) degrees on freedom on data[l..=right_boundary]"
                for p_r in p_r_min..=p_r_max {
                    // note that p_l can never be smaller than 1 because p_r is bounded above by k_dof_plus - 1 per definition
                    let p_l = k_dof_plus - p_r;
                    let d = (self.training_error)(
                        l + 1,
                        right_boundary + 1,
                        // p_r is >= 1 because p_r_min >= 1 per definition
                        // So we could use the unsafe new_unchecked here safely
                        // however the compiler somehow prefers this version - and since it's safe as well, we'll just go with it
                        DegreeOfFreedom::try_from(p_r).unwrap(),
                    );

                    // `p_l - 1` to account for offset in bellman array indexing
                    let energy = unsafe { self.energies[[p_l - 1, l]].unwrap_unchecked() };
                    minh.add_to_min(MinimizationSolution {
                        arg_min: if p_l == 0 {
                            // Safety: if we were to enter this branch the previous p_l - 1 would've already underflowed
                            unsafe { std::hint::unreachable_unchecked() } // CutPath::NoCuts
                        } else {
                            CutPath::SomeCuts {
                                // Saftey: we've previously checkecd that p_l is not zero
                                remaining_dofs: unsafe { DegreeOfFreedom::new_unchecked(p_l) },
                                elem_after_prev_cut: l + 1,
                            }
                        },
                        min: energy + d,
                    });
                }
            }
            // handle the case where we approximate the whole segment with a single model: so p_r = k+1, p_l = 0
            if k_dof_plus
                <= cmp::min(
                    max_seg_dof, // it's also restricted by the maximal dof we allow on any segment
                    right_boundary + 2,
                )
            {
                minh.add_to_min(MinimizationSolution {
                    arg_min: CutPath::NoCuts,
                    min: (self.training_error)(
                        0,
                        right_boundary + 1,
                        DegreeOfFreedom::try_from(k_dof_plus).unwrap(),
                    ),
                });
            }
        })
    }

    /// Solves the minimization problem in the forward step of the bellman equation.
    /// The returned `MinimizationSolution` consists of the optimal energy together with a cut path,
    /// where the cut path is isomorphic to the argmin in the bellman equation
    #[inline(always)]
    pub fn step<M>(
        &mut self,
        right_boundary: usize,
        k_dof_plus: usize, /* k degrees of freedom + 1 */
        max_seg_dof: usize,
        minimizer: &mut M,
    ) -> MinimizationSolution<CutPath, T>
    where
        M: Minimizer<MinimizationSolution<CutPath, T>>,
    {
        let res = self.step_core(right_boundary, k_dof_plus, max_seg_dof, minimizer);
        self.store_energy(k_dof_plus, right_boundary, res.min);
        res
    }

    #[inline(always)]
    pub fn store_energy(&mut self, k_dof_plus_one: usize, right_boundary: usize, min_energy: T) {
        self.energies[[k_dof_plus_one - 1, right_boundary + 1]] = Some(min_energy);
        // self.energies[[usize::from(dofs), right_boundary + 1]] = Some(value);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BellmanTable<T>
where
    T: OrdFloat,
{
    pub(crate) energies: Array2<Option<T>>,
}

impl<F, T> TryFrom<BellmanBuilder<F, T>> for BellmanTable<T>
where
    T: OrdFloat,
{
    type Error = ();
    fn try_from(value: BellmanBuilder<F, T>) -> Result<Self, Self::Error> {
        // TODO: verify that table is complete
        Ok(BellmanTable {
            energies: value.energies,
        })
    }
}

impl<T> BellmanTable<T>
where
    T: OrdFloat,
{
    pub fn data_len(&self) -> usize {
        self.energies.shape()[1]
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MinimizationSolution<ArgMin, Min> {
    arg_min: ArgMin,
    min: Min,
}

impl<T> OptimalJumpData<T>
where
    T: OrdFloat,
{
    /// Construct [OptimalJumpData] - so essentially a piecewise approximation to some data - from non-piecewise training errors
    /// by solving the DP.
    ///
    /// # Arguments
    /// * `timeseries_sample` - the data for which the pcw approximation is to be computed
    /// * `user_params` - some additional user-configurable parameters that can be used to speed up the computation,
    ///     influence the generated model etc.
    /// * `training_error` - a function mapping a `segment_start_idx`, `segment_end_idx` (inclusive) and degree of
    ///     freedom `dof` to the error made by a `dof` degree of freedom model on
    ///     `timeseries_sample[segment_start_idx..=segment_end_idx]`.
    #[inline]
    pub fn from_errors(
        timeseries_sample: &ValidTimeSeriesSample<T>,
        user_params: &MatchedUserParams,
        training_error: impl Fn(usize, usize, DegreeOfFreedom) -> T,
    ) -> Self {
        let data_len = usize::from(timeseries_sample.len());
        let max_total_dof = usize::from(user_params.max_total_dof);
        let max_seg_dof = usize::from(user_params.max_seg_dof);
        // set up array for bellman values / energies
        let mut energies = BellmanBuilder::new(user_params.max_total_dof, data_len, training_error);

        // Note: ndarray's Array types use the "C memory order" (row major) by default which is crucial for our access pattern
        let mut prev_cuts = Array2::from_elem((max_total_dof - 1, data_len), None);

        #[rustfmt::skip]
        let cmp = |
            MinimizationSolution { min: energy_1, .. }: &MinimizationSolution<_,T,>,
            MinimizationSolution { min: energy_2, .. }: &MinimizationSolution<_,T,>| {
                energy_1.cmp(energy_2)
        };

        // This is a somewhat crude lower bound on the needed capacity for the minimizer buffer. Calculating this isn't really
        // necessary but we can easily save some memory here.
        // This is obtained by considering the central two loops in the DP (which use the minimizer) and the maximal values for
        // the respective parameters.
        // let approximate_capacity_bound =
        //     (data_len - 1) * min!(data_len - 1, max_seg_dof, max_total_dof);
        // let mut minimizer = minimizers::ParallelMinimizer::new(approximate_capacity_bound, cmp);
        // let mut minimizer = minimizers::StackMinimizer::new(approximate_capacity_bound, cmp);
        let mut minimizer = minimizers::ImmediateMinimizer::new(cmp);
        // let mut minimizer =
        //     minimizers::VecMinimizer::with_capacity(approximate_capacity_bound, cmp);
        // let mut minimizer = minimizers::PostParallelMinimizer::new(cmp);

        // iteratively solve using bottom-up DP scheme:
        // "for all right segment boundaries"
        for r in 0..data_len - 1 {
            // "for all degrees of freedom on the segment with boundary at r"
            // `r + 2` attains a maximum of `data_len` over all iterations of the outer `r` loop
            let upper_bound_due_to_segment_len = if cfg!(feature = "dofs-sub-one") {
                r + 1
            } else {
                r + 2
            };
            for k_dof_plus_one in 2..=cmp::min(upper_bound_due_to_segment_len, max_total_dof) {
                let MinimizationSolution { arg_min, .. } =
                    energies.step(r, k_dof_plus_one, max_seg_dof, &mut minimizer);
                prev_cuts[[k_dof_plus_one - 2, r + 1]] = Some(arg_min);
            }
        }

        OptimalJumpData {
            energies: BellmanTable::try_from(energies).unwrap(),
            prev_cuts,
            max_seg_dof: user_params.max_seg_dof,
        }
    }

    /// How long the underlying timeseries is.
    #[inline(always)]
    pub fn data_len(&self) -> usize {
        self.energies.data_len()
    }

    /// The optimal "cuts" in the timeseries for a model on the full timeseries. So the
    /// boundaries between the distinct pieces of the piecewise model.
    ///
    /// # Note
    /// Note that this will do some allocations internally;
    /// for hot call-sites prefer [OptimalJumpData::optimal_cuts_with_buf] instead.
    #[inline]
    pub fn optimal_cuts(&self, n_dofs: DegreeOfFreedom) -> Option<OwnedDofPartition> {
        self.optimal_cuts_on_subinterval(n_dofs, self.data_len() - 1)
    }

    /// Finds the optimal sequence of cuts for a piecewise approximation of
    /// `data` using exactly `n_dofs` degrees of freedom.
    ///
    /// This is equivalent to
    ///     `optimal_cuts_on_subinterval_with_buf(..., right_data_index = self.data_len() - 1)`.
    ///
    /// # Args
    /// * `n_dofs` number of dofs for prescribed dof problem.
    /// * `cut_buffer` output buffer for cut indices.
    /// * `dof_buffer` output buffer for dofs assigned to each segment of the partition generated
    ///     by the cuts.
    #[inline]
    pub fn optimal_cuts_with_buf<
        'a,
        B1: AsRef<[MaybeUninit<usize>]> + AsMut<[MaybeUninit<usize>]>,
        B2: AsRef<[MaybeUninit<DegreeOfFreedom>]> + AsMut<[MaybeUninit<DegreeOfFreedom>]>,
    >(
        &self,
        n_dofs: DegreeOfFreedom,
        cut_buffer: &'a mut Stack<usize, B1>,
        dof_buffer: &'a mut Stack<DegreeOfFreedom, B2>,
    ) -> Option<RefDofPartition<'a>> {
        self.optimal_cuts_on_subinterval_with_buf(
            n_dofs,
            self.data_len() - 1,
            cut_buffer,
            dof_buffer,
        )
    }

    /// Finds the optimal sequence of cuts for a piecewise approximation of
    /// `data[0..=right_data_index]` using no more than `n_jumps` jumps.
    pub fn optimal_cuts_on_subinterval(
        &self,
        n_dofs: DegreeOfFreedom,
        right_data_index: usize,
    ) -> Option<OwnedDofPartition> {
        let mut cut_buffer = Stack::with_capacity(right_data_index);
        let mut dof_buffer = Stack::with_capacity(right_data_index + 1);
        self.optimal_cuts_on_subinterval_with_buf(
            n_dofs,
            right_data_index,
            &mut cut_buffer,
            &mut dof_buffer,
        )?;
        Some(DofPartition {
            cut_indices: cut_buffer,
            segment_dofs: dof_buffer,
        })
    }

    /// Finds the optimal sequence of cuts for a piecewise approximation of
    /// `data[0..=right_data_index]` using exactly `n_dofs` degrees of freedom.
    ///
    /// # Args
    /// * `n_dofs` number of dofs for prescribed dof problem.
    /// * `right_data_index` the index of the righmost element considered in the prescribed
    ///     dof calculation.
    /// * `cut_buffer` output buffer for cut indices.
    /// * `dof_buffer` output buffer for dofs assigned to each segment of the partition
    ///     generated by the cuts.
    pub fn optimal_cuts_on_subinterval_with_buf<
        'a,
        B1: AsRef<[MaybeUninit<usize>]> + AsMut<[MaybeUninit<usize>]>,
        B2: AsRef<[MaybeUninit<DegreeOfFreedom>]> + AsMut<[MaybeUninit<DegreeOfFreedom>]>,
    >(
        &self,
        n_dofs: DegreeOfFreedom,
        right_data_index: usize,
        cut_buffer: &'a mut Stack<usize, B1>,
        dof_buffer: &'a mut Stack<DegreeOfFreedom, B2>,
    ) -> Option<RefDofPartition<'a>> {
        if let Some(cut) = self.last_optimal_cut_on_subinterval(n_dofs, right_data_index) {
            cut_buffer.clear();
            dof_buffer.clear();
            // We start writing into the buffer from the very back: that way we always have enough space and
            // don't have to reverse anything after processing.

            // trace backwards through the optimal sequence of jumps starting at the optimal jump for
            // the last considered data point
            cut_buffer.push(cut.cut_idx);
            dof_buffer.push(cut.right_dofs); // TODO: check if this can be unchecked

            let mut remaining_dofs = cut.left_dofs;
            // TODO: might have to tweak the condition for higher polynomial orders
            // Assuming init here is safe since we start out safe by initializing `cut_buf_end` and only ever increase the `cut_count`
            // if we've actually written into the buffer at the locations we're later reading from.
            while remaining_dofs >= dof!(2) && cut_buffer.top() != 0 {
                if let Some(prev_cut_path) =
                    self.prev_cuts[[usize::from(remaining_dofs) - 2, cut_buffer.top()]]
                {
                    // this unwrap can't fail
                    match try_cut_path_to_cut(prev_cut_path, remaining_dofs) {
                        Some(prev_cut) => {
                            cut_buffer.push(prev_cut.cut_idx);
                            dof_buffer.push(prev_cut.right_dofs);
                            remaining_dofs = prev_cut.left_dofs;
                        }
                        None => break,
                    }
                } else {
                    break;
                }
            }
            dof_buffer.push(remaining_dofs);

            Some(DofPartition {
                cut_indices: cut_buffer.filled(),
                segment_dofs: dof_buffer.filled(),
            })
        } else {
            // nothing to do - if there's no cuts we don't have to write out anything
            None
        }
    }

    /// Returns the last `Cut` in the sequence returned by [self.optimal_cuts_on_subinterval]
    /// or `None` if no cut exists.
    pub fn last_optimal_cut_on_subinterval(
        &self,
        n_dofs: DegreeOfFreedom,
        right_data_index: usize,
    ) -> Option<Cut> {
        match n_dofs.into() {
            0 => unreachable!(),
            1 => None, // with one degree of freedom there's always just a single model - so no cuts
            n if n > right_data_index + 1 => {
                panic!("Too many degrees of freedom for the given data interval")
            }
            n => self.prev_cuts[[n - 2, right_data_index]]
                .and_then(|cut_path| try_cut_path_to_cut(cut_path, n_dofs)),
        }
    }
}

// This is essentially an `unbounded::PcwFn<usize, usize>` using borrowed buffers rather than owned ones
// TODO: rewrite using PcwFn by making PcwFn generic over the internal storage (maybe by using GATs and a trait?)
// We've already implemented an abstract interface in the [pcw_fn](https://crates.io/crates/pcw_fn) crate.
/// A "degree of freedom partition": an integer partition related to an ordered partition of the timeseries.
///
/// In the notation of the associated thesis this is a sequence (ν_I)_{I ∈ P} for P ∈ OPart(Timeseries).
#[derive(Debug, PartialEq, Eq)]
pub struct DofPartition<T, S>
where
    T: AsRef<[usize]>,
    S: AsRef<[DegreeOfFreedom]>,
{
    pub cut_indices: T,
    pub segment_dofs: S,
}

/// A [DofPartition] that owns its internal storage.
pub type OwnedDofPartition = DofPartition<
    Stack<usize, Box<[MaybeUninit<usize>]>>,
    Stack<DegreeOfFreedom, Box<[MaybeUninit<DegreeOfFreedom>]>>,
>;

/// A [DofPartition] that doesn't own its internal storage.
pub type RefDofPartition<'a> = DofPartition<&'a [usize], &'a [DegreeOfFreedom]>;

impl<'a> From<&'a OwnedDofPartition> for RefDofPartition<'a> {
    #[inline]
    fn from(value: &'a OwnedDofPartition) -> Self {
        RefDofPartition {
            cut_indices: value.cut_indices.as_ref(),
            segment_dofs: value.segment_dofs.as_ref(),
        }
    }
}

/// A single "cut" of the partition.
#[derive(Debug, PartialEq, Eq)]
pub struct Cut {
    cut_idx: usize,
    left_dofs: DegreeOfFreedom,
    right_dofs: DegreeOfFreedom,
}

/// Try converting a [CutPath] to a [Cut] given some initial number of degrees of freedom.
/// Returns `None` if the cut path indicates that there are no cuts.
#[inline]
fn try_cut_path_to_cut(cut_path: CutPath, n_dofs: DegreeOfFreedom) -> Option<Cut> {
    // A cut index of 0 here means that there is a cut in front of the very first data
    // point - this means there really are no cuts.
    match cut_path {
        CutPath::NoCuts => None,
        CutPath::SomeCuts {
            remaining_dofs,
            elem_after_prev_cut,
        } => {
            Some(Cut {
                cut_idx: elem_after_prev_cut - 1,
                left_dofs: remaining_dofs,
                // subtract one dof for the cut itself
                right_dofs: n_dofs.checked_sub(remaining_dofs).unwrap(),
            })
        }
    }
}

use minimizers::*;
mod minimizers {
    #![allow(dead_code)]

    pub trait MinHandle<T> {
        fn add_to_min(&mut self, val: T);
    }

    /// Provides a context for finding the minimum of a collection of values:
    /// in the context a bunch of values can be sent to a given handle and when the context
    /// is destroyed the minimum of all the sent values is returned.
    /// The minimizer prefers newer value on equality, so it implements the same logic as the snippet
    /// `std::cmp::min_by(new_value, old_value, cmp)`.
    pub trait Minimizer<T> {
        type Handle<'a>: MinHandle<T>;
        fn minimize<'a>(&'a mut self, func: impl for<'b> FnOnce(Self::Handle<'b>)) -> T;
    }

    pub use immediate_minimizer::*;
    #[cfg(feature = "rtrb")]
    pub use parallel_minimizer::*;
    #[allow(unused)]
    pub use post_parallel_minimizer::*;
    #[allow(unused)]
    pub use stack_minimizer::*;
    #[allow(unused)]
    pub use vec_minimizer::*;

    mod stack_minimizer {
        use super::*;
        use crate::stack::{HeapStack, Stack};
        use std::cmp::Ordering;

        pub struct StackMinimizer<T: Copy, F> {
            stack: HeapStack<T>,
            cmp: F,
        }

        impl<T, F> StackMinimizer<T, F>
        where
            T: Copy,
            F: Fn(&T, &T) -> Ordering,
        {
            #[inline]
            pub fn new(stack_size: usize, cmp: F) -> Self {
                Self {
                    stack: Stack::with_capacity(stack_size),
                    cmp,
                }
            }
        }

        impl<T, F> MinHandle<T> for &'_ mut StackMinimizer<T, F>
        where
            T: Copy,
            F: Fn(&T, &T) -> Ordering,
        {
            #[inline(always)]
            fn add_to_min(&mut self, val: T) {
                self.stack.push(val)
            }
        }

        impl<T, F> Minimizer<T> for StackMinimizer<T, F>
        where
            T: Copy + 'static,
            F: Fn(&T, &T) -> Ordering + 'static,
        {
            type Handle<'a> = &'a mut Self;
            #[inline]
            fn minimize<'a>(&'a mut self, func: impl for<'b> FnOnce(Self::Handle<'b>)) -> T {
                self.stack.clear(); // make sure the stack is empty
                func(self);
                self.stack.pop_iter().min_by(&self.cmp).unwrap()
            }
        }
    }

    mod immediate_minimizer {
        use super::*;
        use std::cmp::Ordering;

        pub struct ImmediateMinimizer<T: Copy, F> {
            current_min: Option<T>,
            cmp: F,
        }

        impl<T, F> ImmediateMinimizer<T, F>
        where
            T: Copy,
            F: Fn(&T, &T) -> Ordering,
        {
            #[inline]
            pub fn new(cmp: F) -> Self {
                Self {
                    current_min: None,
                    cmp,
                }
            }
        }

        impl<T, F> MinHandle<T> for &'_ mut ImmediateMinimizer<T, F>
        where
            T: Copy,
            F: Fn(&T, &T) -> Ordering,
        {
            #[inline(always)]
            fn add_to_min(&mut self, new_val: T) {
                self.current_min = match self.current_min {
                    // Note that we prefer the new value since it's the left arg
                    Some(current) => Some(std::cmp::min_by(new_val, current, &self.cmp)),
                    None => Some(new_val),
                };
            }
        }

        impl<T, F> Minimizer<T> for ImmediateMinimizer<T, F>
        where
            T: Copy + 'static,
            F: Fn(&T, &T) -> Ordering + 'static,
        {
            type Handle<'a> = &'a mut Self;
            #[inline]
            fn minimize<'a>(&'a mut self, func: impl for<'b> FnOnce(Self::Handle<'b>)) -> T {
                func(self);
                self.current_min.take().unwrap()
            }
        }
    }

    mod vec_minimizer {
        use super::*;
        use std::cmp::Ordering;

        pub struct VecMinimizer<T: Copy, F> {
            stack: Vec<T>,
            cmp: F,
        }

        impl<T, F> VecMinimizer<T, F>
        where
            T: Copy,
            F: Fn(&T, &T) -> Ordering,
        {
            #[inline]
            pub fn new(cmp: F) -> Self {
                Self {
                    stack: Vec::new(),
                    cmp,
                }
            }

            #[inline]
            pub fn with_capacity(capacity: usize, cmp: F) -> Self {
                Self {
                    stack: Vec::with_capacity(capacity),
                    cmp,
                }
            }
        }

        impl<T, F> MinHandle<T> for &'_ mut VecMinimizer<T, F>
        where
            T: Copy,
            F: Fn(&T, &T) -> Ordering,
        {
            #[inline(always)]
            fn add_to_min(&mut self, val: T) {
                self.stack.push(val)
            }
        }

        impl<T, F> Minimizer<T> for VecMinimizer<T, F>
        where
            T: Copy + 'static,
            F: Fn(&T, &T) -> Ordering + 'static,
        {
            type Handle<'a> = &'a mut Self;
            #[inline]
            fn minimize<'a>(&'a mut self, func: impl for<'b> FnOnce(Self::Handle<'b>)) -> T {
                self.stack.clear(); // make sure the stack is empty
                func(self);
                // we gotta start from the back so we reverse here
                self.stack.drain(..).rev().min_by(&self.cmp).unwrap()
            }
        }
    }

    #[cfg(feature = "rtrb")]
    mod parallel_minimizer {
        use super::{MinHandle, Minimizer};
        use rtrb;
        use std::cmp;
        use std::fmt::Debug;
        use std::sync::mpsc;
        use std::thread;

        #[derive(Debug, Copy, Clone)]
        enum ServerCommand {
            Start,
            Stop,
        }

        struct MinServer<T> {
            vals: rtrb::Consumer<T>,
            out: mpsc::Sender<T>,
            notify_done: mpsc::Receiver<ServerCommand>,
        }

        /// A minimizer that uses another thread (communicating via a lockfree ringbuffer channel) for the actual minimization
        pub struct ParallelMinimizer<T> {
            vals: rtrb::Producer<T>,
            out: mpsc::Receiver<T>,
            notify_done: mpsc::Sender<ServerCommand>,
        }

        pub struct ParMinHandle<'a, T> {
            minimizer: &'a mut ParallelMinimizer<T>,
        }

        impl<'a, T> MinHandle<T> for ParMinHandle<'a, T>
        where
            T: Send + 'static,
        {
            fn add_to_min(&mut self, val: T) {
                self.minimizer.add_to_min(val)
            }
        }

        impl<T> Minimizer<T> for ParallelMinimizer<T>
        where
            T: Send + 'static,
        {
            type Handle<'a> = ParMinHandle<'a, T>;
            fn minimize(&'_ mut self, func: impl FnOnce(ParMinHandle<'_, T>)) -> T {
                self.restart();
                let handle = ParMinHandle { minimizer: self };
                func(handle);
                self.done()
            }
        }

        impl<T> ParallelMinimizer<T>
        where
            T: Send + 'static,
        {
            pub fn new<F>(capacity: usize, cmp: F) -> ParallelMinimizer<T>
            where
                F: Fn(&T, &T) -> std::cmp::Ordering + Send + 'static,
            {
                let (vals_send, vals_recv) = rtrb::RingBuffer::new(capacity);
                let (out_send, out_recv) = mpsc::channel();
                let (notify_done_send, notify_done_recv) = mpsc::channel();
                let _handle = thread::spawn(move || {
                    MinServer {
                        vals: vals_recv,
                        out: out_send,
                        notify_done: notify_done_recv,
                    }
                    .serve(cmp)
                });
                Self {
                    vals: vals_send,
                    out: out_recv,
                    notify_done: notify_done_send,
                }
            }

            fn restart(&mut self) {
                self.notify_done.send(ServerCommand::Start).unwrap();
            }

            fn add_to_min(&mut self, val: T) {
                self.vals.push(val).unwrap()
            }

            fn done(&mut self) -> T {
                self.notify_done.send(ServerCommand::Stop).unwrap();
                self.out.recv().unwrap()
            }
        }

        impl<T> MinServer<T> {
            pub fn serve(mut self, cmp: impl Fn(&T, &T) -> std::cmp::Ordering) {
                let mut current_min = self.blocking_recv();
                'serve: loop {
                    current_min = match self.vals.pop() {
                        // Note that we prefer the new value since it's the left arg
                        Ok(new_val) => cmp::min_by(new_val, current_min, &cmp),
                        Err(_) => current_min,
                    };
                    if let Ok(ServerCommand::Stop) = self.notify_done.recv() {
                        // receive all remaining vals
                        while let Ok(new_val) = self.vals.pop() {
                            // Note that we prefer the new value since it's the left arg
                            current_min = cmp::min_by(new_val, current_min, &cmp);
                        }
                        self.out.send(current_min).unwrap();
                        match self.notify_done.recv() {
                            Ok(ServerCommand::Start) => {
                                current_min = self.blocking_recv();
                            }
                            Ok(ServerCommand::Stop) => {
                                panic!("Tried to stop already stopped server")
                            }
                            Err(_) => {
                                // Partner hung up, we'll shut down
                                break 'serve;
                            }
                        }
                    }
                }
            }

            pub fn blocking_recv(&mut self) -> T {
                loop {
                    // busy wait for first value
                    if let Ok(new_val) = self.vals.pop() {
                        break new_val;
                    }
                    // Indicate to the processor that we're currently
                    // waiting even if it's for a veery short while
                    std::hint::spin_loop();
                }
            }
        }
    }

    #[cfg(feature = "parallel_rayon")]
    mod post_parallel_minimizer {
        use super::{MinHandle, Minimizer};
        use rayon::prelude::{IndexedParallelIterator, ParallelDrainRange, ParallelIterator};
        use rayon::{ThreadPool, ThreadPoolBuilder};

        /// A minimizer that runs minimization in parallel in a thread pool
        pub struct PostParallelMinimizer<T, F> {
            cmp: F,
            vals: Vec<T>,
            pool: ThreadPool,
        }

        pub struct PostParMinHandle<'a, T, F> {
            minimizer: &'a mut PostParallelMinimizer<T, F>,
        }

        impl<T, F> MinHandle<T> for PostParMinHandle<'_, T, F>
        where
            T: Send + 'static,
        {
            fn add_to_min(&mut self, val: T) {
                self.minimizer.vals.push(val)
            }
        }

        impl<T, F> Minimizer<T> for PostParallelMinimizer<T, F>
        where
            T: Send + 'static,
            F: Fn(&T, &T) -> std::cmp::Ordering + Send + Sync + 'static,
        {
            type Handle<'a> = PostParMinHandle<'a, T, F>;
            fn minimize(&'_ mut self, func: impl FnOnce(PostParMinHandle<'_, T, F>)) -> T {
                self.vals.clear();
                let handle: PostParMinHandle<'_, T, F> = PostParMinHandle { minimizer: self };
                func(handle);
                self.pool
                    .install(|| self.vals.par_drain(..).rev().min_by(&self.cmp).unwrap())
            }
        }

        impl<T, F> PostParallelMinimizer<T, F>
        where
            T: Send + 'static,
            F: Fn(&T, &T) -> std::cmp::Ordering + Send + Sync + 'static,
        {
            pub fn new(cmp: F) -> PostParallelMinimizer<T, F> {
                Self {
                    cmp,
                    vals: vec![],
                    pool: ThreadPoolBuilder::new().num_threads(24).build().unwrap(),
                }
            }
        }
    }
}
