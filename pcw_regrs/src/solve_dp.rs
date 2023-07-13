use std::{cmp, mem::MaybeUninit};

use crate::{
    prelude::*,
    stack::{HeapStack, Stack},
};

use ndarray::{s, Array2, ArrayView2};

/// A single "cut" of the partition given by reference data for which cut should come before it.
// TODO: Make this into an enum with either "some cuts" or "no cuts".
// Note that this is essentially a linked list through some indirection.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct CutPath {
    /// How many degrees of freedom may still be expended; note that this might be 0
    pub remaining_dofs: usize,
    /// The index of the data point after the cut
    pub elem_after_prev_cut: usize,
}

pub struct OptimalJumpData {
    /// `energies[[k,n]]` contains the optimal energy when approximating `data[0..=n]` with exactly
    /// `k + 1` degrees of freedom.
    // TODO: optimize by exploiting triangle structure
    pub energies: Array2<Option<OFloat>>,
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
    max_seg_dof: DegreeOfFreedom,
}

/// Solves the core dynamic program
pub fn solve_dp(
    timeseries_sample: &ValidTimeSeriesSample,
    user_params: &MatchedUserParams,
    // Calculates the residual error / training error of a (non-pcw) model fit given a
    // segment_start_idx, segment_stop_idx (into the timeseries sample; inclusive) and
    // the number of degrees of freedom of the approximation.
    training_error: impl Fn(usize, usize, DegreeOfFreedom) -> OFloat,
) -> OptimalJumpData {
    OptimalJumpData::from_approximations(timeseries_sample, user_params, training_error)
}

/// Returns the minimum of three values.
#[inline]
fn min3<T: Ord>(a: T, b: T, c: T) -> T {
    std::cmp::min(a, std::cmp::min(b, c))
}

// TODO: Make Bellman / Energies into a struct and abstract the operations on it out into methods.
// Should also get rid of ugly index shifting.

impl OptimalJumpData {
    /// Construct `OptimalJumpData` from a piecewise approximation of some data by solving the DP.
    pub fn from_approximations(
        timeseries_sample: &ValidTimeSeriesSample,
        user_params: &MatchedUserParams,
        training_error: impl Fn(usize, usize, DegreeOfFreedom) -> OFloat,
    ) -> Self {
        let data_len = usize::from(timeseries_sample.len());
        let max_total_dof = usize::from(user_params.max_total_dof);
        let max_seg_dof = usize::from(user_params.max_seg_dof);
        // set up array for bellman values / energies
        let mut energies = Array2::from_elem((max_total_dof, data_len), None);

        // with 1 dof we have to approximate all of the considered data with a constant
        for (n, energy) in energies.slice_mut(s![0, ..]).iter_mut().enumerate() {
            *energy = Some(training_error(0, n, DegreeOfFreedom::one()));
        }

        let mut prev_cuts = Array2::from_elem((max_total_dof - 1, data_len), None);

        // a stack onto which we push all the partial `ls`, `p_l`s and energies before finding their minimum
        // TODO: verify the actually correct size for this stack. The current is guaranteed to be large enough
        // but it uses a rather bad bound.
        let mut solutions = Stack::with_capacity(data_len * max_total_dof);
        // iteratively solve using bottom-up DP scheme:
        // "for all right segment boundaries"
        for r in 0..data_len - 1 {
            // `r + 2` attains a maximum of `data_len` over all iterations of the outer `r` loop
            for k_dof_plus_one in 2..=cmp::min(r + 2, max_total_dof) {
                let (arg_min, min_energy) = Self::bellman_step(
                    energies.view(),
                    r,
                    k_dof_plus_one,
                    max_seg_dof,
                    &mut solutions,
                    &training_error,
                );

                // offset dof index again for bellman
                energies[[k_dof_plus_one - 1, r + 1]] = Some(min_energy);
                prev_cuts[[k_dof_plus_one - 2, r + 1]] = {
                    let (l_min, p_l_min) = arg_min;
                    Some(CutPath {
                        remaining_dofs: p_l_min,
                        elem_after_prev_cut: (l_min + 1) as usize,
                    })
                };
            }
        }

        OptimalJumpData {
            energies,
            prev_cuts,
            max_seg_dof: user_params.max_seg_dof,
        }
    }

    /// Solves the minimization in the forward step of the bellman equation.
    fn bellman_step(
        bellman: ArrayView2<Option<OFloat>>,
        right_boundary: usize,
        k_degrees_of_freedom_plus_one: usize,
        max_seg_dof: usize,
        solutions: &mut HeapStack<((isize, usize), OFloat)>,
        // Calculates the residual error / training error of a model fit given a segment_start_idx,
        // segment_stop_idx and the number of degrees of freedom of the approximation.
        training_error: impl Fn(usize, usize, DegreeOfFreedom) -> OFloat,
    ) -> ((isize, usize), OFloat) {
        let r = right_boundary;
        let k_dof_plus = k_degrees_of_freedom_plus_one;

        for l in 0..=r {
            let p_r_min = cmp::max(k_dof_plus.saturating_sub(l + 1), 1);
            let p_r_max = min3(
                r + 1 - l,      // p_r can't be more larger than the number of data point in the segment
                max_seg_dof,    // it's also restricted by the maximal dof we allow on any segment
                k_dof_plus - 1, // and of course the total dof count minus 1 since 1 dof has to be spent on p_l
            );
            for p_r in p_r_min..=p_r_max {
                let p_l = k_dof_plus - p_r;
                let d = training_error(l + 1, r + 1, DegreeOfFreedom::try_from(p_r).unwrap());

                // `p_l - 1` to account for offset in bellman array indexing
                if let Some(energy) = bellman[[p_l - 1, l]] {
                    solutions.push_unchecked(((l as isize, p_l), energy + d));
                }
            }
            // handle the case where we approximate the whole segment with a single model: so p_r = k+1, p_l = 0
            if k_dof_plus
                <= std::cmp::min(
                    r + 2,
                    max_seg_dof, // it's also restricted by the maximal dof we allow on any segment
                )
            {
                solutions.push_unchecked((
                    (-1, 0),
                    training_error(0, r + 1, DegreeOfFreedom::try_from(k_dof_plus).unwrap()),
                ));
            }
        }

        let (arg_min, min_energy) = solutions
            .pop_iter()
            .min_by(|(_, energy_1), (_, energy_2)| energy_1.cmp(energy_2))
            .unwrap();
        (arg_min, min_energy)
    }

    /// How long the underlying timeseries is.
    pub fn data_len(&self) -> usize {
        self.energies.shape()[1]
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
            dof_buffer.push(DegreeOfFreedom::try_from(cut.right_dofs).unwrap()); // TODO: check if this can be unchecked

            let mut remaining_dofs = cut.left_dofs;
            // TODO: might have to tweak the condition for higher polynomial orders
            // Assuming init here is safe since we start out safe by initializing `cut_buf_end` and only ever increase the `cut_count`
            // if we've actually written into the buffer at the locations we're later reading from.
            while remaining_dofs >= 2 && cut_buffer.top() != 0 {
                if let Some(prev_cut_path) = self.prev_cuts[[remaining_dofs - 2, cut_buffer.top()]]
                {
                    // this unwrap can't fail
                    match try_cut_path_to_cut(prev_cut_path, remaining_dofs) {
                        Some(prev_cut) => {
                            cut_buffer.push(prev_cut.cut_idx);
                            dof_buffer
                                .push(DegreeOfFreedom::try_from(prev_cut.right_dofs).unwrap());
                            remaining_dofs = prev_cut.left_dofs;
                        }
                        None => break,
                    }
                } else {
                    break;
                }
            }
            dof_buffer.push(DegreeOfFreedom::try_from(remaining_dofs).unwrap());

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
            1 => None, // with one degree of freedom there's always just a single model - so no cuts
            n if n > right_data_index + 1 => {
                panic!("Too many degrees of freedom for the given data interval")
            }
            n => self.prev_cuts[[n - 2, right_data_index]]
                .and_then(|cut_path| try_cut_path_to_cut(cut_path, n)),
        }
    }
}

// This is essentially an `unbounded::PcwFn<usize, usize>` using borrowed buffers rather than owned ones
// TODO: rewrite using PcwFn by making PcwFn generic over the internal storage (maybe by using GATs and a trait?)
// We've already implemented an abstract interface in the [pcw_fn](https://crates.io/crates/pcw_fn) crate.
/// A "degree of freedom partition": an integer partition related to an ordered partition of the timeseries.
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
    left_dofs: usize,
    right_dofs: usize,
}

/// Try converting a [CutPath] to a [Cut] given some initial number of degrees of freedom.
/// Returns `None` if the cut path indicates that there are no cuts.
fn try_cut_path_to_cut(cut_path: CutPath, n_dofs: usize) -> Option<Cut> {
    // A cut index of 0 here means that there is a cut in front of the very first data
    // point - this means there really are no cuts.
    if cut_path.elem_after_prev_cut == 0 {
        assert_eq!(cut_path.remaining_dofs, 0);
        None
    } else {
        Some(Cut {
            cut_idx: cut_path.elem_after_prev_cut - 1,
            left_dofs: cut_path.remaining_dofs,
            // subtract one dof for the cut itself
            right_dofs: n_dofs - cut_path.remaining_dofs,
        })
    }
}
