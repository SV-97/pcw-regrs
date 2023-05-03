//! The part of the solution concerned with the optimal partitions and general implementation
//! of the dynamic program.
use crate::approximators::{
    DegreeOfFreedom, ErrorApproximator, PcwApproximator, PcwPolynomialArgs, PolynomialArgs,
};
use crate::stack::{HeapStack, Stack};
use derive_new::new;
use ndarray::{s, Array2, ArrayView2};
use num_traits::real::Real;

use std::cmp;
use std::convert::{AsMut, AsRef};
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;

// TODO: this doesn't currently work for any other value but 1
const MINIMAL_SEGMENT_LENGTH: usize = 1; // should be positive

/// Represents a solution of the key dynamic program.
#[derive(new, Debug, Eq, PartialEq)]
pub struct OptimalJumpData<E> {
    /// `energies[[k,n]]` contains the optimal energy when approximating `data[0..n]` with exactly
    /// `k + 1` degrees of freedom.
    // TODO: optimize by exploiting triangle structure
    pub energies: Array2<Option<E>>,
    /// Index `[[k,r]]` will tell us the index of the element after the previous cut,
    /// and how many degrees of freedom may still be expended on the left segment if
    /// we start with `data[0..=r]` and `k + 2` degrees of freedom.
    // TODO: optimize by exploiting triangle structure
    // TODO: remove first column since it's always full of `None`s
    pub prev_cuts: Array2<Option<CutPath>>,
    /// The maximal number of degrees of freedom to be spent on a single segment
    max_seg_dof: NonZeroUsize,
}

/// A single "cut" of the partition given by reference data for which cut should come before it.
#[derive(new, Copy, Clone, Eq, PartialEq, Debug)]
pub struct CutPath {
    /// How many degrees of freedom may still be expended
    pub remaining_dofs: usize,
    /// The index of the data point after the cut
    pub elem_after_prev_cut: usize,
}

/// A single "cut" of the partition.
#[derive(Debug, PartialEq, Eq)]
pub struct Cut {
    cut_idx: usize,
    left_dofs: usize,
    right_dofs: usize,
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
    S: AsRef<[NonZeroUsize]>,
{
    pub cut_indices: T,
    pub segment_dofs: S,
}

/// A [DofPartition] that owns its internal storage.
pub type OwnedDofPartition = DofPartition<
    Stack<usize, Box<[MaybeUninit<usize>]>>,
    Stack<NonZeroUsize, Box<[MaybeUninit<NonZeroUsize>]>>,
>;

/// A [DofPartition] that doesn't own its internal storage.
pub type RefDofPartition<'a> = DofPartition<&'a [usize], &'a [NonZeroUsize]>;

impl<'a> From<&'a OwnedDofPartition> for RefDofPartition<'a> {
    fn from(value: &'a OwnedDofPartition) -> Self {
        RefDofPartition {
            cut_indices: value.cut_indices.as_ref(),
            segment_dofs: value.segment_dofs.as_ref(),
        }
    }
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

/// Returns the minimum of three elements.
#[inline]
fn min3<T: Ord>(a: T, b: T, c: T) -> T {
    std::cmp::min(a, std::cmp::min(b, c))
}

impl<E> OptimalJumpData<E> {
    // TODO: rename this
    /// Number of dofs required for the error to vanish on all the data
    pub fn max_total_dofs(&self) -> DegreeOfFreedom {
        unsafe { NonZeroUsize::new(self.energies.shape()[0]).unwrap_unchecked() }
    }

    /// The maximal total number of degrees of freedom allowed on a given segment (including potential jumps / cuts).
    pub fn max_total_seg_dofs(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
    ) -> DegreeOfFreedom {
        cmp::min(
            unsafe { NonZeroUsize::new_unchecked(segment_stop_idx + 1 - segment_start_idx) },
            self.max_total_dofs(), // self.max_seg_dof
        )
    }

    /// The maximal number of degrees of freedom allowed on a given segment.
    pub fn max_seg_dofs(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
    ) -> DegreeOfFreedom {
        cmp::min(
            unsafe { NonZeroUsize::new_unchecked(segment_stop_idx + 1 - segment_start_idx) },
            self.max_seg_dof,
        )
    }

    /// How long the underlying timeseries is.
    pub fn data_len(&self) -> usize {
        self.energies.shape()[1]
    }

    /// The optimal cuts for a model on the full timeseries.
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
    pub fn optimal_cuts_with_buf<
        'a,
        B1: AsRef<[MaybeUninit<usize>]> + AsMut<[MaybeUninit<usize>]>,
        B2: AsRef<[MaybeUninit<NonZeroUsize>]> + AsMut<[MaybeUninit<NonZeroUsize>]>,
    >(
        &self,
        n_dofs: DegreeOfFreedom,
        cut_buffer: &'a mut Stack<usize, B1>,
        dof_buffer: &'a mut Stack<NonZeroUsize, B2>,
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
        B2: AsRef<[MaybeUninit<NonZeroUsize>]> + AsMut<[MaybeUninit<NonZeroUsize>]>,
    >(
        &self,
        n_dofs: NonZeroUsize,
        right_data_index: usize,
        cut_buffer: &'a mut Stack<usize, B1>,
        dof_buffer: &'a mut Stack<NonZeroUsize, B2>,
    ) -> Option<RefDofPartition<'a>> {
        if let Some(cut) = self.last_optimal_cut_on_subinterval(n_dofs, right_data_index) {
            cut_buffer.clear();
            dof_buffer.clear();
            // We start writing into the buffer from the very back: that way we always have enough space and
            // don't have to reverse anything after processing.

            // trace backwards through the optimal sequence of jumps starting at the optimal jump for
            // the last considered data point
            cut_buffer.push(cut.cut_idx);
            dof_buffer.push(NonZeroUsize::new(cut.right_dofs).unwrap()); // TODO: check if this can be unchecked

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
                            dof_buffer.push(NonZeroUsize::new(prev_cut.right_dofs).unwrap());
                            remaining_dofs = prev_cut.left_dofs;
                        }
                        None => break,
                    }
                } else {
                    break;
                }
            }
            dof_buffer.push(NonZeroUsize::new(remaining_dofs).unwrap());

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

    /// Solves the minimization in the forward step of the bellman equation.
    fn bellman_step<T, D, S, A>(
        approx: &A,
        bellman: ArrayView2<Option<E>>,
        right_boundary: usize,
        k_degrees_of_freedom_plus_one: usize,
        max_seg_dof: usize,
        solutions: &mut HeapStack<((isize, usize), E)>,
    ) -> ((isize, usize), E)
    where
        S: ErrorApproximator<T, D, E, Model = PolynomialArgs<T>>,
        A: Sync + PcwApproximator<S, T, D, E, Model = PcwPolynomialArgs<T>>,
        E: Send + Sync + Real + Ord,
    {
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
                let d = approx.training_error(
                    l + 1,
                    r + 1,
                    PolynomialArgs::from(unsafe { NonZeroUsize::new_unchecked(p_r) }),
                    |_, _| unreachable!(),
                );

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
                    approx.training_error(
                        0,
                        r + 1,
                        PolynomialArgs::from(unsafe { NonZeroUsize::new_unchecked(k_dof_plus) }),
                        |_, _| unreachable!(),
                    ),
                ));
            }
        }

        let (arg_min, min_energy) = solutions
            .pop_iter()
            .min_by(|(_, e1), (_, e2)| e1.cmp(e2))
            .unwrap();
        (arg_min, min_energy)
    }

    /// Solves the minimization in the forward step of the bellman equation similar to `bellman_step`
    /// with the core logic written using iterators.
    #[allow(unused)]
    fn bellman_step_iter<T, D, S, A>(
        approx: &A,
        bellman: ArrayView2<Option<E>>,
        right_boundary: usize,
        k_degrees_of_freedom_plus_one: usize,
        max_seg_dof: usize,
        _solutions: &mut HeapStack<((isize, usize), E)>, // not needed
    ) -> ((isize, usize), E)
    where
        S: ErrorApproximator<T, D, E, Model = DegreeOfFreedom>,
        A: Sync + PcwApproximator<S, T, D, E, Model = Option<DegreeOfFreedom>>,
        E: Send + Sync + Real + Ord,
    {
        let r = right_boundary;
        let k_dof_plus = k_degrees_of_freedom_plus_one;

        let mut min1 = (0..=r)
            //.into_par_iter()
            .flat_map(|l: usize| {
                let p_r_min = cmp::max(k_dof_plus.saturating_sub(l + 1), 1);
                let p_r_max = min3(
                    r + 1 - l,      // p_r can't be more larger than the number of data point in the segment
                    max_seg_dof, // it's also restricted by the maximal dof we allow on any segment
                    k_dof_plus - 1, // and of course the total dof count minus 1 since 1 dof has to be spent on p_l
                );
                std::iter::repeat(l).zip(p_r_min..p_r_max + 1)
            })
            .map(|(l, p_r)| {
                let p_l = k_dof_plus - p_r;
                debug_assert_ne!(p_l, 0); // >= 0
                debug_assert_ne!(p_r, 0); // >= 0
                debug_assert_ne!(p_l, k_dof_plus); // < k + 1
                debug_assert_ne!(p_r, k_dof_plus); // < k + 1

                let d = approx.training_error(
                    l + 1,
                    r + 1,
                    unsafe { NonZeroUsize::new_unchecked(p_r) },
                    |_, _| unreachable!(),
                );

                // `p_l - 1` to account for offset in bellman array indexing
                let energy = bellman[[p_l - 1, l]].unwrap() + d;
                ((l as isize, p_l), energy)
            })
            .min_by(|(_, e1), (_, e2)| e1.cmp(e2))
            .unwrap();
        // handle the case where we approximate the whole segment with a single model: so p_r = k+1, p_l = 0
        if k_dof_plus
            <= std::cmp::min(
                r + 2,
                max_seg_dof, // it's also restricted by the maximal dof we allow on any segment
            )
        {
            min1 = cmp::min_by(
                min1,
                (
                    (-1, 0),
                    approx.training_error(
                        0,
                        r + 1,
                        unsafe { NonZeroUsize::new_unchecked(k_dof_plus) },
                        |_, _| unreachable!(),
                    ),
                ),
                |(_, e1), (_, e2)| e1.cmp(e2),
            );
        }

        let (arg_min, min_energy) = min1;
        (arg_min, min_energy)
    }

    /// Construct `OptimalJumpData` from a piecewise approximation of some data by solving the DP.
    pub fn from_approximations<T, D, S, A>(
        max_total_dof: Option<DegreeOfFreedom>,
        approx: &A,
    ) -> Self
    where
        S: ErrorApproximator<T, D, E, Model = PolynomialArgs<T>>,
        A: Sync + PcwApproximator<S, T, D, E, Model = PcwPolynomialArgs<T>>,
        E: Send + Sync + Real + Ord,
    {
        let data_len = approx.data_len();
        let max_total_dof = if let Some(max_dof) = max_total_dof {
            std::cmp::min(data_len, usize::from(max_dof))
        } else {
            data_len
        };
        let max_seg_dof = if let Some(max_dof) = approx.model().max_seg_dof {
            std::cmp::min(max_total_dof, usize::from(max_dof))
        } else {
            max_total_dof
        };
        // `bellman[[k,r]]` contains the optimal energy when approximating `data[0..=r]` with exactly
        // `k + 1` degrees of freedom / a `k` degree polynomial
        let mut bellman = Array2::from_elem((max_total_dof, data_len), None);
        // with 1 dof we have to approximate all of the considered data with a constant
        for (n, energy) in bellman.slice_mut(s![0, ..]).iter_mut().enumerate() {
            if n + 1 < MINIMAL_SEGMENT_LENGTH {
                *energy = None;
            } else {
                *energy = Some(approx.training_error(
                    0,
                    n,
                    PolynomialArgs::from(unsafe { NonZeroUsize::new_unchecked(1) }),
                    |_, _| unreachable!(),
                ));
            }
        }

        // Index `[[k,r]]` will tell us the index of the element after the previous cut,
        // and how many degrees of freedom may still be expended on the left segment if
        // we start with `data[0..=r]` and `k + 1` degrees of freedom.
        // A prev value of 0 means there's a jump in front of the first element - which is
        // really no jump at all.
        let mut prev_cuts = Array2::from_elem((max_total_dof - 1, data_len), None);

        // a stack onto which we push all the partial `ls`, `p_l`s and energies before finding their minimum
        // TODO: verify the actually correct size
        let mut solutions = Stack::with_capacity(data_len * max_total_dof);
        // iteratively solve using bottom-up DP scheme
        for r in 0..data_len - 1 {
            for k_dof_plus in std::cmp::min(2, MINIMAL_SEGMENT_LENGTH + 1)
                ..std::cmp::max(2, MINIMAL_SEGMENT_LENGTH + 1)
            {
                bellman[[k_dof_plus - 1, r + 1]] = None;
                prev_cuts[[k_dof_plus - 2, r + 1]] = None;
            }
            // `r + 2` attains a maximum of `data_len` over all iterations of the outer `r` loop
            for k_dof_plus in
                std::cmp::max(2, MINIMAL_SEGMENT_LENGTH + 1)..=cmp::min(r + 2, max_total_dof)
            {
                let (arg_min, min_energy) = Self::bellman_step(
                    approx,
                    bellman.view(),
                    r,
                    k_dof_plus,
                    max_seg_dof,
                    &mut solutions,
                );
                // offset dof index again for bellman
                bellman[[k_dof_plus - 1, r + 1]] = Some(min_energy);
                prev_cuts[[k_dof_plus - 2, r + 1]] = {
                    let (l_min, p_l_min) = arg_min;
                    Some(CutPath {
                        remaining_dofs: p_l_min,
                        elem_after_prev_cut: (l_min + 1) as usize,
                    })
                };
            }
        }

        OptimalJumpData::new(bellman, prev_cuts, NonZeroUsize::new(max_seg_dof).unwrap())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::approximators::{PcwPolynomialApproximator, PcwPolynomialArgs, TimeSeries};
    use ordered_float::OrderedFloat;

    fn fit(
        raw_data: Vec<f64>,
        max_seg_dof: Option<DegreeOfFreedom>,
        max_total_dof: Option<DegreeOfFreedom>,
    ) -> OptimalJumpData<OrderedFloat<f64>> {
        use itertools::Itertools;

        let data: Vec<_> = raw_data.clone().into_iter().map(OrderedFloat).collect();
        let metric = |_: &OrderedFloat<f64>, _: &OrderedFloat<f64>| -> OrderedFloat<f64> {
            panic!("Called metric that shouldn't be called") // TODO: use unreachable?
        };
        let times = (0..data.len())
            .into_iter()
            .map(|x| x as f64)
            .map(OrderedFloat)
            .collect_vec();
        let approx = PcwPolynomialApproximator::fit_metric_data_from_model(
            PcwPolynomialArgs {
                max_seg_dof,
                weights: None,
            },
            metric,
            TimeSeries::new(times, data),
        );
        OptimalJumpData::from_approximations(max_total_dof, &approx)
    }

    #[test]
    fn optimal_jump_data() {
        let raw_data = vec![8., 9., 10., 1., 4., 9., 16.];
        let opt = fit(raw_data.clone(), None, None);
        let n = |v: Vec<usize>| -> Vec<NonZeroUsize> {
            v.into_iter()
                .map(NonZeroUsize::new)
                .map(Option::unwrap)
                .collect()
        };
        println!("{:?}", opt.energies);
        assert_eq!(
            RefDofPartition {
                cut_indices: &vec![2],
                segment_dofs: &n(vec![2, 3]),
            },
            RefDofPartition::from(&opt.optimal_cuts(NonZeroUsize::new(5).unwrap()).unwrap())
        );
        assert_eq!(
            RefDofPartition {
                cut_indices: &vec![2],
                segment_dofs: &n(vec![1, 3]),
            },
            RefDofPartition::from(&opt.optimal_cuts(NonZeroUsize::new(4).unwrap()).unwrap())
        );
        assert_eq!(
            RefDofPartition {
                cut_indices: &vec![2],
                segment_dofs: &n(vec![1, 2]),
            },
            RefDofPartition::from(&opt.optimal_cuts(NonZeroUsize::new(3).unwrap()).unwrap())
        );
    }
}
