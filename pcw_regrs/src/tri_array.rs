#![allow(dead_code)]
//! Implements efficient 2-dimensional (square) triangular arrays

use std::borrow::{Borrow, BorrowMut};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct UpperTriArray<S> {
    /*
    data represents an array like this as [1,2,3,4,5,6,7,8,9,...]
        ╭ 1 2 3 4 5 ╮
        │   6 7 8 9 │
        │     X X X │
        │       O O │
        ╰         O ╯

    n_rows and n_cols give the number of rows and columns of Xs in
    such an array
        ╭ X X X X X ╮
        │   X X X X │
        │     X X X │
        │       O O │
        ╰         O ╯

    We may want to consider only a subview into this array that we mark
    with S in this array:
        ╭ S S S S X ╮
        │   S S S X │
        │     X X X │
        │       O O │
        ╰         O ╯
    and for this the number of columns containing Ss is called n_cols_view.
    Similarly we have n_rows_view, however this only comes in when bounds-
    checking view indices.
    */
    // data storage in row-major order ommitting elements below the diagonal
    data: S,
    // number of rows
    n_rows: usize,
    // number of columns
    n_cols: usize,
    // the width of the columns actually used for indexing
    // using this allows us to implement subviews without special logic
    n_rows_view: usize,
    // the width of the columns actually used for indexing
    // using this allows us to implement subviews without special logic
    n_cols_view: usize,
}

#[inline(always)]
const fn triangle(n: usize) -> usize {
    (n * (n + 1)) / 2
}

// Turns 2D index into linear index for row-major indexing
// const fn to_linear_idx(row: usize, col: usize, size: usize) -> usize {
//     row * size - sum_to_n(row) + col
// }

#[inline(always)]
const fn partial_triangle(n_rows: usize, n_cols: usize) -> usize {
    /*
    We want to count the Xs in this
        ╭ X X X X X ╮
        │   X X X X │
        │     X X X │
        │       O O │
        ╰         O ╯
    given out numbers of cols and rows of Xs.
    We do so by finding the triangle Xs + Os and
    subtracting the triangle of Os.
    */
    let max_size = triangle(n_cols);
    let empty = triangle(n_cols - n_rows);
    max_size - empty
}

/// Turns 2D index into linear index for row-major indexing
#[inline]
const fn to_linear_idx(_idx @ [row, col]: [usize; 2], n_cols: usize) -> usize {
    /*
    We want to find out how many elements are above I or on the same line and to the left of it.
        ╭ X X X X X ╮
        │   X X X X │
        │     X I X │
        │       O O │
        ╰         O ╯
    The row index tells us exactly how many filled rows with n_cols elems are above it.
    the column index then has to be offset by the row index so that the first element of
    the diagonal "has index 0".
    */
    partial_triangle(row, n_cols) + col - row
}

/// Turns 2D index into a view into linear index for row-major indexing into the base array
#[inline]
const fn view_idx_to_linear_idx(_idx @ [row, col]: [usize; 2], n_cols: usize) -> usize {
    /*

             n_cols
         ╭────────┴────────╮
         n_cols_view
         ╭──────┴─────╮

          S S S S S X X       ╮        ╮
            S S I S X X  ←row ├n_rows  │
              X X X X X       ╯        │
                O O O O               ├n_cols
                  O O O               │
                    O O               │
                      O               ╯
                ↑
               col

    We want index of I in [S, S, S, S, S, X, X, S, S, I, S, X, X, ...]

    The row index tells us exactly how many rows of Ss elems are above I.
    Each of these has to additionally be padded with n_cols - n_cols_view Xs to fill the row.

    The column index then has to be offset by the row index so that the first element of
    the diagonal "has index 0".
    */
    // Using the above reasoning the index has to be
    // partial_triangle(row, n_cols_view) + col - row + row * (n_cols - n_cols_view)
    // via some algebra this simplifies to
    col + n_cols * row - triangle(row)
}

impl<T> UpperTriArray<Vec<T>> {
    /// # Safety
    /// The given [data] vector has to have a len of exactly `partial_triangle(n_rows, n_cols)`.
    pub const unsafe fn from_raw_parts(data: Vec<T>, n_rows: usize, n_cols: usize) -> Self {
        Self {
            data,
            n_rows,
            n_cols,
            n_rows_view: n_rows,
            n_cols_view: n_cols,
        }
    }

    pub fn from_elem(n_rows: usize, n_cols: usize, elem: T) -> Self
    where
        T: Clone,
    {
        if n_rows > n_cols {
            panic!("Upper triangular arrays can't have more rows than columns. There are no elements under the diagonal.");
        } else {
            Self {
                data: vec![elem; partial_triangle(n_rows, n_cols)],
                n_rows,
                n_cols,
                n_rows_view: n_rows,
                n_cols_view: n_cols,
            }
        }
    }

    pub fn from_fn(n_rows: usize, n_cols: usize, mut func: impl FnMut() -> T) -> Self {
        let n = partial_triangle(n_rows, n_cols);
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(func());
        }
        Self {
            data,
            n_rows,
            n_cols,
            n_rows_view: n_rows,
            n_cols_view: n_cols,
        }
    }

    pub fn from_row_major_vec(n_rows: usize, n_cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), partial_triangle(n_rows, n_cols));
        Self {
            data: data,
            n_rows,
            n_cols,
            n_rows_view: n_rows,
            n_cols_view: n_cols,
        }
    }
}

impl<S> UpperTriArray<S> {
    #[inline]
    fn idx_in_bounds(&self, [row, col]: [usize; 2]) -> bool {
        row < self.n_rows_view && col < self.n_cols_view && col >= row
    }
}

/*
impl<S> UpperTriArray<S>
where
    S: Index<usize>,
{
    #[inline]
    pub fn get(&self, idx @ [row, col]: [usize; 2]) -> Option<&S::Output> {
        // index out of bounds of valid region
        if row >= self.n_rows_view || col >= self.n_cols_view || col < row {
            None
        } else {
            Some(&self.data[view_idx_to_linear_idx(idx, self.n_cols)])
        }
    }
}

impl<'a, X, Y> From<&'a UpperTriArray<X>> for UpperTriArray<&Y>
where
    X: Borrow<Y>,
{
    fn from(value: &'a UpperTriArray<X>) -> Self {
        UpperTriArray {
            data: value.data.borrow(),
            n_rows: value.n_rows,
            n_cols: value.n_cols,
            n_rows_view: value.n_rows_view,
            n_cols_view: value.n_cols_view,
        }
    }
}
*/

pub trait Get<Idx> {
    type Output;
    fn get<'a>(&'a self, idx: Idx) -> Option<&'a Self::Output>;
}

impl<'a, T> Get<[usize; 2]> for UpperTriArray<&'a [T]> {
    type Output = T;
    fn get(&self, idx: [usize; 2]) -> Option<&Self::Output> {
        if !self.idx_in_bounds(idx) {
            None
        } else {
            Some(&self.data[view_idx_to_linear_idx(idx, self.n_cols)])
        }
    }
}

impl<'a, T> Get<[usize; 2]> for UpperTriArray<&'a mut [T]> {
    type Output = T;
    fn get(&self, idx: [usize; 2]) -> Option<&Self::Output> {
        if !self.idx_in_bounds(idx) {
            None
        } else {
            Some(&self.data[view_idx_to_linear_idx(idx, self.n_cols)])
        }
    }
}

impl<T> Get<[usize; 2]> for UpperTriArray<Vec<T>> {
    type Output = T;
    fn get(&self, idx: [usize; 2]) -> Option<&Self::Output> {
        if !self.idx_in_bounds(idx) {
            None
        } else {
            Some(&self.data[view_idx_to_linear_idx(idx, self.n_cols)])
        }
    }
}

pub trait GetMut<Idx>: Get<Idx> {
    fn get_mut(&mut self, idx: Idx) -> Option<&mut Self::Output>;
}

impl<'a, T> GetMut<[usize; 2]> for UpperTriArray<&'a mut [T]> {
    #[inline]
    fn get_mut(&mut self, idx: [usize; 2]) -> Option<&mut Self::Output> {
        // index out of bounds of valid region
        if !self.idx_in_bounds(idx) {
            None
        } else {
            Some(&mut self.data[view_idx_to_linear_idx(idx, self.n_cols)])
        }
    }
}

impl<'a, T> GetMut<[usize; 2]> for UpperTriArray<Vec<T>> {
    #[inline]
    fn get_mut(&mut self, idx: [usize; 2]) -> Option<&mut Self::Output> {
        // index out of bounds of valid region
        if !self.idx_in_bounds(idx) {
            None
        } else {
            Some(&mut self.data[view_idx_to_linear_idx(idx, self.n_cols)])
        }
    }
}

impl<S> Index<[usize; 2]> for UpperTriArray<S>
where
    Self: Get<[usize; 2]>,
{
    type Output = <Self as Get<[usize; 2]>>::Output;
    #[inline]
    fn index(&self, idx: [usize; 2]) -> &Self::Output {
        &self.get(idx).expect("Index out of bounds")
    }
}

impl<S> IndexMut<[usize; 2]> for UpperTriArray<S>
where
    Self: GetMut<[usize; 2]>,
{
    #[inline]
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        self.get_mut(idx).expect("Index out of bounds")
    }
}

pub type UpperTriArrayView<'a, T> = UpperTriArray<&'a [T]>;
pub trait UpperTriArrayViewTrait<T> {
    fn view(&self, n_rows_view: usize, n_cols_view: usize) -> UpperTriArray<&[T]>;
}

impl<S, T> UpperTriArrayViewTrait<T> for UpperTriArray<S>
where
    S: Borrow<[T]>,
    [T]: Index<usize, Output = T>,
{
    /// Any array with internal storage that can be borrowed as a slice of
    /// some data can provide views into that data.
    fn view(&self, n_rows_view: usize, n_cols_view: usize) -> UpperTriArray<&[T]> {
        assert!(
            n_rows_view <= self.n_rows && n_cols_view <= self.n_cols,
            "View can't be larger than parent array"
        );
        assert!(
            n_rows_view <= n_cols_view,
            "Trying to create a view into data below the diagonal"
        );
        UpperTriArray {
            data: self.data.borrow(),
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            n_rows_view,
            n_cols_view,
        }
    }
}

pub type UpperTriArrayViewMut<'a, T> = UpperTriArray<&'a mut [T]>;
pub trait UpperTriArrayViewMutTrait<T>: UpperTriArrayViewTrait<T> {
    fn view_mut(&mut self, n_rows_view: usize, n_cols_view: usize) -> UpperTriArray<&mut [T]>;
}

impl<S, T> UpperTriArrayViewMutTrait<T> for UpperTriArray<S>
where
    S: BorrowMut<[T]>,
    [T]: Index<usize>,
    UpperTriArray<S>: UpperTriArrayViewTrait<T>,
{
    /// Any array with internal storage that can be borrowed as a slice of
    /// some data can provide views into that data.
    fn view_mut(&mut self, n_rows_view: usize, n_cols_view: usize) -> UpperTriArray<&mut [T]> {
        assert!(
            n_rows_view <= self.n_rows && n_cols_view <= self.n_cols,
            "View can't be larger than parent array"
        );
        UpperTriArray {
            data: self.data.borrow_mut(),
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            n_rows_view,
            n_cols_view,
        }
    }
}

pub struct RowIter<'a, T> {
    arr: UpperTriArrayView<'a, T>,
    idx: Option<[usize; 2]>,
}

impl<S> UpperTriArray<S> {
    pub fn into_iter_row_major<'a, T>(&'a self) -> RowIter<'a, T>
    where
        Self: UpperTriArrayViewTrait<T>,
    {
        RowIter {
            arr: self.view(self.n_rows_view, self.n_cols_view),
            idx: Some([0, 0]),
        }
    }
}

impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        // Safety:
        // The transmutes here are safe since we can guarantee to never return a value
        // that's been previously returned because col and row always differ on subsequent
        // calls due to the above two lines. Furthermore they can't outlive the original
        // `RowIter`.
        match self.idx {
            None => None,
            Some(idx @ [row, col]) => {
                let to_ret: Option<Self::Item> =
                    Some(unsafe { std::mem::transmute(&self.arr[idx]) });
                // try moving one col to the right to get the next elemenet
                let next_col_idx = [row, col + 1];
                self.idx = if self.arr.idx_in_bounds(next_col_idx) {
                    Some(next_col_idx)
                } else {
                    // try moving onto the diagonal entry on the next line
                    let next_row_idx = [row + 1, row + 1];
                    if self.arr.idx_in_bounds(next_row_idx) {
                        Some(next_row_idx)
                    } else {
                        None
                    }
                };
                to_ret
            }
        }
    }
}

pub struct RowIterMut<'a, T> {
    arr: UpperTriArrayViewMut<'a, T>,
    idx: Option<[usize; 2]>,
}

impl<S> UpperTriArray<S> {
    pub fn into_iter_row_major_mut<'a, T>(&'a mut self) -> RowIterMut<'a, T>
    where
        Self: UpperTriArrayViewMutTrait<T>,
    {
        let idx = if self.n_cols_view == 0 || self.n_rows_view == 0 {
            // if either dimension is 0 the view is empty and we must immediately set the index to 0
            None
        } else {
            Some([0, 0])
        };
        RowIterMut {
            arr: self.view_mut(self.n_rows_view, self.n_cols_view),
            idx,
        }
    }
}

impl<'a, T> Iterator for RowIterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        // Safety:
        // The transmutes here are safe since we can guarantee to never return a value
        // that's been previously returned because col and row always differ on subsequent
        // calls due to the above two lines. Furthermore they can't outlive the original
        // `RowIter`.
        match self.idx {
            None => None,
            Some(idx @ [row, col]) => {
                let to_ret: Option<Self::Item> =
                    Some(unsafe { std::mem::transmute(&mut self.arr[idx]) });
                // try moving one col to the right to get the next elemenet
                let next_col_idx = [row, col + 1];
                self.idx = if self.arr.idx_in_bounds(next_col_idx) {
                    Some(next_col_idx)
                } else {
                    // try moving onto the diagonal entry on the next line
                    let next_row_idx = [row + 1, row + 1];
                    if self.arr.idx_in_bounds(next_row_idx) {
                        Some(next_row_idx)
                    } else {
                        None
                    }
                };
                to_ret
            }
        }
    }
}

pub type OwnedUpperTriArray<T> = UpperTriArray<Vec<T>>;

impl<T> OwnedUpperTriArray<T> {
    /// Construct a UpperTriArray by running a recursive function on increasingly larger sublocks
    /// of the array.
    /// So the element at index [i,j] may depend on all values with indices [i',j'] such that i'<i
    /// and j'<j.
    /// Note that this is essentially a special kind of `scan`
    pub fn from_principal_block_rec(
        n_rows: usize,
        n_cols: usize,
        mut rec_func: impl for<'a> FnMut([usize; 2], UpperTriArrayViewMut<'a, T>) -> T,
    ) -> Self {
        let mut arr: OwnedUpperTriArray<MaybeUninit<T>> = {
            let n = partial_triangle(n_rows, n_cols);
            let mut data: Vec<MaybeUninit<_>> = Vec::with_capacity(n);
            unsafe {
                // Safety: we explicitily set the capacity using the same value n
                // Since we use a Vec of MaybeUninits it's fine that we're assuming
                // the uninitialized values to be initialized.
                data.set_len(n);
                // Safety: n has the size required by from_raw_parts
                UpperTriArray::from_raw_parts(data, n_rows, n_cols)
            }
        };

        for row in 0..n_rows {
            for col in row..n_cols {
                let view = arr.view_mut(row + 1, col);
                // Safety:
                // At this point we know that all rows above the current one are fully init,
                // and all elements in the cols left to the current one are also initialized.
                // Thus the intersection of these two regions is also fully initialized. This
                // is the one we're interested in for the recursion and the one we've just gotten
                // a view for.
                // So we can safely transmute this region into an initialized one
                let view_init: UpperTriArrayViewMut<T> = unsafe { std::mem::transmute(view) };
                arr[[row, col]] = MaybeUninit::new(rec_func([row, col], view_init));
            }
        }
        unsafe { std::mem::transmute(arr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_tri_example() {
        assert_eq!(partial_triangle(3, 5), 12);
        assert_eq!(partial_triangle(1, 4), 4);
    }

    #[test]
    fn row_major_linear_idx() {
        assert_eq!(to_linear_idx([0, 0], 4), 0);
        assert_eq!(to_linear_idx([0, 1], 4), 1);
        assert_eq!(to_linear_idx([0, 2], 4), 2);
        assert_eq!(to_linear_idx([0, 3], 4), 3);
        assert_eq!(to_linear_idx([1, 1], 4), 4);
        assert_eq!(to_linear_idx([1, 2], 4), 5);
        assert_eq!(to_linear_idx([1, 3], 4), 6);
        assert_eq!(to_linear_idx([2, 2], 4), 7);
        assert_eq!(to_linear_idx([2, 3], 4), 8);
        assert_eq!(to_linear_idx([3, 3], 4), 9);
    }

    #[test]
    fn linear_idx_view() {
        /*
        S S S S S X X
          S S S S X X
            S S S X X
              O O O O
                O O O
                  O O
                    O
        */
        assert_eq!(view_idx_to_linear_idx([0, 0], 7), 0);
        assert_eq!(view_idx_to_linear_idx([0, 1], 7), 1);
        assert_eq!(view_idx_to_linear_idx([0, 2], 7), 2);
        assert_eq!(view_idx_to_linear_idx([0, 3], 7), 3);
        assert_eq!(view_idx_to_linear_idx([0, 4], 7), 4);

        assert_eq!(view_idx_to_linear_idx([1, 1], 7), 7);
        assert_eq!(view_idx_to_linear_idx([1, 2], 7), 8);
        assert_eq!(view_idx_to_linear_idx([1, 3], 7), 9);
        assert_eq!(view_idx_to_linear_idx([1, 4], 7), 10);

        assert_eq!(view_idx_to_linear_idx([2, 2], 7), 13);
        assert_eq!(view_idx_to_linear_idx([2, 3], 7), 14);
        assert_eq!(view_idx_to_linear_idx([2, 4], 7), 15);

        assert_eq!(view_idx_to_linear_idx([3, 3], 7), 18);
        assert_eq!(view_idx_to_linear_idx([3, 4], 7), 19);

        assert_eq!(view_idx_to_linear_idx([4, 4], 7), 22);
    }

    #[test]
    fn indexing() {
        let data = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        let array: UpperTriArray<Vec<u32>> = UpperTriArray::from_row_major_vec(4, 4, data);

        assert_eq!(array[[0, 0]], 2);
        assert_eq!(array[[0, 1]], 4);
        assert_eq!(array[[0, 2]], 8);
        assert_eq!(array[[0, 3]], 16);
        assert_eq!(array[[1, 1]], 32);
        assert_eq!(array[[1, 2]], 64);
        assert_eq!(array[[1, 3]], 128);
        assert_eq!(array[[2, 2]], 256);
        assert_eq!(array[[2, 3]], 512);
        assert_eq!(array[[3, 3]], 1024);
    }

    #[test]
    #[should_panic]
    fn view_below_diagonal() {
        let data = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        let array: UpperTriArray<Vec<u32>> = UpperTriArray::from_row_major_vec(4, 4, data);
        array.view(3, 2);
    }

    #[test]
    fn view_iter() {
        let data = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        let array: UpperTriArray<Vec<u32>> = UpperTriArray::from_row_major_vec(4, 4, data);
        let view = array.view(2, 3);

        /*
        We get two rows and 3 cols from this
        2    4    8   16
            32   64  128
                256  512
                    1024
        */
        let mut it = view.into_iter_row_major();

        assert_eq!(it.next(), Some(&2));
        assert_eq!(it.next(), Some(&4));
        assert_eq!(it.next(), Some(&8));
        assert_eq!(it.next(), Some(&32));
        assert_eq!(it.next(), Some(&64));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn from_block_rec_const() {
        let arr = UpperTriArray::from_principal_block_rec(3, 3, |_, _| 0);
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[0, 1]], 0);
        assert_eq!(arr[[0, 2]], 0);
        assert_eq!(arr[[1, 1]], 0);
        assert_eq!(arr[[1, 2]], 0);
        assert_eq!(arr[[2, 2]], 0);
    }

    #[test]
    fn from_block_rec() {
        let arr = UpperTriArray::from_principal_block_rec(3, 3, |idx, view| {
            dbg!(&view);
            match idx {
                [0, 0] => 0,
                [row, col] if row == col => view[[row - 1, col - 1]] + 2,
                [row, col] => view[[row, col - 1]] * 2 + 1,
            }
        });
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[0, 1]], 1);
        assert_eq!(arr[[0, 2]], 3);
        assert_eq!(arr[[1, 1]], 2);
        assert_eq!(arr[[1, 2]], 5);
        assert_eq!(arr[[2, 2]], 4);
    }

    #[test]
    fn from_block_rec2() {
        let arr = UpperTriArray::from_principal_block_rec(
            3,
            3,
            |idx, view: UpperTriArrayViewMut<'_, i32>| match idx {
                [0, 0] => 0,
                [row, col] if row == col => view[[row - 1, col - 1]] + 2,
                [_, _] => {
                    let x = view.into_iter_row_major().max().unwrap();
                    x * 2 + 1
                }
            },
        );
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[0, 1]], 1);
        assert_eq!(arr[[0, 2]], 3);
        assert_eq!(arr[[1, 1]], 2);
        assert_eq!(arr[[1, 2]], 5);
        assert_eq!(arr[[2, 2]], 4);
    }
}
