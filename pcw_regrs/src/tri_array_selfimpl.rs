//! Implements efficient 2-dimensional (square) triangular arrays

use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct UpperTriArray<T> {
    // data storage in row-major order ommitting elements below the diagonal
    data: Vec<T>,
    // number of rows / columns
    size: usize,
}

const fn sum_to_n(n: usize) -> usize {
    (n * (n + 1)) / 2
}

/// Turns 2D index into linear index for row-major indexing
const fn to_linear_idx(row: usize, col: usize, size: usize) -> usize {
    row * size - sum_to_n(row) + col
}

impl<T> UpperTriArray<T> {
    pub fn from_elem(size: usize, elem: T) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![elem; sum_to_n(size)],
            size,
        }
    }

    pub fn from_fn(size: usize, mut func: impl FnMut() -> T) -> Self {
        let n = sum_to_n(size);
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(func());
        }
        Self { data, size }
    }

    pub fn from_row_major_vec(data: Vec<T>, size: usize) -> Self {
        assert_eq!(data.len(), sum_to_n(size));
        Self { data: data, size }
    }

    pub fn get(&self, [row, col]: [usize; 2]) -> Option<&T> {
        // index out of bounds of valid region
        if row >= self.size || col >= self.size || col < row {
            None
        } else {
            Some(&self.data[to_linear_idx(row, col, self.size)])
        }
    }

    pub fn get_mut(&mut self, [row, col]: [usize; 2]) -> Option<&mut T> {
        // index out of bounds of valid region
        if row >= self.size || col >= self.size || col < row {
            None
        } else {
            Some(&mut self.data[to_linear_idx(row, col, self.size)])
        }
    }
}

impl<T> Index<[usize; 2]> for UpperTriArray<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &Self::Output {
        self.get(idx).expect("Index out of bounds")
    }
}

impl<T> IndexMut<[usize; 2]> for UpperTriArray<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        self.get_mut(idx).expect("Index out of bounds")
    }
}

mod view {
    use super::*;

    /// A view into a triangular array extending over a [view_size] ⨉ [view_size] block
    #[derive(Clone, Debug)]
    pub struct UpperTriArrayView<'a, T> {
        /// array into which we're providing a view
        parent: &'a UpperTriArray<T>,
        /// number of rows *of the view*
        view_rows: usize,
        /// number of columns *of the view*
        view_cols: usize,
    }

    impl<T> UpperTriArray<T> {
        pub fn view(&self, view_rows: usize, view_cols: usize) -> UpperTriArrayView<T> {
            UpperTriArrayView {
                parent: self,
                view_rows,
                view_cols,
            }
        }
    }

    impl<'a, T> UpperTriArrayView<'a, T> {
        pub fn get(&self, [row, col]: [usize; 2]) -> Option<&T> {
            if row >= self.view_rows || col >= self.view_cols {
                None
            } else {
                self.parent.get([row, col])
            }
        }
    }

    impl<'a, T> Index<[usize; 2]> for UpperTriArrayView<'a, T> {
        type Output = T;

        fn index(&self, idx: [usize; 2]) -> &Self::Output {
            self.get(idx).expect("Index out of bounds")
        }
    }

    /// A view into a triangular array extending over a [view_size] ⨉ [view_size] block
    #[derive(Debug)]
    pub struct UpperTriArrayViewMut<'a, T> {
        /// array into which we're providing a view
        parent: &'a mut UpperTriArray<T>,
        /// number of rows *of the view*
        view_rows: usize,
        /// number of columns *of the view*
        view_cols: usize,
    }

    impl<T> UpperTriArray<T> {
        pub fn view_mut(&mut self, view_rows: usize, view_cols: usize) -> UpperTriArrayViewMut<T> {
            UpperTriArrayViewMut {
                parent: self,
                view_rows,
                view_cols,
            }
        }
    }

    impl<'a, T> UpperTriArrayViewMut<'a, T> {
        pub fn get(&self, [row, col]: [usize; 2]) -> Option<&T> {
            if row > self.view_rows || col > self.view_cols {
                None
            } else {
                self.parent.get([row, col])
            }
        }

        pub fn get_mut(&mut self, [row, col]: [usize; 2]) -> Option<&mut T> {
            if row >= self.view_rows || col >= self.view_cols {
                None
            } else {
                self.parent.get_mut([row, col])
            }
        }
    }

    impl<'a, T> Index<[usize; 2]> for UpperTriArrayViewMut<'a, T> {
        type Output = T;

        fn index(&self, idx: [usize; 2]) -> &Self::Output {
            self.get(idx).expect("Index out of bounds")
        }
    }

    impl<'a, T> IndexMut<[usize; 2]> for UpperTriArrayViewMut<'a, T> {
        fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
            self.get_mut(idx).expect("Index out of bounds")
        }
    }

    impl<T> UpperTriArray<T> {
        /// Construct a UpperTriArray by running a recursive function on increasingly larger sublocks
        /// of the array.
        /// So the element at index [i,j] may depend on all values with indices [i',j'] such that i'<i
        /// and j'<j.
        /// Note that this is essentially a special kind of `scan`
        pub fn from_block_rec(
            size: usize,
            mut rec_func: impl for<'a> FnMut([usize; 2], UpperTriArrayViewMut<'a, T>) -> T,
        ) -> Self {
            let mut arr: UpperTriArray<MaybeUninit<T>> =
                UpperTriArray::from_fn(size, MaybeUninit::uninit);
            for row in 0..size {
                for col in row..size {
                    let view = arr.view_mut(row, col);
                    arr[[row, col]] = MaybeUninit::new(unsafe {
                        // Safety:
                        // At this point we know that all rows above the current one are fully init,
                        // and all elements in the cols left to the current one are also initialized.
                        // Thus the intersection of these two regions is also fully initialized. This
                        // is the one we're interested in for the recursion and the one we've just gotten
                        // a view for.
                        // So we can safely transmute this region into an initialized one
                        rec_func([row, col], std::mem::transmute(view))
                    });
                }
            }
            unsafe { std::mem::transmute(arr) }
        }
    }

    pub struct ViewIter<'a, T> {
        view: &'a UpperTriArrayView<'a, T>,
        row: usize,
        col: usize,
    }

    impl<'a, T> UpperTriArrayView<'a, T> {
        pub fn iter(&'a self) -> ViewIter<'a, T> {
            ViewIter {
                view: self,
                row: 0,
                col: 0,
            }
        }
    }

    impl<'a, T> Iterator for ViewIter<'a, T> {
        type Item = &'a T;
        fn next(&mut self) -> Option<Self::Item> {
            let last_row_idx = self.view.view_rows - 1;
            let last_col_idx: usize = self.view.view_cols - 1;
            if self.row != last_row_idx {
                if self.col != last_col_idx {
                    self.col += 1;
                } else {
                    self.row += 1;
                    self.col = self.row;
                }
                Some(&self.view[[self.row, self.col]])
            } else {
                None
            }
        }
    }

    pub struct ViewIterMut<'a, T> {
        view: &'a mut UpperTriArrayViewMut<'a, T>,
        row: usize,
        col: usize,
    }

    impl<'a, T> UpperTriArrayViewMut<'a, T> {
        pub fn iter_mut(&'a mut self) -> ViewIterMut<'a, T> {
            ViewIterMut {
                view: self,
                row: 0,
                col: 0,
            }
        }
    }

    impl<'a, T> Iterator for ViewIterMut<'a, T> {
        type Item = &'a mut T;
        fn next(&mut self) -> Option<Self::Item> {
            let last_row_idx = self.view.view_rows - 1;
            let last_col_idx: usize = self.view.view_cols - 1;
            if self.row != last_row_idx {
                if self.col != last_col_idx {
                    self.col += 1;
                } else {
                    self.row += 1;
                    self.col = self.row;
                }
                // Safety:
                // It's safe to transmute here since we can guarantee to never return a value
                // that's been previously returned because col and row always differ on subsequent
                // calls due to the above two lines
                Some(unsafe { std::mem::transmute(&mut self.view[[self.row, self.col]]) })
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_linear_idx() {
        assert_eq!(to_linear_idx(0, 0, 4), 0);
        assert_eq!(to_linear_idx(0, 1, 4), 1);
        assert_eq!(to_linear_idx(0, 2, 4), 2);
        assert_eq!(to_linear_idx(0, 3, 4), 3);

        assert_eq!(to_linear_idx(1, 1, 4), 4);
        assert_eq!(to_linear_idx(1, 2, 4), 5);
        assert_eq!(to_linear_idx(1, 3, 4), 6);

        assert_eq!(to_linear_idx(2, 2, 4), 7);
        assert_eq!(to_linear_idx(2, 3, 4), 8);

        assert_eq!(to_linear_idx(3, 3, 4), 9);
    }

    #[test]
    fn indexing() {
        let data = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        let array: UpperTriArray<u32> = UpperTriArray::from_row_major_vec(data, 4);

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
    fn from_block_rec_const() {
        let arr = UpperTriArray::from_block_rec(3, |_, _| 0);
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[0, 1]], 0);
        assert_eq!(arr[[0, 2]], 0);
        assert_eq!(arr[[1, 1]], 0);
        assert_eq!(arr[[1, 2]], 0);
        assert_eq!(arr[[2, 2]], 0);
    }

    #[test]
    fn from_block_rec() {
        let arr = UpperTriArray::from_block_rec(3, |idx, view| match idx {
            [0, 0] => 0,
            [row, col] if row == col => view[[row - 1, col - 1]] + 2,
            [row, col] => view[[row, col - 1]] * 2 + 1,
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
        let arr = UpperTriArray::from_block_rec(
            3,
            |idx, mut view: view::UpperTriArrayViewMut<'_, i32>| {
                let x = view.iter_mut().map(|&mut x| x).max().unwrap();
                match idx {
                    [0, 0] => 0,
                    [row, col] if row == col => view[[row - 1, col - 1]] + 2,
                    [_, _] => x * 2 + 1,
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
