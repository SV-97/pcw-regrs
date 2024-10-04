//! Implements a simple stack with FIFO push and pop and FILO slice access that can use
//! non-owned buffers for internal storage.
//!
//! We use this to very efficiently add items onto a stack in such a way that they
//! can later be accessed in reverse order as contiguous slices of memory.
use std::{iter::FusedIterator, marker::PhantomData, mem::MaybeUninit};

/// A stack allocated on the heap.
pub type HeapStack<T> = Stack<T, Box<[MaybeUninit<T>]>>;

// TODO: the stack itself actually works fine for non-copy types - we just gotta initialize
// the buffers another way. Fix this.
/// Stack with FIFO push and pop and FILO slice access.
pub struct Stack<T, B>
where
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
    T: Copy,
{
    buffer: B,
    push_count: usize,
    phantom: PhantomData<T>,
}

impl<T, B> Stack<T, B>
where
    T: Copy,
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
{
    /// Split the stack into two of approximately equal size. This empties the current stack.
    /// Former TOS is now on the right return value.
    pub fn split(&mut self) -> [Stack<T, &mut [MaybeUninit<T>]>; 2] {
        let buffer_r_len = self.push_count / 2;
        println!(
            "splitting {} into {} and {}",
            self.push_count,
            buffer_r_len,
            self.push_count - buffer_r_len
        );
        // We split off the right buffer *from the back*
        // so buffer = [..., buffer_l, buffer_r]
        //                           ^
        //                         split
        let split_loc = self.buffer.as_ref().len() - buffer_r_len;
        let (buffer_l, buffer_r) = self.buffer.as_mut().split_at_mut(split_loc);
        let stack_l: Stack<T, &mut [MaybeUninit<T>]> = Stack {
            buffer: buffer_l,
            push_count: self.push_count - buffer_r_len,
            phantom: PhantomData,
        };
        let stack_r: Stack<T, &mut [MaybeUninit<T>]> = Stack {
            buffer: buffer_r,
            push_count: buffer_r_len,
            phantom: PhantomData,
        };
        self.push_count = 0;
        [stack_l, stack_r]
    }
}

impl<T: Copy, const N: usize> Stack<T, [MaybeUninit<T>; N]> {
    /// Construct a new stack on the stack
    pub const fn new_on_stack() -> Self {
        Stack {
            buffer: [MaybeUninit::uninit(); N],
            push_count: 0,
            phantom: PhantomData,
        }
    }
}

impl<T: Copy> Stack<T, Box<[MaybeUninit<T>]>> {
    /// Construct a new stack on the heap
    pub fn with_capacity(stack_size: usize) -> Self {
        Stack {
            buffer: vec![MaybeUninit::zeroed(); stack_size].into_boxed_slice(),
            push_count: 0,
            phantom: PhantomData,
        }
    }
}

impl<T, B> Stack<T, B>
where
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
    T: Copy,
{
    /// Construct a new stack using some given buffer
    #[inline]
    pub fn new(buffer: B) -> Self {
        Stack {
            buffer,
            push_count: 0,
            phantom: PhantomData,
        }
    }

    #[inline]
    fn buffer(&self) -> &[MaybeUninit<T>] {
        self.buffer.as_ref()
    }

    #[inline]
    fn buffer_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.buffer.as_mut()
    }

    #[inline]
    fn first_filled_idx(&self) -> usize {
        self.buffer().len() - self.push_count
    }

    #[inline]
    fn filled_raw(&self) -> &[MaybeUninit<T>] {
        let i = self.first_filled_idx();
        &self.buffer()[i..]
    }

    #[inline]
    fn filled_raw_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let i = self.first_filled_idx();
        &mut self.buffer_mut()[i..]
    }

    /// Iterate over values on the stack in an owned way by popping
    #[inline]
    pub fn pop_iter(&mut self) -> StackIter<T, B> {
        StackIter { stack: self }
    }

    /// Returns slice containing all the pushed values in reverse order than the one in which they were pushed
    #[inline]
    pub fn filled(&self) -> &[T] {
        // since `MaybeUninit<T>` is guaranteed to have the same layout as `T` and we only transmute
        // the buffer segment we've actually written into, this is safe.
        unsafe { std::mem::transmute(self.filled_raw()) }
    }

    #[inline]
    pub fn filled_mut(&mut self) -> &mut [T] {
        unsafe { std::mem::transmute(self.filled_raw_mut()) }
    }

    /// clear the whole stack
    #[inline]
    pub fn clear(&mut self) {
        self.drop_inits();
        self.push_count = 0;
    }

    #[inline]
    pub fn push_unchecked(&mut self, val: T) {
        // If the buffer is too small the subtraction here will panic and as such the new value is not
        // written, the push_count not increased and we correctly drop everything on unwind
        let i = self.buffer().len();
        let count = self.push_count;
        self.buffer_mut()[i - (count + 1)] = MaybeUninit::new(val);
        self.push_count += 1;
    }

    #[inline]
    pub fn try_push(&mut self, val: T) -> Result<(), ()> {
        if self.push_count + 1 > self.buffer().len() {
            Err(())
        } else {
            self.push_unchecked(val);
            Ok(())
        }
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        self.try_push(val)
            .expect("Failed push - stack is too small")
    }

    /// Remove the most recently pushed element
    pub fn pop(&mut self) -> T {
        let i = self.first_filled_idx();
        let val = unsafe {
            std::mem::replace(&mut self.buffer_mut()[i], MaybeUninit::uninit()).assume_init()
        };
        self.push_count -= 1;
        val
    }

    /// Remove the most recently pushed element; return `None` if the stack is empty
    pub fn try_pop(&mut self) -> Option<T> {
        if self.push_count != 0 {
            Some(self.pop())
        } else {
            None
        }
    }

    /// Returns a reference to the most recently pushed element
    #[inline]
    pub fn peek_top(&self) -> &T {
        &self.filled()[0]
    }

    /// Returns a copy of the most recently pushed element
    #[inline]
    pub fn top(&self) -> T {
        self.filled()[0]
    }

    fn drop_inits(&mut self) {
        for x in self.filled_raw_mut() {
            // we assume them to be init to drop it
            unsafe { x.assume_init_drop() };
        }
    }
}

impl<T, B> Drop for Stack<T, B>
where
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
    T: Copy,
{
    fn drop(&mut self) {
        // we gotta drop all the xs that have been pushed
        self.drop_inits()
    }
}

impl<T: Copy, B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>> AsRef<[T]> for Stack<T, B> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.filled()
    }
}

impl<T: Copy, B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>> AsMut<[T]> for Stack<T, B> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.filled_mut()
    }
}

/// Iterate over the items of the stack in FILO order.
pub struct StackIter<'a, T, B>
where
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
    T: Copy,
{
    stack: &'a mut Stack<T, B>,
}

impl<T, B> Iterator for StackIter<'_, T, B>
where
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
    T: Copy,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.stack.try_pop()
    }
}

impl<T, B> ExactSizeIterator for StackIter<'_, T, B>
where
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
    T: Copy,
{
    fn len(&self) -> usize {
        self.stack.push_count
    }
}

impl<T, B> FusedIterator for StackIter<'_, T, B>
where
    B: AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]>,
    T: Copy,
{
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::Stack;

    #[test]
    fn new() {
        let s: Stack<usize, _> = Stack::with_capacity(10);
        assert_eq!(s.buffer.len(), 10);
        assert_eq!(s.push_count, 0);
    }

    #[test]
    fn push_pop() {
        let mut s = Stack::with_capacity(10);
        s.push(2);
        s.push(4);
        s.push(8);
        s.push(16);
        s.push(32);
        s.push(64);
        assert_eq!(s.pop(), 64);
        assert_eq!(s.pop(), 32);
        assert_eq!(s.pop(), 16);
        assert_eq!(s.pop(), 8);
        assert_eq!(s.pop(), 4);
        assert_eq!(s.pop(), 2);
    }

    #[test]
    #[should_panic]
    fn push_full() {
        let mut s = Stack::with_capacity(1);
        s.push(2);
        s.push(4);
    }

    #[test]
    #[should_panic]
    fn push_empty() {
        let mut s = Stack::with_capacity(0);
        s.push(2);
    }

    #[test]
    #[should_panic]
    fn pop_empty() {
        let mut s: Stack<usize, _> = Stack::with_capacity(0);
        s.pop();
    }

    #[test]
    #[should_panic]
    fn pop_empty_2() {
        let mut s: Stack<usize, _> = Stack::with_capacity(10);
        s.pop();
    }

    #[test]
    fn filled_and_clear() {
        fn fill<'a, B: AsRef<[MaybeUninit<usize>]> + AsMut<[MaybeUninit<usize>]>>(
            stack: &'a mut Stack<usize, B>,
        ) -> &'a mut [usize] {
            for i in 0..4 {
                stack.push(2 * i + 1);
            }
            stack.filled_mut()
        }
        let mut s = Stack::<_, [_; 20]>::new_on_stack();
        assert_eq!(fill(&mut s), &[7, 5, 3, 1]);
        assert_eq!(fill(&mut s), &[7, 5, 3, 1, 7, 5, 3, 1]);
        s.clear();
        let empty: &[usize] = &[];
        assert_eq!(s.filled(), empty);
        assert_eq!(fill(&mut s), &[7, 5, 3, 1]);
    }

    #[test]
    fn double_clear() {
        let mut s = Stack::with_capacity(2);
        s.push(2);
        s.push(4);
        s.clear();
        s.clear();
        s.push(5);
        assert_eq!(s.pop(), 5);
    }

    #[test]
    fn split_even() {
        let mut s = Stack::<_, [_; 20]>::new_on_stack();
        for i in 0..8 {
            s.push(2 * i + 1);
        }
        {
            let [s1, s2] = s.split();
            assert_eq!(s1.filled(), &[15, 13, 11, 9,]);
            assert_eq!(s2.filled(), &[7, 5, 3, 1,]);
        }
        assert_eq!(s.filled(), &[]);
    }

    #[test]
    fn split_odd() {
        let mut s = Stack::<_, [_; 20]>::new_on_stack();
        for i in 0..9 {
            s.push(2 * i + 1);
        }
        {
            let [s1, s2] = s.split();
            assert_eq!(s1.filled(), &[17, 15, 13, 11, 9,]);
            assert_eq!(s2.filled(), &[7, 5, 3, 1,]);
        }
        assert_eq!(s.filled(), &[]);
    }
}
