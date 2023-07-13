//! Provides a simple type to annotate some general data with a piece of metadata to track how
//! data "moves through sufficiently general algorithms".
use derive_new::new;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
    ops::Deref,
};

/// A piece of data annotated with some metadata.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(new, Clone, Copy, Eq, PartialEq, Default)]
pub struct Annotated<D, M> {
    pub data: D,
    pub metadata: M,
}

impl<D, M> Borrow<D> for Annotated<D, M> {
    fn borrow(&self) -> &D {
        &self.data
    }
}

impl<D, M> BorrowMut<D> for Annotated<D, M> {
    fn borrow_mut(&mut self) -> &mut D {
        &mut self.data
    }
}

impl<D, M> Deref for Annotated<D, M> {
    type Target = D;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<D, M> Debug for Annotated<D, M>
where
    D: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}
