#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Burn distributed computing.

extern crate alloc;

mod backend;
mod tensor;

pub mod sharding;

pub use backend::*;
pub use tensor::*;
