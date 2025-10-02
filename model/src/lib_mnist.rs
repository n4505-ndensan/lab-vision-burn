// MNIST専用のlibエントリーポイント

extern crate alloc;

mod config;
mod data;
mod model;
mod web_mnist;

pub use web_mnist::*;