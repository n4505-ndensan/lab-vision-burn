// CIFAR-10専用のlibエントリーポイント

extern crate alloc;

mod config;
mod data;
mod model;
mod web_cifar10;

pub use web_cifar10::*;
