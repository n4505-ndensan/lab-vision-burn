#![cfg_attr(all(not(test), target_arch = "wasm32"), no_std)]

extern crate alloc;

#[cfg(not(target_arch = "wasm32"))]
pub mod data; // training / dataset logic (non-wasm)
pub mod model;
#[cfg(not(target_arch = "wasm32"))]
pub mod train; // training entry (non-wasm)

#[cfg(target_arch = "wasm32")]
pub mod web; // wasm entry points (only compile for wasm)

// Re-export commonly used types for web
pub use model::LeNet;
