// no_std を外して serde_json を wasm でも利用可能にする
extern crate alloc;

pub mod config; // 設定ファイル管理 (both wasm and native)

#[cfg(not(target_arch = "wasm32"))]
pub mod data; // training / dataset logic (non-wasm)
pub mod model;
#[cfg(not(target_arch = "wasm32"))]
pub mod train; // training entry (non-wasm)

#[cfg(target_arch = "wasm32")]
pub mod state;
#[cfg(target_arch = "wasm32")]
pub mod web; // wasm entry points (only compile for wasm)

// Re-export commonly used types for web
pub use model::{LeNet, CifarNet, ModelTrait};
pub use config::DatasetConfig;
