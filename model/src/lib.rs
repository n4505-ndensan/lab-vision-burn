// no_std を外して serde_json を wasm でも利用可能にする
extern crate alloc;

pub mod config; // 設定ファイル管理 (both wasm and native)

#[cfg(not(target_arch = "wasm32"))]
pub mod data; // training / dataset logic (non-wasm)
pub mod model;
#[cfg(not(target_arch = "wasm32"))]
pub mod train; // training entry (non-wasm)

// 条件付きでWASMモジュールをinclude
#[cfg(all(target_arch = "wasm32", feature = "mnist-only"))]
pub mod web_mnist;
#[cfg(all(target_arch = "wasm32", feature = "mnist-only"))]
pub use web_mnist::*;

#[cfg(all(target_arch = "wasm32", feature = "cifar10-only"))]
pub mod web_cifar10;
#[cfg(all(target_arch = "wasm32", feature = "cifar10-only"))]
pub use web_cifar10::*;

// デフォルト（既存の統合版）
#[cfg(all(target_arch = "wasm32", not(any(feature = "mnist-only", feature = "cifar10-only"))))]
pub mod state;
#[cfg(all(target_arch = "wasm32", not(any(feature = "mnist-only", feature = "cifar10-only"))))]
pub mod web; // wasm entry points (only compile for wasm)

// Re-export commonly used types for web
pub use model::{LeNet, CifarNet, ModelTrait};
pub use config::DatasetConfig;
