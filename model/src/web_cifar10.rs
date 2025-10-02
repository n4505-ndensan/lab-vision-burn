// CIFAR-10専用のWebエントリーポイント
#![allow(clippy::new_without_default)]

use alloc::string::String;
use js_sys::Array;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::config::DatasetConfig;
use crate::model::{CifarNet, ModelTrait};
// CIFAR-10クラス名の定義（WASM用）
const CIFAR10_CLASSES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::Tensor,
};
use burn_wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};

type Backend = burn_wgpu::Wgpu;

// CIFAR-10モデルのバイナリとコンフィグを埋め込み
static CIFAR10_STATE_ENCODED: &[u8] = include_bytes!("../artifacts/cifar10/model.bin");
static CIFAR10_CONFIG: &str = include_str!("../configs/cifar10.json");

#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

/// CIFAR-10専用の推論クラス
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Cifar10Model {
    model: Option<CifarNet<Backend>>,
    config: DatasetConfig,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Cifar10Model {
    /// コンストラクタ
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        let config: DatasetConfig =
            serde_json::from_str(CIFAR10_CONFIG).expect("CIFAR-10設定の解析に失敗");

        Self {
            model: None,
            config,
        }
    }

    /// モデルをロード
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn load(&mut self) -> Result<(), String> {
        if self.model.is_some() {
            return Ok(());
        }

        init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

        let device = WgpuDevice::default();
        let model: CifarNet<Backend> = CifarNet::new(&device, &self.config);

        let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
            .load(CIFAR10_STATE_ENCODED, &device)
            .map_err(|e| format!("CIFAR-10モデルのロードに失敗: {}", e))?;

        self.model = Some(model.load_record(record));
        Ok(())
    }

    /// 推論実行（確率配列を返す）
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn inference(&mut self, input: &[f32]) -> Result<Array, String> {
        if self.model.is_none() {
            self.load().await?;
        }

        let model = self.model.as_ref().unwrap();
        let device = WgpuDevice::default();

        // 入力テンソルの準備（32x32x3のRGB画像）
        let input = Tensor::<Backend, 1>::from_floats(input, &device)
            .reshape([3, 32, 32])
            .reshape([1usize, 3, 32, 32]);

        // CIFAR-10正規化（[0,1]範囲）
        let input = input / 255.0;

        // 推論実行
        let output: Tensor<Backend, 2> = model.forward(input);
        let output = burn::tensor::activation::softmax(output, 1);
        let output = output.into_data_async().await;

        let array = Array::new();
        for value in output.iter::<f32>() {
            array.push(&value.into());
        }

        Ok(array)
    }

    /// Top-1予測クラスのみ返す
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = "inferenceTop1"))]
    pub async fn inference_top1(&mut self, input: &[f32]) -> Result<u32, String> {
        if self.model.is_none() {
            self.load().await?;
        }

        let model = self.model.as_ref().unwrap();
        let device = WgpuDevice::default();

        let input = Tensor::<Backend, 1>::from_floats(input, &device)
            .reshape([3, 32, 32])
            .reshape([1usize, 3, 32, 32]);
        let input = input / 255.0;

        let output: Tensor<Backend, 2> = model.forward(input);
        let pred = output.argmax(1).into_data_async().await;

        let class_id = pred.iter::<i32>().next().unwrap_or(0) as u32;
        Ok(class_id)
    }

    /// 予測クラス名を返す
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = "getClassName"))]
    pub fn get_class_name(&self, class_id: u32) -> String {
        CIFAR10_CLASSES
            .get(class_id as usize)
            .unwrap_or(&"unknown")
            .to_string()
    }

    /// モデルがロード済みか確認
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = "isLoaded"))]
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// クラス一覧を返す
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = "getClassNames"))]
    pub fn get_class_names(&self) -> Array {
        let array = Array::new();
        for class_name in CIFAR10_CLASSES.iter() {
            array.push(&JsValue::from_str(class_name));
        }
        array
    }
}
