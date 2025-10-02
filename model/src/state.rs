use crate::config::DatasetConfig;
use crate::model::{LeNet, CifarNet, ModelTrait};
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use burn_wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};
type Backend = burn_wgpu::Wgpu;

// MNISTモデルのバイナリ
static MNIST_STATE_ENCODED: &[u8] = include_bytes!("../artifacts/mnist/model.bin");

// MNIST用の設定（コンパイル時に埋め込み）
static MNIST_CONFIG: &str = include_str!("../configs/mnist.json");

// TODO: CIFAR-10が実装された後に追加
// static CIFAR10_STATE_ENCODED: &[u8] = include_bytes!("../artifacts/cifar10/model.bin");
// static CIFAR10_CONFIG: &str = include_str!("../configs/cifar10.json");

pub enum ModelInstance {
    Mnist(LeNet<Backend>),
    Cifar10(CifarNet<Backend>),
}

impl ModelInstance {
    pub fn forward(&self, x: burn::prelude::Tensor<Backend, 4>) -> burn::prelude::Tensor<Backend, 2> {
        match self {
            ModelInstance::Mnist(model) => model.forward(x),
            ModelInstance::Cifar10(model) => model.forward(x),
        }
    }
}

/// 指定されたデータセットの学習済みモデルを構築・ロード
pub async fn build_and_load_model(dataset_name: &str) -> Result<(ModelInstance, DatasetConfig), String> {
    init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

    match dataset_name {
        "mnist" => {
            let config: DatasetConfig = serde_json::from_str(MNIST_CONFIG)
                .map_err(|e| format!("MNIST設定の解析に失敗: {}", e))?;
                
            let model: LeNet<Backend> = LeNet::new(&Default::default(), &config);
            let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
                .load(MNIST_STATE_ENCODED, &Default::default())
                .map_err(|e| format!("MNISTモデルのロードに失敗: {}", e))?;

            let model = model.load_record(record);
            Ok((ModelInstance::Mnist(model), config))
        },
        "cifar10" => {
            // TODO: CIFAR-10の実装
            Err("CIFAR-10はまだ実装されていません".to_string())
        },
        _ => Err(format!("未対応のデータセット: {}", dataset_name))
    }
}

/// レガシーサポート: MNISTモデルのみ返す（既存のweb.rsとの互換性）
pub async fn build_and_load_model_legacy() -> LeNet<Backend> {
    let (model, _) = build_and_load_model("mnist").await
        .expect("MNISTモデルのロードに失敗");
    
    match model {
        ModelInstance::Mnist(mnist_model) => mnist_model,
        _ => panic!("予期しないモデルタイプ")
    }
}
