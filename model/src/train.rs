// src/train.rs
use crate::config::DatasetConfig;
use crate::data::{MnistBatch, MnistBatcher};
use crate::model::{LeNet, CifarNet, ModelTrait};
use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use burn::{
    backend::Autodiff,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{BinBytesRecorder, CompactRecorder, FullPrecisionSettings, Recorder},
};
use burn_wgpu::WgpuDevice;
use std::fs;
type B = burn_wgpu::Wgpu;

pub struct TrainConfig {
    pub dataset_config: DatasetConfig,
    pub epochs: u32,
    pub batch_size: usize,
}

pub fn train(cfg: TrainConfig) -> Result<()> {
    let device = WgpuDevice::default();
    
    // アーティファクトディレクトリを作成
    fs::create_dir_all(&cfg.dataset_config.artifacts.dir)?;

    // データセット & ローダーをデータセットタイプに応じて作成
    match cfg.dataset_config.name.as_str() {
        "mnist" => train_mnist(cfg, device),
        "cifar10" => train_cifar10(cfg, device),
        _ => Err(anyhow::anyhow!("未対応のデータセット: {}", cfg.dataset_config.name))
    }
}

fn train_mnist(cfg: TrainConfig, device: WgpuDevice) -> Result<()> {
    // データセット & ローダー
    let train_ds = MnistDataset::train();
    let test_ds = MnistDataset::test();
    let batcher = MnistBatcher::default();

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(cfg.batch_size)
        .shuffle(42)
        .build(train_ds);

    let test_loader = DataLoaderBuilder::new(batcher)
        .batch_size(cfg.batch_size)
        .build(test_ds);

    // モデル & オプティマイザ（Autodiff バックエンドで）
    type AD = Autodiff<B>;
    let device_ad = device.clone().into();
    let mut model = LeNet::<AD>::new(&device_ad, &cfg.dataset_config);
    let mut optim = AdamConfig::new().init();

    let ce = CrossEntropyLossConfig::new().init(&device_ad);

    for epoch in 1..=cfg.epochs {
        // ===== Train =====
        let mut running_loss = 0.0f32;
        let mut n = 0usize;

        for batch in train_loader.iter() {
            let images = batch.images.to_device(&device_ad).require_grad();
            let targets = batch.targets.to_device(&device_ad);

            let logits = model.forward(images);
            let loss = ce.forward(logits.clone(), targets.clone());

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads::<AD, _>(grads, &model);
            model = optim.step(cfg.dataset_config.training.learning_rate, model, grads_params);
            let loss_value = loss.into_data().to_vec::<f32>().expect("loss value")[0];
            running_loss += loss_value;
            n += 1;
        }

        // ===== Eval =====
        let (acc, count) = evaluate_mnist::<B, AD>(&model, &test_loader, &device);
        println!(
            "epoch {epoch:02} | train_loss {:.4} | test_acc {:.2}% ({count} samples)",
            running_loss / (n.max(1) as f32),
            acc * 100.0
        );
    }

    // 保存
    let base_model: LeNet<B> = model.clone().valid();
    let model_path = cfg.dataset_config.get_model_path();
    let bin_path = cfg.dataset_config.get_model_bin_path();
    
    base_model
        .clone()
        .save_file(&model_path, &CompactRecorder::new())
        .expect("save");
    println!("Saved: {}", model_path);

    let record = base_model.clone().into_record();
    let bytes: Vec<u8> = BinBytesRecorder::<FullPrecisionSettings, Vec<u8>>::default()
        .record(record, ())
        .expect("serialize bin bytes");
    fs::write(&bin_path, &bytes).expect("write model.bin");
    println!("Saved: {} ({} bytes)", bin_path, bytes.len());
    Ok(())
}

fn train_cifar10(_cfg: TrainConfig, _device: WgpuDevice) -> Result<()> {
    // TODO: CIFAR-10データセットの実装が必要
    // 現在はプレースホルダーとして エラーを返す
    Err(anyhow::anyhow!("CIFAR-10データセットはまだ実装されていません。後でカスタムデータローダーを作成します。"))
}

fn evaluate_mnist<Bx, ADx>(
    model_ad: &LeNet<ADx>,
    loader: &std::sync::Arc<dyn burn::data::dataloader::DataLoader<Bx, MnistBatch<Bx>>>,
    device: &Bx::Device,
) -> (f32, usize)
where
    Bx: Backend,
    ADx: AutodiffBackend<InnerBackend = Bx>,
{
    let model_eval: LeNet<Bx> = model_ad.clone().valid();

    let mut correct = 0usize;
    let mut total = 0usize;

    for batch in loader.iter() {
        let logits = model_eval.forward(batch.images.to_device(device));
        let preds = logits.argmax(1).reshape([-1]);
        let eq = preds.equal(batch.targets.to_device(device));
        let batch_size = eq.dims()[0];
        let correct_batch = eq.int().sum().into_data().to_vec::<i32>().expect("sum")[0] as usize;
        correct += correct_batch;
        total += batch_size;
    }
    ((correct as f32) / (total.max(1) as f32), total)
}

#[allow(dead_code)]
fn evaluate_cifar10<Bx, ADx>(
    model_ad: &CifarNet<ADx>,
    loader: &std::sync::Arc<dyn burn::data::dataloader::DataLoader<Bx, MnistBatch<Bx>>>,
    device: &Bx::Device,
) -> (f32, usize)
where
    Bx: Backend,
    ADx: AutodiffBackend<InnerBackend = Bx>,
{
    let model_eval: CifarNet<Bx> = model_ad.clone().valid();

    let mut correct = 0usize;
    let mut total = 0usize;

    for batch in loader.iter() {
        let logits = model_eval.forward(batch.images.to_device(device));
        let preds = logits.argmax(1).reshape([-1]);
        let eq = preds.equal(batch.targets.to_device(device));
        let batch_size = eq.dims()[0];
        let correct_batch = eq.int().sum().into_data().to_vec::<i32>().expect("sum")[0] as usize;
        correct += correct_batch;
        total += batch_size;
    }
    ((correct as f32) / (total.max(1) as f32), total)
}
