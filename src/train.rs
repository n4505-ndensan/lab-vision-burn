// src/train.rs
use crate::data::{MnistBatch, MnistBatcher};
use crate::model::LeNet;
use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use burn::{
    backend::Autodiff,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
};
use burn_wgpu::WgpuDevice;
type B = burn_wgpu::Wgpu;

pub struct TrainConfig {
    pub epochs: u32,
    pub batch_size: usize,
}

pub fn train(cfg: TrainConfig) -> Result<()> {
    let device = WgpuDevice::default();

    // データセット & ローダー
    let train_ds = MnistDataset::train();
    let test_ds = MnistDataset::test();
    let batcher = MnistBatcher::default();

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(cfg.batch_size)
        .shuffle(42) // burn 0.18: shuffle は seed (u64) を受け取る
        .build(train_ds);

    let test_loader = DataLoaderBuilder::new(batcher)
        .batch_size(cfg.batch_size)
        .build(test_ds);

    // モデル & オプティマイザ（Autodiff バックエンドで）
    type AD = Autodiff<B>;
    let device_ad = device.clone().into();
    let mut model = LeNet::<AD>::new(&device_ad);
    let mut optim = AdamConfig::new().init(); // 0.18 Optimizer 初期化 (モデルは update 時に参照)

    let ce = CrossEntropyLossConfig::new().init(&device_ad); // 平均化デフォルト

    for epoch in 1..=cfg.epochs {
        // ===== Train =====
        let mut running_loss = 0.0f32;
        let mut n = 0usize;

        for batch in train_loader.iter() {
            // Autodiff バックエンドのデバイスへ転送
            let images = batch.images.to_device(&device_ad).require_grad();
            let targets = batch.targets.to_device(&device_ad);

            let logits = model.forward(images);
            let loss = ce.forward(logits.clone(), targets.clone());

            // 逆伝播 & 更新
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads::<AD, _>(grads, &model);
            model = optim.step(1e-3, model, grads_params); // 学習率 1e-3
            let loss_value = loss.into_data().to_vec::<f32>().expect("loss value")[0];
            running_loss += loss_value;
            n += 1;
        }

        // ===== Eval =====
        let (acc, count) = evaluate::<B, AD>(&model, &test_loader, &device);
        println!(
            "epoch {epoch:02} | train_loss {:.4} | test_acc {:.2}% ({count} samples)",
            running_loss / (n.max(1) as f32),
            acc * 100.0
        );
    }

    // 保存：ベースバックエンドへ変換してから保存
    std::fs::create_dir_all("artifacts")?;
    let base_model: LeNet<B> = model.clone().valid();
    base_model
        .save_file("artifacts/model.burn", &CompactRecorder::new())
        .expect("save");
    println!("Saved: artifacts/model.burn");
    Ok(())
}

fn evaluate<Bx, ADx>(
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
        let preds = logits.argmax(1).reshape([-1]); // argmax は [B,1] になるため 1D に整形
        let eq = preds.equal(batch.targets.to_device(device));
        let batch_size = eq.dims()[0];
        let correct_batch = eq.int().sum().into_data().to_vec::<i64>().expect("sum")[0] as usize;
        correct += correct_batch;
        total += batch_size;
    }
    ((correct as f32) / (total.max(1) as f32), total)
}
