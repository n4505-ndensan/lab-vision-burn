#![recursion_limit = "256"]
// src/main.rs
mod data;
mod model;
mod train;

use anyhow::Result;
use burn::{prelude::*, record::CompactRecorder};
use burn_wgpu::{Wgpu, WgpuDevice};
use clap::{Args, Parser, Subcommand};
use image::{DynamicImage, ImageReader};

#[derive(Subcommand)]
enum Commands {
    Train(TrainArgs),
    Eval(EvalArgs),
    Infer(InferArgs),
}

#[derive(Args)]
struct TrainArgs {
    #[arg(short, long, default_value_t = 5)]
    epochs: u32,
    #[arg(short, long, default_value_t = 64)]
    batch_size: usize,
}

#[derive(Args)]
struct EvalArgs {}

#[derive(Args)]
struct InferArgs {
    #[arg(short, long, required = true)]
    path: String,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CLI {
    #[command(subcommand)]
    command: Commands,
}

fn main() -> Result<()> {
    let cli = CLI::parse();
    match &cli.command {
        Commands::Train(args) => {
            train::train(train::TrainConfig {
                epochs: args.epochs,
                batch_size: args.batch_size,
            })?;
        }
        Commands::Eval(_args) => {
            // artifacts/model.burn を読み込み、test精度だけ出す（学習後の再確認用）
            // 簡易には train の最後の evaluate を見る運用でもOKなので省略可。
            println!("(tip) 現状は学習ログの test_acc を参照してください。");
        }
        Commands::Infer(args) => {
            infer_once(&args.path)?;
        }
    }
    Ok(())
}

fn infer_once(path: &str) -> Result<()> {
    type B = Wgpu;
    let device = WgpuDevice::default();

    // モデル読込（非AD）
    let model = model::LeNet::<B>::new(&device)
        .load_file("artifacts/model.burn", &CompactRecorder::new(), &device)
        .expect("load model");

    // 画像読み込み → 28x28 グレースケール → 正規化 → [1,1,28,28]
    let img = ImageReader::open(path)?.decode()?;
    let img = to_mnist_tensor::<B>(&img, &device);
    let logits = model.forward(img);
    let pred = logits
        .argmax(1)
        .into_data()
        .to_vec::<i64>()
        .expect("prediction vec")[0];

    println!("Predicted: {}", pred);
    Ok(())
}

fn to_mnist_tensor<B: Backend>(img: &DynamicImage, device: &B::Device) -> Tensor<B, 4> {
    let img = img.to_luma8();
    let img = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Nearest);
    let data: Vec<f32> = img
        .pixels()
        .map(|p| (p[0] as f32 / 255.0 - 0.1307) / 0.3081)
        .collect();
    let t = Tensor::<B, 2>::from_floats(data.as_slice(), device).reshape([1, 28, 28]);
    t.reshape([1, 1, 28, 28])
}
