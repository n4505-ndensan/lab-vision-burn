#![recursion_limit = "256"]
// src/main.rs
mod data;
mod model;
mod train;

use anyhow::{Result, anyhow};
use burn::{prelude::*, record::CompactRecorder};
use burn_wgpu::{Wgpu, WgpuDevice};
use clap::{Args, Parser, Subcommand};
use image::{DynamicImage, ImageReader};
use std::{fs, path::Path};

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
            println!("(tip) 現状は学習ログの test_acc を参照してください。");
        }
        Commands::Infer(args) => {
            infer_paths(&args.path)?;
        }
    }
    Ok(())
}

fn infer_paths(path: &str) -> Result<()> {
    type B = Wgpu;
    let device = WgpuDevice::default();

    // モデルを一度だけロード
    let model = model::LeNet::<B>::new(&device)
        .load_file("artifacts/model.burn", &CompactRecorder::new(), &device)
        .map_err(|e| anyhow!("モデル読み込み失敗: {e}"))?;

    let meta = fs::metadata(path)?;
    if meta.is_dir() {
        let mut files: Vec<_> = fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if files.is_empty() {
            println!("(info) ディレクトリ内に .png ファイルがありません: {path}");
            return Ok(());
        }
        println!("File,Pred");
        for p in files {
            match infer_single_path(&model, &device, &p) {
                Ok(pred) => println!("{},{}", p.display(), pred),
                Err(e) => eprintln!("{},ERROR:{e}", p.display()),
            }
        }
    } else if Path::new(path).is_file() {
        let pred = infer_single_path(&model, &device, Path::new(path))?;
        println!("Predicted: {pred}");
    } else {
        return Err(anyhow!(
            "指定パスがファイルでもディレクトリでもありません: {path}"
        ));
    }
    Ok(())
}

fn infer_single_path<B: Backend>(
    model: &model::LeNet<B>,
    device: &<B as Backend>::Device,
    path: &Path,
) -> Result<i32> {
    let img = ImageReader::open(path)?.decode()?;
    let tensor = to_mnist_tensor::<B>(&img, device);
    let logits = model.forward(tensor);
    let pred = logits
        .argmax(1)
        .into_data()
        .to_vec::<i32>()
        .expect("prediction vec")[0];
    Ok(pred)
}

fn to_mnist_tensor<B: Backend>(img: &DynamicImage, device: &B::Device) -> Tensor<B, 4> {
    // 1) グレースケール化 & 28x28 へリサイズ
    let img = img.to_luma8();
    let img = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Nearest);
    // 2) フラット (784) -> 1D Tensor を生成 (ランク 1 で作成し後段 reshape)
    let data: Vec<f32> = img
        .pixels()
        .map(|p| (p[0] as f32 / 255.0 - 0.1307) / 0.3081)
        .collect();
    let t = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 28, 28]); // [C,H,W] = [1,28,28]
    t.reshape([1, 1, 28, 28]) // [B,C,H,W] = [1,1,28,28]
}
