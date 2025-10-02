#![recursion_limit = "256"]
// src/main.rs
mod config;
mod data;
mod model;
mod train;

use anyhow::{Result, anyhow};
use burn::{prelude::*, record::CompactRecorder};
use burn_wgpu::{Wgpu, WgpuDevice};
use clap::{Args, Parser, Subcommand};
use image::{DynamicImage, ImageReader};
use std::{fs, path::Path};
use config::DatasetConfig;
use model::{LeNet, CifarNet, ModelTrait};

#[derive(Subcommand)]
enum Commands {
    Train(TrainArgs),
    Eval(EvalArgs),
    Infer(InferArgs),
}

#[derive(Args)]
struct TrainArgs {
    #[arg(short, long, required = true)]
    dataset: String,
    #[arg(short, long)]
    epochs: Option<u32>,
    #[arg(short, long)]
    batch_size: Option<usize>,
}

#[derive(Args)]
struct EvalArgs {
    #[arg(short, long, required = true)]
    dataset: String,
}

#[derive(Args)]
struct InferArgs {
    #[arg(short, long, required = true)]
    dataset: String,
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
            let config = DatasetConfig::load(&args.dataset)?;
            let train_config = train::TrainConfig {
                dataset_config: config.clone(),
                epochs: args.epochs.unwrap_or(config.training.epochs),
                batch_size: args.batch_size.unwrap_or(config.training.batch_size),
            };
            train::train(train_config)?;
        }
        Commands::Eval(args) => {
            let _config = DatasetConfig::load(&args.dataset)?;
            println!("(tip) 現状は学習ログの test_acc を参照してください。");
        }
        Commands::Infer(args) => {
            let config = DatasetConfig::load(&args.dataset)?;
            infer_paths(&config, &args.path)?;
        }
    }
    Ok(())
}

fn infer_paths(config: &DatasetConfig, path: &str) -> Result<()> {
    type B = Wgpu;
    let device = WgpuDevice::default();

    let model_path = config.get_model_path();
    
    // モデルタイプに応じて適切なモデルを作成
    let (model_mnist, model_cifar): (Option<LeNet<B>>, Option<CifarNet<B>>) = match config.model.model_type.as_str() {
        "lenet" => {
            let model = LeNet::<B>::new(&device, config)
                .load_file(&model_path, &CompactRecorder::new(), &device)
                .map_err(|e| anyhow!("モデル読み込み失敗: {e}"))?;
            (Some(model), None)
        },
        "cifar_net" => {
            let model = CifarNet::<B>::new(&device, config)
                .load_file(&model_path, &CompactRecorder::new(), &device)
                .map_err(|e| anyhow!("モデル読み込み失敗: {e}"))?;
            (None, Some(model))
        },
        _ => return Err(anyhow!("未対応のモデルタイプ: {}", config.model.model_type))
    };

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
        println!("File,Pred,Class");
        for p in files {
            match infer_single_path(config, &model_mnist, &model_cifar, &device, &p) {
                Ok((pred_idx, class_name)) => println!("{},{},{}", p.display(), pred_idx, class_name),
                Err(e) => eprintln!("{},ERROR:{e},", p.display()),
            }
        }
    } else if Path::new(path).is_file() {
        let (pred_idx, class_name) = infer_single_path(config, &model_mnist, &model_cifar, &device, Path::new(path))?;
        println!("Predicted: {} ({})", pred_idx, class_name);
    } else {
        return Err(anyhow!(
            "指定パスがファイルでもディレクトリでもありません: {path}"
        ));
    }
    Ok(())
}

fn infer_single_path<B: Backend>(
    config: &DatasetConfig,
    model_mnist: &Option<LeNet<B>>,
    model_cifar: &Option<CifarNet<B>>,
    device: &<B as Backend>::Device,
    path: &Path,
) -> Result<(i32, String)> {
    let img = ImageReader::open(path)?.decode()?;
    let tensor = to_tensor::<B>(&img, config, device);
    
    let logits = match config.model.model_type.as_str() {
        "lenet" => {
            let model = model_mnist.as_ref().ok_or_else(|| anyhow!("LeNetモデルが初期化されていません"))?;
            model.forward(tensor)
        },
        "cifar_net" => {
            let model = model_cifar.as_ref().ok_or_else(|| anyhow!("CifarNetモデルが初期化されていません"))?;
            model.forward(tensor)
        },
        _ => return Err(anyhow!("未対応のモデルタイプ: {}", config.model.model_type))
    };
    
    let pred_idx = logits
        .argmax(1)
        .into_data()
        .to_vec::<i32>()
        .expect("prediction vec")[0];
    
    let class_name = config.class_names.get(pred_idx as usize)
        .cloned()
        .unwrap_or_else(|| format!("unknown_{}", pred_idx));
    
    Ok((pred_idx, class_name))
}

fn to_tensor<B: Backend>(img: &DynamicImage, config: &DatasetConfig, device: &B::Device) -> Tensor<B, 4> {
    let [height, width] = config.input_size;
    
    match config.input_channels {
        1 => {
            // グレースケール (MNIST)
            let img = img.to_luma8();
            let img = image::imageops::resize(&img, width as u32, height as u32, image::imageops::FilterType::Nearest);
            
            let mean = match &config.training.normalization.mean {
                config::NormalizationValue::Single(v) => *v,
                _ => panic!("グレースケール画像には単一の平均値が必要です")
            };
            let std = match &config.training.normalization.std {
                config::NormalizationValue::Single(v) => *v,
                _ => panic!("グレースケール画像には単一の標準偏差が必要です")
            };
            
            let data: Vec<f32> = img
                .pixels()
                .map(|p| (p[0] as f32 / 255.0 - mean) / std)
                .collect();
            let t = Tensor::<B, 1>::from_floats(data.as_slice(), device)
                .reshape([1, height, width]); // [C,H,W]
            t.reshape([1, 1, height, width]) // [B,C,H,W]
        },
        3 => {
            // RGB (CIFAR-10)
            let img = img.to_rgb8();
            let img = image::imageops::resize(&img, width as u32, height as u32, image::imageops::FilterType::Nearest);
            
            let mean = match &config.training.normalization.mean {
                config::NormalizationValue::Triple(v) => *v,
                _ => panic!("RGB画像には3つの平均値が必要です")
            };
            let std = match &config.training.normalization.std {
                config::NormalizationValue::Triple(v) => *v,
                _ => panic!("RGB画像には3つの標準偏差が必要です")
            };
            
            let mut data = Vec::with_capacity(3 * height * width);
            let pixels: Vec<_> = img.pixels().collect();
            
            // Channelごとに分離 (CHW format)
            for c in 0..3 {
                for pixel in &pixels {
                    let value = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                    data.push(value);
                }
            }
            
            let t = Tensor::<B, 1>::from_floats(data.as_slice(), device)
                .reshape([3, height, width]); // [C,H,W]
            t.reshape([1, 3, height, width]) // [B,C,H,W]
        },
        _ => panic!("未対応のチャンネル数: {}", config.input_channels)
    }
}
