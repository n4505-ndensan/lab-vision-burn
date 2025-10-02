use burn::{data::dataloader::batcher::Batcher, data::dataset::vision::MnistItem, prelude::*};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// CIFAR-10用のアイテム構造体を定義
#[derive(Clone, Debug)]
pub struct Cifar10Item {
    pub image: [[[f32; 32]; 32]; 3], // RGB channels, 32x32
    pub label: usize,
}

#[derive(Clone, Default)]
pub struct MnistBatcher;

#[derive(Clone, Default)]
pub struct CifarBatcher;

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 4>, // ← 3 → 4 に
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct CifarBatch<B: Backend> {
    pub images: Tensor<B, 4>, // [batch_size, channels, height, width]
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|t| t.reshape([1, 28, 28])) // [C,H,W]
            .map(|t| ((t / 255.0) - 0.1307) / 0.3081) // 正規化 (MNIST慣例)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0).reshape([-1, 1, 28, 28]); // [B,1,28,28]
        let targets = Tensor::cat(targets, 0); // [B]

        MnistBatch { images, targets }
    }
}

impl<B: Backend> Batcher<B, Cifar10Item, CifarBatch<B>> for CifarBatcher {
    fn batch(&self, items: Vec<Cifar10Item>, device: &B::Device) -> CifarBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                // image is [3, 32, 32] - convert to tensor
                let mut data = Vec::new();
                for c in 0..3 {
                    for h in 0..32 {
                        for w in 0..32 {
                            data.push(item.image[c][h][w]);
                        }
                    }
                }
                TensorData::from(data.as_slice()).convert::<B::FloatElem>()
            })
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|t| t.reshape([3, 32, 32])) // [C,H,W]
            .map(|t| (t / 255.0)) // 正規化 [0,1]
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0).reshape([-1, 3, 32, 32]); // [B,3,32,32]
        let targets = Tensor::cat(targets, 0); // [B]

        CifarBatch { images, targets }
    }
}

// CIFAR-10データセットの実装
pub struct Cifar10Dataset {
    samples: Vec<Cifar10Item>,
}

impl Cifar10Dataset {
    pub fn train() -> Self {
        match Self::load_cifar10_binary(true) {
            Ok(dataset) => {
                println!(
                    "✓ CIFAR-10訓練データを読み込みました: {} samples",
                    dataset.samples.len()
                );
                dataset
            }
            Err(e) => {
                println!("⚠ CIFAR-10データの読み込みに失敗: {}", e);
                println!("  データセットをダウンロードするには:");
                println!("  .\\scripts\\download_cifar10.ps1");
                println!("  ダミーデータで代用します...");
                Self::create_dummy_dataset(50000)
            }
        }
    }

    pub fn test() -> Self {
        match Self::load_cifar10_binary(false) {
            Ok(dataset) => {
                println!(
                    "✓ CIFAR-10テストデータを読み込みました: {} samples",
                    dataset.samples.len()
                );
                dataset
            }
            Err(_) => Self::create_dummy_dataset(10000),
        }
    }

    fn create_dummy_dataset(size: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut samples = Vec::new();
        for i in 0..size {
            let mut image = [[[0.0f32; 32]; 32]; 3];

            // より意味のあるランダムパターンを生成
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let seed = hasher.finish();

            for c in 0..3 {
                for h in 0..32 {
                    for w in 0..32 {
                        // チャンネルと位置に基づいたパターンを生成
                        let val = ((seed as usize + c * 1000 + h * 10 + w) % 256) as f32;
                        image[c][h][w] = val;
                    }
                }
            }

            let label = (seed % 10) as usize; // 10クラス
            samples.push(Cifar10Item { image, label });
        }

        Cifar10Dataset { samples }
    }

    fn load_cifar10_binary(is_train: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let data_dir = Path::new("datasets/cifar-10/cifar-10-batches-bin");

        if !data_dir.exists() {
            return Err(
                format!("CIFAR-10データディレクトリが見つかりません: {:?}", data_dir).into(),
            );
        }

        if is_train {
            // 訓練データ: data_batch_1.bin ~ data_batch_5.bin
            let mut all_samples = Vec::new();
            for i in 1..=5 {
                let batch_path = data_dir.join(format!("data_batch_{}.bin", i));
                let batch_samples = Self::read_cifar_batch(&batch_path)?;
                all_samples.extend(batch_samples);
            }
            Ok(Cifar10Dataset {
                samples: all_samples,
            })
        } else {
            // テストデータ: test_batch.bin
            let test_path = data_dir.join("test_batch.bin");
            let samples = Self::read_cifar_batch(&test_path)?;
            Ok(Cifar10Dataset { samples })
        }
    }

    fn read_cifar_batch(path: &Path) -> Result<Vec<Cifar10Item>, Box<dyn std::error::Error>> {
        if !path.exists() {
            return Err(format!("バッチファイルが見つかりません: {:?}", path).into());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut samples = Vec::new();

        // CIFAR-10バイナリ形式: 各サンプルは3073バイト
        // 1バイト目: ラベル (0-9)
        // 残り3072バイト: 32x32x3の画像データ (R,G,B順)
        let mut buffer = [0u8; 3073];

        while reader.read_exact(&mut buffer).is_ok() {
            let label = buffer[0] as usize;
            let mut image = [[[0.0f32; 32]; 32]; 3];

            let mut idx = 1;
            // RGBの順で読み込み
            for c in 0..3 {
                for h in 0..32 {
                    for w in 0..32 {
                        image[c][h][w] = buffer[idx] as f32;
                        idx += 1;
                    }
                }
            }

            samples.push(Cifar10Item { image, label });
        }

        println!(
            "  バッチ読み込み完了: {} samples from {:?}",
            samples.len(),
            path.file_name().unwrap()
        );
        Ok(samples)
    }
}

impl burn::data::dataset::Dataset<Cifar10Item> for Cifar10Dataset {
    fn get(&self, index: usize) -> Option<Cifar10Item> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

// CIFAR-10クラス名の定義
pub const CIFAR10_CLASSES: [&str; 10] = [
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

// 便利メソッド
impl Cifar10Dataset {
    pub fn get_class_name(label: usize) -> &'static str {
        CIFAR10_CLASSES.get(label).map_or("unknown", |v| v)
    }

    pub fn class_count() -> usize {
        10
    }
}
