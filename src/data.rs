use burn::{data::dataloader::batcher::Batcher, data::dataset::vision::MnistItem, prelude::*};

#[derive(Clone, Default)]
pub struct MnistBatcher;

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 4>, // ← 3 → 4 に
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
