// src/model.rs
use burn::nn::{
    Linear, LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct LeNet<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    act: Relu,
}

impl<B: Backend> LeNet<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([1, 32], [5, 5])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);
        let conv2 = Conv2dConfig::new([32, 64], [5, 5])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);
        let pool = MaxPool2dConfig::new([2, 2]).init();
        let fc1 = LinearConfig::new(64 * 7 * 7, 128).init(device);
        let fc2 = LinearConfig::new(128, 10).init(device);
        let act = Relu::new();

        Self {
            conv1,
            conv2,
            pool,
            fc1,
            fc2,
            act,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // x: [B,1,28,28]
        let x = self.pool.forward(self.act.forward(self.conv1.forward(x))); // -> [B,32,14,14]
        let x = self.pool.forward(self.act.forward(self.conv2.forward(x))); // -> [B,64,7,7]

        let dims = x.dims(); // [B,64,7,7]
        let b = dims[0];
        let x = x.reshape([b, 64 * 7 * 7]);

        let x = self.act.forward(self.fc1.forward(x));
        self.fc2.forward(x) // logits [B,10]
    }
}
