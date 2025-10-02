// src/model.rs
use crate::config::DatasetConfig;
use burn::nn::{
    Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
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

#[derive(Module, Debug)]
pub struct CifarNet<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    dropout: Dropout,
    act: Relu,
}

pub trait ModelTrait<B: Backend> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2>;
}

impl<B: Backend> LeNet<B> {
    pub fn new(device: &B::Device, config: &DatasetConfig) -> Self {
        let conv1_out = config.model.conv1_out.unwrap_or(32);
        let conv2_out = config.model.conv2_out.unwrap_or(64);

        let conv1 = Conv2dConfig::new([config.input_channels, conv1_out], [5, 5])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);
        let conv2 = Conv2dConfig::new([conv1_out, conv2_out], [5, 5])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);
        let pool = MaxPool2dConfig::new([2, 2]).init();

        // 入力サイズから最終的な特徴マップサイズを計算
        let final_size = config.input_size[0] / 4; // 2回のプーリングで1/4
        let fc1 = LinearConfig::new(conv2_out * final_size * final_size, config.model.fc1_out)
            .init(device);
        let fc2 = LinearConfig::new(config.model.fc1_out, config.num_classes).init(device);
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
}

impl<B: Backend> ModelTrait<B> for LeNet<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.pool.forward(self.act.forward(self.conv1.forward(x)));
        let x = self.pool.forward(self.act.forward(self.conv2.forward(x)));

        let dims = x.dims();
        let b = dims[0];
        let flatten_size = dims[1] * dims[2] * dims[3];
        let x = x.reshape([b, flatten_size]);

        let x = self.act.forward(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}

impl<B: Backend> CifarNet<B> {
    pub fn new(device: &B::Device, config: &DatasetConfig) -> Self {
        let conv1_out = config.model.conv1_out.unwrap_or(64);
        let conv2_out = config.model.conv2_out.unwrap_or(128);
        let conv3_out = config.model.conv3_out.unwrap_or(256);
        let fc2_out = config.model.fc2_out.unwrap_or(256);

        let conv1 = Conv2dConfig::new([config.input_channels, conv1_out], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([conv1_out, conv2_out], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv3 = Conv2dConfig::new([conv2_out, conv3_out], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let pool = MaxPool2dConfig::new([2, 2]).init();

        // CIFAR-10: 32x32 -> 16x16 -> 8x8 -> 4x4
        let fc1 = LinearConfig::new(conv3_out * 4 * 4, config.model.fc1_out).init(device);
        let fc2 = LinearConfig::new(config.model.fc1_out, fc2_out).init(device);
        let fc3 = LinearConfig::new(fc2_out, config.num_classes).init(device);
        let dropout = DropoutConfig::new(0.5).init();
        let act = Relu::new();

        Self {
            conv1,
            conv2,
            conv3,
            pool,
            fc1,
            fc2,
            fc3,
            dropout,
            act,
        }
    }
}

impl<B: Backend> ModelTrait<B> for CifarNet<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // x: [B,3,32,32]
        let x = self.pool.forward(self.act.forward(self.conv1.forward(x))); // -> [B,64,16,16]
        let x = self.pool.forward(self.act.forward(self.conv2.forward(x))); // -> [B,128,8,8]
        let x = self.pool.forward(self.act.forward(self.conv3.forward(x))); // -> [B,256,4,4]

        let dims = x.dims();
        let b = dims[0];
        let x = x.reshape([b, dims[1] * dims[2] * dims[3]]);

        let x = self.dropout.forward(self.act.forward(self.fc1.forward(x)));
        let x = self.dropout.forward(self.act.forward(self.fc2.forward(x)));
        self.fc3.forward(x)
    }
}
