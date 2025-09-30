use crate::model::LeNet;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use burn_wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};
type Backend = burn_wgpu::Wgpu;

static STATE_ENCODED: &[u8] = include_bytes!("../artifacts/model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> LeNet<Backend> {
    init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

    let model: LeNet<Backend> = LeNet::new(&Default::default());
    let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
        .load(STATE_ENCODED, &Default::default())
        .expect("Failed to decode state");

    model.load_record(record)
}
