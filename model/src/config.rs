use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelConfig {
    #[serde(rename = "type")]
    pub model_type: String,
    pub conv1_out: Option<usize>,
    pub conv2_out: Option<usize>,
    pub conv3_out: Option<usize>,
    pub fc1_out: usize,
    pub fc2_out: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NormalizationConfig {
    pub mean: NormalizationValue,
    pub std: NormalizationValue,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum NormalizationValue {
    Single(f32),
    Triple([f32; 3]),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub normalization: NormalizationConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ArtifactsConfig {
    pub dir: String,
    pub model_file: String,
    pub model_bin: String,
    pub wasm_bg: String,
    pub wasm_js: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DatasetConfig {
    pub name: String,
    pub input_channels: usize,
    pub input_size: [usize; 2],
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub artifacts: ArtifactsConfig,
}

impl DatasetConfig {
    pub fn load(dataset_name: &str) -> Result<Self> {
        let config_path = format!("configs/{}.json", dataset_name);
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("設定ファイル読み込み失敗 {}: {}", config_path, e))?;

        let config: DatasetConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("設定ファイル解析失敗 {}: {}", config_path, e))?;

        Ok(config)
    }

    pub fn get_model_path(&self) -> String {
        format!("{}/{}", self.artifacts.dir, self.artifacts.model_file)
    }

    pub fn get_model_bin_path(&self) -> String {
        format!("{}/{}", self.artifacts.dir, self.artifacts.model_bin)
    }
}
