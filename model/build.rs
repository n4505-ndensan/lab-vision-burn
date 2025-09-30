use std::{env, fs, path::PathBuf};

fn main() {
    let artifact_path = PathBuf::from("artifacts/model.bin");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::create_dir_all(&out_dir).unwrap();
    let dest = out_dir.join("model.bin");

    if artifact_path.exists() {
        println!("cargo:rerun-if-changed=artifacts/model.bin");
        fs::copy(&artifact_path, &dest).expect("copy model.bin");
    } else {
        println!(
            "cargo:warning=artifacts/model.bin not found; embedding empty placeholder (run training first for real model)"
        );
        // 空ファイル生成 (include_bytes が失敗しないように)
        fs::write(&dest, &[] as &[u8]).expect("write placeholder model.burn");
    }
}
