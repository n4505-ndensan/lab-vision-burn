curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh -s -- -y

# Build MNIST WASM module
cd ../model
wasm-pack build --target bundler --out-dir ../web/src/lib/mnist -- --features mnist
# Build CIFAR-10 WASM module
wasm-pack build --target bundler --out-dir ../web/src/lib/cifar10 -- --features cifar10

# Return to web directory and build the app
cd ../web
npm run build