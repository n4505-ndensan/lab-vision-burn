<script lang="ts">
	import DrawCanvas from '../../components/DrawCanvas.svelte';
	import { MnistModel } from '$lib/mnist/lab_vision_burn_model.js';
	import ResultChart from '../../components/ResultChart.svelte';
	import { onMount } from 'svelte';

	// Svelte5 rune 環境で bind:this による更新を警告なく扱えるよう $state 化
	let originalImageCanvas: HTMLCanvasElement | undefined = $state();
	let processedImageCanvas: HTMLCanvasElement | undefined = $state();

	let image: ImageData | undefined = $state();
	let shouldInfer = $state(false);
	let inferTimer: any = null; // debounce timer
	let result: number[] = $state([]);
	let guess: number | undefined = $state();
	let ready = $state(false); // WASM 初期化 & モデルロード完了したか
	let mnist: MnistModel | null = $state(null);

	// 画像 (RGBA) -> グレースケール(0..255) Float32Array(28*28) へ変換
	function processImage(img: ImageData): ImageData {
		const { data, width, height } = img;
		const processed = data.slice();
		for (let i = 0; i < width * height; i++) {
			processed[i * 4] = 255 - data[i * 4];
			processed[i * 4 + 1] = 255 - data[i * 4 + 1];
			processed[i * 4 + 2] = 255 - data[i * 4 + 2];
			processed[i * 4 + 3] = data[i * 4 + 3];
		}
		return new ImageData(processed, width, height);
	}

	// 画像 (RGBA) -> グレースケール(0..255) Float32Array(28*28) へ変換
	function imageDataToFloat32(img: ImageData): Float32Array {
		const { data, width, height } = img;
		const out = new Float32Array(width * height);
		for (let i = 0; i < width * height; i++) {
			const r = data[i * 4];
			const g = data[i * 4 + 1];
			const b = data[i * 4 + 2];
			const gray = 0.299 * r + 0.587 * g + 0.114 * b;
			out[i] = gray;
		}
		return out;
	}

	// WASM 初期化（wasm-bindgen が生成した default export を await）
	$effect(() => {
		if (ready) return; // 既に初期化済み
		(async () => {
			const init = (await import('$lib/mnist/lab_vision_burn_model.js')).default;
			await init(); // wasm インスタンス化 -> 内部の `wasm` 変数がセット
			mnist = new MnistModel();
			await mnist.load(); // 埋め込み model.bin 読み込み
			ready = true;
		})();
	});

	async function runInferenceOnce() {
		if (!(ready && mnist && image)) return;
		if (originalImageCanvas) {
			originalImageCanvas.getContext('2d')?.putImageData(image, 0, 0);
		}
		const processedImage = processImage(image);
		if (processedImageCanvas) {
			processedImageCanvas.getContext('2d')?.putImageData(processedImage, 0, 0);
		}
		const input = imageDataToFloat32(processedImage);
		result = await mnist.inference(input);
		guess = result.indexOf(Math.max(...result));
	}

	function scheduleInference() {
		shouldInfer = true;
		if (inferTimer) clearTimeout(inferTimer);
		inferTimer = setTimeout(async () => {
			await runInferenceOnce();
			shouldInfer = false;
		}, 250); // 250ms デバウンス (調整可)
	}

	onMount(() => {
		return () => {
			mnist?.free();
			mnist = null;
		};
	});
</script>

{#if !ready}
	<div class="loading_root">
		<h1>LOADING...</h1>
	</div>
{:else}
	<div class="root">
		<a class="header" data-sveltekit-reload href="/">Lab-Vision-Burn</a>

		<!-- <p>WASM: {ready ? 'ready' : 'loading...'}</p> -->

		<div class="content">
			<div class="left">
				<p
					style="color: #00FFFF; font-family: ZFB09; font-size: 8px; text-transform: none; margin-bottom: 8px;"
				>
					canvas
				</p>
				<DrawCanvas
					width={28}
					height={28}
					onStart={() => {
						// 途中で再描画開始 → 推論キャンセル
						if (inferTimer) clearTimeout(inferTimer);
					}}
					onUpdate={(i) => {
						// ライブで表示更新のみ
						image = i;
					}}
					onCommit={(i) => {
						image = i;
						scheduleInference();
					}}
				/>
			</div>
			<div class="right">
				<p style="color: #00FFFF; font-family: ZFB09; font-size: 8px;text-transform: none;">
					result
				</p>
				<!-- <p>result: {JSON.stringify(result)}</p> -->

				<ResultChart result={result?.length ? result : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]} />

				{#if guess !== undefined}
					<p>
						<span style="text-transform: none;">x</span> it's
						<span style="color: #FF00FF;">{guess}</span>!
					</p>
				{:else}
					<p>Write your favorite number!</p>
				{/if}

				<div style="margin-top: 36px;">
					<p style="font-size: 8px">process</p>
					<div class="process_canvas_container">
						<canvas bind:this={originalImageCanvas} class="process_canvas" width={28} height={28}
						></canvas>

						<p>&gt;</p>

						<canvas bind:this={processedImageCanvas} class="process_canvas" width={28} height={28}
						></canvas>
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.header {
		font-size: 32px;
		margin-top: 16px;
		margin-bottom: 24px;
		font-family: 'ZFB09', monospace;
		text-decoration: none;
	}

	.header:hover {
		color: #ff00ff;
	}

	.loading_root {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;

		width: 100%;
		height: 100vh;
	}

	.root {
		display: flex;
		flex-direction: column;
		padding: 36px 80px;
		width: fit-content;
		justify-self: center;
		height: 100%;
		background-color: #101010;
		border-left: 1px solid #80808080;
		border-right: 1px solid #80808080;
	}
	.content {
		display: flex;
		flex-direction: row;
		width: 100%;
		gap: 72px;
		margin-top: 24px;
	}
	.left {
		display: flex;
		flex-direction: column;
	}
	.right {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.process_canvas_container {
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 12px;
		margin-top: 4px;
	}
	.process_canvas {
		width: 48px;
		height: 48px;
		image-rendering: pixelated;

		background-color: white;
	}

	/* ========== Responsive (Smartphone) ========== */
	@media (max-width: 680px) {
		.root {
			padding: 24px 24px 64px 24px;
			width: 100%;
			min-height: 100vh;
			height: auto;
			box-sizing: border-box;
			border-left: none;
			border-right: none;
		}
		.content {
			flex-direction: column;
			gap: 40px;
		}
		.left,
		.right {
			width: 100%;
		}
		.left {
		}
		.right {
			gap: 20px;
		}
		.process_canvas_container {
		}
		.process_canvas {
			width: 64px;
			height: 64px;
		}
	}
</style>
