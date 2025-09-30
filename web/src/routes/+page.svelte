<script lang="ts">
	import DrawCanvas from '../components/DrawCanvas.svelte';
	import { Mnist } from '$lib/model-wasm/lab_vision_burn_model';
	import ResultChart from '../components/ResultChart.svelte';
	import { onMount } from 'svelte';

	let originalImageCanvas: HTMLCanvasElement;
	let processedImageCanvas: HTMLCanvasElement;

	let image: ImageData | undefined = $state();
	let shouldInfer = $state(false);
	let inferTimer: any = null; // debounce timer
	let result: number[] = $state([]);
	let guess: number | undefined = $state();
	let ready = $state(false); // WASM 初期化 & モデルロード完了したか
	let mnist: Mnist | null = $state(null);

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
			const init = (await import('$lib/model-wasm/lab_vision_burn_model.js')).default;
			await init(); // wasm インスタンス化 -> 内部の `wasm` 変数がセット
			mnist = new Mnist();
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

<div class="root">
	<h1>LAB-VISION-BURN</h1>
	<p>WASM: {ready ? 'ready' : 'loading...'}</p>
	<div class="content">
		<div class="left">
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
			<!-- <p>result: {JSON.stringify(result)}</p> -->

			{#if guess !== undefined}
				<p>I guess it's {guess}!</p>
			{:else}
				<p>Write your favorite number!</p>
			{/if}

			<ResultChart result={result?.length ? result : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]} />

			<div>
				<p style="font-size: 12px">process</p>
				<div class="process_canvas_container">
					<canvas bind:this={originalImageCanvas} class="process_canvas" width={28} height={28}
					></canvas>

					<p>→</p>

					<canvas bind:this={processedImageCanvas} class="process_canvas" width={28} height={28}
					></canvas>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	.root {
		display: flex;
		flex-direction: column;
		width: 100%;
		height: 100%;
		margin: 36px 48px;
	}
	.content {
		display: flex;
		flex-direction: row;
		width: 100%;
		gap: 12px;
		margin-top: 36px;
	}
	.left {
		display: flex;
		flex-direction: column;
		min-width: 400px;
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
		gap: 8px;
		margin-top: 4px;
	}
	.process_canvas {
		width: 48px;
		height: 48px;
		border: 1px solid black;
		image-rendering: pixelated;
	}
</style>
