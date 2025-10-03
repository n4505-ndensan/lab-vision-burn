<script lang="ts">
	import Cifar10ImagesGrid, { type ImageData } from '../../components/Cifar10ImagesGrid.svelte';
	import { Cifar10Model } from '$lib/cifar10/lab_vision_burn_model.js';
	import ResultChart from '../../components/ResultChart.svelte';
	import { onMount } from 'svelte';

	let selectedImage: ImageData | null = $state(null);
	let selectedImageElement: HTMLImageElement | null = $state(null);
	let result: number[] = $state([]);
	let guess: number | undefined = $state();
	let ready = $state(false); // WASM 初期化 & モデルロード完了したか
	let cifar10: Cifar10Model | null = $state(null);
	let isInferring = $state(false);

	let isCorrect = $derived((): boolean | undefined => {
		if (!cifar10 || guess === undefined || !selectedImage) return undefined;
		return cifar10.getClassName(guess) === selectedImage.className;
	});

	// 画像をFloat32Arrayに変換（32x32x3のRGBデータ）
	function imageToFloat32Array(img: HTMLImageElement): Promise<Float32Array> {
		return new Promise((resolve) => {
			const canvas = document.createElement('canvas');
			const ctx = canvas.getContext('2d')!;
			canvas.width = 32;
			canvas.height = 32;

			// 画像を32x32にリサイズして描画
			ctx.drawImage(img, 0, 0, 32, 32);

			const imageData = ctx.getImageData(0, 0, 32, 32);
			const data = imageData.data;

			// RGBAからRGBに変換（0~255の値をそのまま使用）
			const float32Data = new Float32Array(32 * 32 * 3);

			for (let i = 0; i < 32 * 32; i++) {
				// モデルとして正規化は不要 (0~255のままでOK)
				// const r = data[i * 4] / 255.0;
				// const g = data[i * 4 + 1] / 255.0;
				// const b = data[i * 4 + 2] / 255.0;

				const r = data[i * 4];
				const g = data[i * 4 + 1];
				const b = data[i * 4 + 2];

				// CHW形式で格納 (Channels first)
				float32Data[i] = r;
				float32Data[32 * 32 + i] = g;
				float32Data[32 * 32 * 2 + i] = b;
			}

			resolve(float32Data);
		});
	}

	// WASM 初期化
	$effect(() => {
		if (ready) return;
		(async () => {
			try {
				const init = (await import('$lib/cifar10/lab_vision_burn_model.js')).default;
				await init();
				cifar10 = new Cifar10Model();
				await cifar10.load();
				ready = true;
			} catch (error) {
				console.error('Failed to initialize CIFAR-10 model:', error);
			}
		})();
	});

	async function runInference() {
		if (!(ready && cifar10 && selectedImage && selectedImageElement)) return;

		isInferring = true;
		try {
			const input = await imageToFloat32Array(selectedImageElement);
			result = await cifar10.inference(input);
			guess = result.indexOf(Math.max(...result));
		} catch (error) {
			console.error('Inference failed:', error);
		} finally {
			isInferring = false;
		}
	}

	function handleImageSelect(imageData: ImageData) {
		selectedImage = imageData;
		// 新しい画像を読み込み、推論を実行
		const img = new Image();
		img.crossOrigin = 'anonymous';
		img.onload = () => {
			selectedImageElement = img;
			runInference();
		};
		img.src = imageData.path;
	}

	onMount(() => {
		return () => {
			cifar10?.free();
			cifar10 = null;
		};
	});
</script>

{#if !ready}
	<div class="loading_root">
		<h1>LOADING CIFAR-10...</h1>
	</div>
{:else}
	<div class="root">
		<a class="header" data-sveltekit-reload href="/">Lab-Vision-Burn</a>
		<p class="subtitle">CIFAR-10 Image Classification</p>

		<div class="content">
			<div class="left">
				<Cifar10ImagesGrid onImageSelect={handleImageSelect} />
			</div>

			<div class="right">
				{#if selectedImage}
					<div class="selected-image">
						<p
							style="color: #00FFFF; font-family: ZFB09; font-size: 8px; text-transform: none; margin-bottom: 16px;"
						>
							selected image
						</p>
						<div class="image-preview">
							<img src={selectedImage.path} alt={selectedImage.className} />
							<div class="image-info">
								<p class="label">Class</p>
								<p class="highlight">{selectedImage.className}</p>

								<p class="label">Path</p>
								<p class="path">{selectedImage.path}</p>

								{#if isInferring}
									<p class="inferring">Inferring...</p>
								{/if}
							</div>
						</div>
					</div>

					{#if result.length > 0}
						<div class="result-section">
							<p style="color: #00FFFF; font-family: ZFB09; font-size: 8px; text-transform: none;">
								result
							</p>

							<ResultChart
								{result}
								labels={[
									'air',
									'car',
									'bird',
									'cat',
									'deer',
									'dog',
									'frog',
									'horse',
									'ship',
									'truck'
								]}
							/>

							{#if guess !== undefined}
								<div class="prediction">
									<p>
										Predicted: <span class="prediction-result">{cifar10?.getClassName(guess)}</span>
									</p>
									<p class="confidence">
										Confidence: {(Math.max(...result) * 100).toFixed(1)}%
									</p>
								</div>

								<div class="result">
									{#if isCorrect()}
										<p class="correct">s</p>
										<p class="detail">successfully guessed</p>
									{:else if isCorrect() === false}
										<p class="wrong">t</p>
										<p class="detail">wrongly guessed</p>
									{/if}
									<p class="note"></p>
								</div>
							{/if}
						</div>
					{/if}
				{:else}
					<div class="instructions">
						<p>Click an image to classify!</p>
						<p class="note">
							The model will predict one of 10 classes:<br />
							airplane, automobile, bird, cat, deer,<br />
							dog, frog, horse, ship, truck
						</p>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.header {
		font-size: 32px;
		margin-top: 16px;
		margin-bottom: 8px;
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

	.subtitle {
		color: #00ffff;
		font-size: 16px;
		margin-bottom: 24px;
	}

	.content {
		display: flex;
		flex-direction: row;
		gap: 64px;
		margin-top: 24px;
		width: 100%;
	}

	.left {
		flex: 2;
	}

	.right {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 24px;
		min-width: 400px;
	}

	.selected-image {
		display: flex;
		flex-direction: column;
	}

	.image-preview {
		display: flex;
		flex-direction: row;
		gap: 16px;
	}

	.image-preview img {
		width: 128px;
		height: 128px;
		object-fit: cover;
		border: 2px solid #ff00ff;
		image-rendering: pixelated;
	}

	.image-info .label {
		font-size: 8px;
		margin-top: 8px;
		margin-bottom: 4px;
	}

	.image-info .highlight {
		font-size: 16px;
		color: #ff00ff;
	}

	.image-info .path {
		font-size: 8px;
		opacity: 0.6;
	}

	.inferring {
		color: #00ffff;
		animation: pulse 1.5s infinite;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}

	.result-section {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.prediction {
	}

	.prediction p {
		font-size: 8px;
		margin: 4px 0;
	}

	.prediction-result {
		color: #ff00ff;
		font-weight: normal;
	}

	.confidence {
		color: #00ffff;
	}

	.instructions {
		color: #cccccc;
	}

	.instructions p {
		font-size: 8px;
		margin: 8px 0;
	}

	.result {
		display: flex;
		flex-direction: column;
		align-items: end;
		margin-right: 16px;
	}

	.result .correct {
		font-size: 32px;
		color: #00ff00;
		font-family: 'ZFB21', 'monospace';
		text-transform: none;
	}

	.result .wrong {
		font-size: 32px;
		color: #ff0000;
		font-family: 'ZFB21', 'monospace';
		text-transform: none;
	}

	.result .detail {
		font-size: 8px;
		font-family: 'ZFB09', 'monospace';
		text-transform: none;
		margin-top: 8px;
	}

	.note {
		color: #808080;
		line-height: 1.4;
	}

	/* Responsive */
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
			gap: 24px;
		}

		.left,
		.right {
			flex: none;
			width: 100%;
		}

		.image-preview img {
			width: 96px;
			height: 96px;
		}
	}
</style>
