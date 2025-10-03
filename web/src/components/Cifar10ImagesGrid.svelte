<script lang="ts">
	export type ImageData = {
		path: string;
		className: string;
	};

	let selectedImage: ImageData | null = $state(null);
	let images: ImageData[] = $state([]);

	// CIFAR-10のクラス名
	const classNames = [
		'airplane',
		'automobile',
		'bird',
		'cat',
		'deer',
		'dog',
		'frog',
		'horse',
		'ship',
		'truck'
	];

	// 各クラスから2-3枚ずつランダムに選択してシャッフル（合計25枚）
	function generateRandomImages(): ImageData[] {
		const result: ImageData[] = [];
		// const imagesPerClass = [3, 3, 2, 3, 2, 3, 2, 3, 3, 3]; // 合計25枚になるよう配分
		const targetTotal = 25;

		for (let i = 0; i < targetTotal; i++) {
			const className = classNames[Math.floor(Math.random() * classNames.length)];
			const index = Math.floor(Math.random() * 1000); // 0-999の範囲でランダム選択

			result.push({
				path: `/cifar10-test/${className}/${index.toString().padStart(4, '0')}.jpg`,
				className
			});
		}

		// シャッフル
		for (let i = result.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[result[i], result[j]] = [result[j], result[i]];
		}

		return result;
	}

	// 初期化
	$effect(() => {
		images = generateRandomImages();
	});

	// 画像選択時のコールバックプロップス
	interface Props {
		onImageSelect?: (imageData: ImageData) => void;
	}

	const { onImageSelect = () => {} }: Props = $props();

	function handleImageClick(imageData: ImageData) {
		selectedImage = imageData;
		onImageSelect(imageData);
	}

	function regenerateImages() {
		images = generateRandomImages();
		selectedImage = null;
	}
</script>

<div class="grid-container">
	<div class="header">
		<p>CIFAR-10 Test Images</p>
		<button class="regenerate-btn" onclick={regenerateImages}> Regenerate </button>
	</div>

	<div class="images-grid">
		{#each images as imageData, index}
			<button
				class="image-item"
				class:selected={selectedImage?.path === imageData.path}
				onclick={() => handleImageClick(imageData)}
			>
				<img src={imageData.path} alt={imageData.className} loading="lazy" />
				<div class="image-label">
					{imageData.className}
				</div>
			</button>
		{/each}
	</div>

	<!-- {#if selectedImage}
		<div class="selected-info">
			<p>Selected: <span class="highlight">{selectedImage.className}</span></p>
		</div>
	{/if} -->
</div>

<style>
	.grid-container {
		display: flex;
		flex-direction: column;
		width: 100%;
	}

	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 16px;
	}

	.header p {
		font-size: 16px;
		margin: 0;
	}

	.regenerate-btn {
		background: transparent;
		border: 1px solid #ff00ff;
		color: #ff00ff;
		font-family: 'ZFB09', monospace;
		font-size: 8px;
		padding: 4px 8px;
		cursor: pointer;
		transition: all 0.3s ease;
	}

	.regenerate-btn:hover {
		background: #ff00ff;
		color: #000000;
	}

	.images-grid {
		display: grid;
		grid-template-columns: repeat(5, 80px);
		gap: 16px;
		margin-bottom: 16px;
		justify-content: center;
	}

	.image-item {
		position: relative;
		aspect-ratio: 1;
		cursor: pointer;
		border: 2px solid transparent;
		transition: all 0.2s ease;
		overflow: hidden;
		background: none;
		padding: 0;
		display: block;
		width: 100%;
		min-height: 80px;
		min-width: 80px;
	}

	.image-item:hover {
		border-color: #00ffff;
	}

	.image-item.selected {
		border-color: #ff00ff;
		box-shadow: 0 0 8px #ff00ff80;
	}

	.image-item img {
		width: 100%;
		height: 100%;
		object-fit: cover;
		display: block;
		image-rendering: pixelated;
	}

	.image-label {
		position: absolute;
		bottom: 0;
		left: 0;
		right: 0;
		background: rgba(0, 0, 0, 0.8);
		color: #ffffff;
		font-size: 8px;
		padding: 2px 4px;
		text-align: center;
		font-family: 'ZFB09', monospace;
	}

	.selected-info {
		text-align: center;
		margin-top: 8px;
	}

	.selected-info p {
		font-size: 8px;
		margin: 0;
	}

	.highlight {
		color: #ff00ff;
	}

	/* Responsive */
	@media (max-width: 680px) {
		.images-grid {
			grid-template-columns: repeat(5, 50px);
			gap: 12px;
		}

		.image-item {
			min-height: 50px;
			min-width: 50px;
		}

		.header {
			flex-direction: column;
			gap: 8px;
			align-items: flex-start;
		}

		.image-label {
			font-size: 6px;
			padding: 1px 2px;
		}
	}
</style>
