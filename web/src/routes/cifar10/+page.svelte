<script lang="ts">
	// TODO: CIFAR-10用のWASMモジュールが完成したら実装
	let ready = $state(false);

	// プレースホルダーの実装
	$effect(() => {
		// 仮の初期化処理
		setTimeout(() => {
			ready = true;
		}, 1000);
	});
</script>

{#if !ready}
	<div class="loading_root">
		<h1>LOADING CIFAR-10...</h1>
	</div>
{:else}
	<div class="root">
		<h1>CIFAR-10 Image Classification</h1>
		<p class="subtitle">Coming Soon...</p>

		<div class="content">
			<div class="placeholder">
				<h2>🚧 Under Development</h2>
				<p>CIFAR-10モデルの学習と統合を進めています。</p>
				<p>完成予定機能：</p>
				<ul>
					<li>
						10クラス画像分類（airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck）
					</li>
					<li>テスト用画像のアップロード機能</li>
					<li>リアルタイム推論結果の表示</li>
				</ul>

				<a href="/" class="back-link">← Back to Home</a>
			</div>
		</div>
	</div>
{/if}

<style>
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
		flex-direction: column;
		align-items: center;
		margin-top: 24px;
	}

	.placeholder {
		text-align: center;
		max-width: 500px;
	}

	.placeholder h2 {
		color: #ff00ff;
		font-size: 24px;
		margin-bottom: 16px;
	}

	.placeholder p {
		color: #cccccc;
		line-height: 1.6;
		margin-bottom: 16px;
	}

	.placeholder ul {
		text-align: left;
		color: #cccccc;
		margin-bottom: 24px;
	}

	.placeholder li {
		margin-bottom: 8px;
	}

	.back-link {
		color: #00ffff;
		text-decoration: none;
		font-family: 'ZFB09', monospace;
		font-size: 14px;
		padding: 8px 16px;
		border: 1px solid #00ffff;
		border-radius: 4px;
		display: inline-block;
		transition: all 0.3s ease;
	}

	.back-link:hover {
		background-color: #00ffff;
		color: #000000;
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
	}
</style>
