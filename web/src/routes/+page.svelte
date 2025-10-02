<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';

	const models = [
		{
			name: 'MNIST',
			description: '手書き数字認識（0-9）',
			path: '/mnist',
			status: 'ready'
		},
		{
			name: 'CIFAR-10',
			description: '画像分類（10クラス）',
			path: '/cifar10',
			status: 'coming_soon' // 実装後は 'ready' に変更
		}
	];

	function navigateToModel(path: string, status: string) {
		if (status === 'ready') {
			goto(path);
		}
	}
</script>

<div class="root">
	<h1>Lab-Vision-Burn</h1>
	<p class="subtitle">Burn Deep Learning Models in the Browser</p>

	<div class="models">
		{#each models as model}
			<div
				class="model-card {model.status}"
				role="button"
				tabindex="0"
				on:click={() => navigateToModel(model.path, model.status)}
				on:keydown={(e) => {
					if (e.key === 'Enter' || e.key === ' ') {
						navigateToModel(model.path, model.status);
					}
				}}
			>
				<div class="model-header">
					<h2>{model.name}</h2>
					<div class="status-badge {model.status}">
						{model.status === 'ready' ? 'READY' : 'COMING SOON'}
					</div>
				</div>
				<p class="model-description">{model.description}</p>
				{#if model.status === 'ready'}
					<div class="cta">Click to try →</div>
				{/if}
			</div>
		{/each}
	</div>

	<footer class="footer">
		<p>
			Built with 🔥 <a href="https://burn.dev/" target="_blank" rel="noopener">Burn</a> & 🚀
			<a href="https://kit.svelte.dev/" target="_blank" rel="noopener">SvelteKit</a>
		</p>
	</footer>
</div>

<style>
	.root {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 100vh;
		padding: 36px 80px;
		background-color: #101010;
	}

	h1 {
		font-size: 48px;
		margin-bottom: 8px;
		text-align: center;
	}

	.subtitle {
		font-size: 16px;
		color: #00ffff;
		margin-bottom: 48px;
		text-align: center;
	}

	.models {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
		gap: 24px;
		width: 100%;
		max-width: 800px;
	}

	.model-card {
		background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
		border: 1px solid #404040;
		border-radius: 8px;
		padding: 24px;
		cursor: pointer;
		transition: all 0.3s ease;
		position: relative;
		overflow: hidden;
	}

	.model-card:hover {
		border-color: #00ffff;
		transform: translateY(-2px);
		box-shadow: 0 8px 25px rgba(0, 255, 255, 0.15);
	}

	.model-card.coming_soon {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.model-card.coming_soon:hover {
		transform: none;
		border-color: #606060;
		box-shadow: none;
	}

	.model-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 12px;
	}

	.model-card h2 {
		font-size: 24px;
		margin: 0;
		font-family: 'ZFB09', monospace;
	}

	.status-badge {
		font-size: 8px;
		padding: 4px 8px;
		border-radius: 4px;
		font-family: 'ZFB09', monospace;
	}

	.status-badge.ready {
		background-color: #00ff00;
		color: #000000;
	}

	.status-badge.coming_soon {
		background-color: #404040;
		color: #ffffff;
	}

	.model-description {
		color: #cccccc;
		font-size: 16px;
		margin-bottom: 16px;
		line-height: 1.5;
	}

	.cta {
		color: #ff00ff;
		font-size: 16px;
		font-family: 'ZFB09', monospace;
		text-transform: uppercase;
	}

	.footer {
		margin-top: 96px;
		text-align: center;
	}

	.footer p {
		font-size: 8px;
		color: #808080;
	}

	.footer a {
		color: #00ffff;
		text-decoration: none;
	}

	.footer a:hover {
		text-decoration: underline;
	}

	/* Responsive */
	@media (max-width: 680px) {
		.root {
			padding: 24px 24px 64px 24px;
		}

		h1 {
			font-size: 36px;
		}

		.models {
			grid-template-columns: 1fr;
		}

		.model-card {
			padding: 20px;
		}
	}
</style>
