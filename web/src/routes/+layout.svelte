<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children?.()}

<style>
	:global(html),
	:global(body) {
		width: 100%;
		height: 100%;
		margin: 0;
		padding: 0;
		background-color: #000000;
	}
	:global(*) {
		color: white;
	}

	:global(p) {
		line-height: 1;
		padding: 0;
		margin: 0;

		font-family: ZFB21;
		text-transform: uppercase;
	}
	:global(h1) {
		font-weight: normal;
		font-size: 24px;
		font-family: ZFB09;
	}

	:global(button) {
		width: fit-content;
		font-family: ZFB21;
		text-transform: uppercase;
		font-size: 8px;

		background-color: #101010;
		color: white;

		border: 1px solid white;
		border-radius: 3px;
		padding: 4px 10px;

		cursor: pointer;
	}

	@media (any-hover: hover) {
		:global(button:hover) {
			background-color: #ff00ff;
		}
	}
</style>
