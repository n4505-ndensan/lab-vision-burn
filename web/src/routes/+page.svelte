<script lang="ts">
	import { onMount } from 'svelte';
	import { Anvil } from '$lib/anvil';
	import DrawCanvas from '../components/DrawCanvas.svelte';

	async () => {
		const canvas = document.getElementById('canvas') as HTMLCanvasElement;
		if (!canvas) return;
	};
</script>

<h1>LAB-VISION-BURN</h1>

<DrawCanvas />
