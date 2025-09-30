<script lang="ts">
	import { Anvil } from '$lib/anvil';
	import { onMount } from 'svelte';

	interface Props {
		width?: number;
		height?: number;
		onUpdate?: (image: ImageData) => void; // 描画中の都度更新
		onCommit?: (image: ImageData) => void; // pointerup で最終画像確定
		onStart?: () => void; // pointerdown で描画開始
	}

	let {
		width = 28,
		height = 28,
		onUpdate = (i) => {},
		onCommit = (i) => {},
		onStart = () => {}
	}: Props = $props();

	const scale = 12;

	let canvas: HTMLCanvasElement;

	let anvil = new Anvil(width, height, 10);

	let rawX = $state(0);
	let rawY = $state(0);
	let pxX = $derived(Math.floor(rawX));
	let pxY = $derived(Math.floor(rawY));
	let lastPxX = $state<number | undefined>(0);
	let lastPxY = $state<number | undefined>(0);
	let drawing = $state(false);

	const updateCanvas = () => {
		const imageData = new ImageData(anvil.getBufferData().slice(), width, height);
		onUpdate(imageData);
		if (canvas) {
			const ctx = canvas.getContext('2d');
			if (ctx) {
				ctx.putImageData(imageData, 0, 0);
			}
		}
	};

	$effect(() => {
		if (drawing) {
			anvil.setPixel(pxX, pxY, [0, 0, 0, 255]);

			// completion line
			if (lastPxX && lastPxY) {
				const dx = pxX - lastPxX;
				const dy = pxY - lastPxY;
				for (let step = 1; step <= Math.max(Math.abs(dx), Math.abs(dy)); step++) {
					const intermediateX =
						lastPxX + Math.round((dx * step) / Math.max(Math.abs(dx), Math.abs(dy)));
					const intermediateY =
						lastPxY + Math.round((dy * step) / Math.max(Math.abs(dx), Math.abs(dy)));
					anvil.setPixel(intermediateX, intermediateY, [0, 0, 0, 255]);
				}
			}

			updateCanvas();
			lastPxX = pxX;
			lastPxY = pxY;
		}
	});

	onMount(() => {
		canvas = document.getElementById('canvas') as HTMLCanvasElement;
		if (!canvas) return;

		// initial reset
		anvil.fillAll([255, 255, 255, 255]);
		updateCanvas();

		canvas.addEventListener(
			'pointerdown',
			(e) => {
				e.preventDefault();
				rawX = e.offsetX;
				rawY = e.offsetY;
				drawing = true;
				onStart();
			},
			{ passive: false }
		);

		canvas.addEventListener(
			'pointermove',
			(e) => {
				e.preventDefault();
				rawX = e.offsetX;
				rawY = e.offsetY;
			},
			{ passive: false }
		);

		window.addEventListener(
			'pointerup',
			(e) => {
				e.preventDefault();
				lastPxX = undefined;
				lastPxY = undefined;
				drawing = false;
				// 最終画像を callback
				const imageData = new ImageData(anvil.getBufferData().slice(), width, height);
				onCommit(imageData);
			},
			{ passive: false }
		);

		canvas.addEventListener(
			'pointerout',
			(e) => {
				e.preventDefault();
				rawX = e.offsetX;
				rawY = e.offsetY;
				// drawing = false;
			},
			{ passive: false }
		);
		canvas.addEventListener(
			'pointercancel',
			(e) => {
				e.preventDefault();
				rawX = e.offsetX;
				rawY = e.offsetY;
				drawing = false;
			},
			{ passive: false }
		);
	});
</script>

<div id="canvas-container" style="width: {width * scale}px; height: {height * scale}px;">
	<canvas
		id="canvas"
		style="width: {width}px; height: {height}px; transform: scale({scale}); image-rendering: pixelated;"
		{width}
		{height}
	></canvas>
</div>

<button
	style="align-self: end;"
	onclick={() => {
		anvil.fillAll([255, 255, 255, 255]);
		updateCanvas();
	}}>reset</button
>

<div id="info-container">
	<!-- <p>raw: {rawX}, {rawY}</p> -->
	<p>pixel: {pxX}, {pxY}</p>
	<p>{drawing ? 'drawing' : 'not drawing'}</p>
</div>

<style>
	#canvas-container {
		border: 1px solid black;
		margin-bottom: 12px;
	}

	#canvas {
		transform-origin: 0 0;
	}

	#info-container {
		display: flex;
		flex-direction: column;
		gap: 0.5em;
		margin-top: 1em;

		font-size: 8px;

		opacity: 0.5;
	}
</style>
