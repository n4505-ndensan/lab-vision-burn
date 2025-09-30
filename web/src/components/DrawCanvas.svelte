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

	// 基準スケール（十分な横幅がある場合）
	const baseScale = 14;
	// 実際に適用されるスケール（画面幅に応じて縮小）
	let scale = $state(baseScale);

	function recomputeScale() {
		// 余白を考慮（左右合計で 100px くらい確保したい要求）
		const margin = 100; // px
		const targetWidth = width * baseScale;
		const viewport = typeof window !== 'undefined' ? window.innerWidth : targetWidth + margin;
		if (viewport < targetWidth + margin) {
			// 入りきらないので縮小（整数スケールを優先しつつ最低1）
			const candidate = (viewport - margin) / width;
			// candidate が baseScale 以上なら縮小不要
			if (candidate >= baseScale) {
				scale = baseScale;
			} else {
				// ピクセルアートのにじみを避けるため整数へ（1未満は1）
				const intCandidate = Math.max(1, Math.floor(candidate));
				scale = intCandidate;
			}
		} else {
			scale = baseScale;
		}
	}

	let canvas: HTMLCanvasElement;

	let anvil = new Anvil(width, height, 10);

	let rawX = $state(0);
	let rawY = $state(0);
	let pxX = $derived(Math.floor(rawX));
	let pxY = $derived(Math.floor(rawY));
	let lastPxX = $state<number | undefined>(undefined);
	let lastPxY = $state<number | undefined>(undefined);
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
		if (!drawing) return;
		// 範囲外なら無視
		if (pxX < 0 || pxY < 0 || pxX >= width || pxY >= height) return;
		anvil.setPixel(pxX, pxY, [0, 0, 0, 255]);

		// 直線補完（前フレームからの差分を埋める）
		if (lastPxX !== undefined && lastPxY !== undefined) {
			const dx = pxX - lastPxX;
			const dy = pxY - lastPxY;
			const steps = Math.max(Math.abs(dx), Math.abs(dy));
			for (let step = 1; step <= steps; step++) {
				const intermediateX = lastPxX + Math.round((dx * step) / steps);
				const intermediateY = lastPxY + Math.round((dy * step) / steps);
				if (
					intermediateX >= 0 &&
					intermediateY >= 0 &&
					intermediateX < width &&
					intermediateY < height
				) {
					anvil.setPixel(intermediateX, intermediateY, [0, 0, 0, 255]);
				}
			}
		}

		updateCanvas();
		lastPxX = pxX;
		lastPxY = pxY;
	});

	// CSS transform(scale) やレイアウトに依存しない座標計算
	function computeCanvasCoords(e: PointerEvent) {
		if (!canvas) return { cx: -1, cy: -1 };
		const rect = canvas.getBoundingClientRect();
		// 可視上の位置 -> 論理キャンバス座標へスケール変換
		const scaleX = canvas.width / rect.width;
		const scaleY = canvas.height / rect.height;
		const cx = (e.clientX - rect.left) * scaleX;
		const cy = (e.clientY - rect.top) * scaleY;
		return { cx, cy };
	}

	onMount(() => {
		canvas = document.getElementById('canvas') as HTMLCanvasElement;
		if (!canvas) return;

		// 初期スケール計算
		recomputeScale();
		const resizeHandler = () => recomputeScale();
		window.addEventListener('resize', resizeHandler);

		// initial reset
		anvil.fillAll([255, 255, 255, 255]);
		updateCanvas();

		canvas.addEventListener(
			'pointerdown',
			(e) => {
				if (!e.isPrimary) return; // マルチタッチは無視
				e.preventDefault();
				canvas.setPointerCapture(e.pointerId);
				const { cx, cy } = computeCanvasCoords(e);
				rawX = cx;
				rawY = cy;
				drawing = true;
				onStart();
			},
			{ passive: false }
		);

		canvas.addEventListener(
			'pointermove',
			(e) => {
				if (!drawing) return;
				if (!e.isPrimary) return;
				e.preventDefault();
				const { cx, cy } = computeCanvasCoords(e);
				rawX = cx;
				rawY = cy;
			},
			{ passive: false }
		);

		window.addEventListener(
			'pointerup',
			(e) => {
				if (!e.isPrimary) return;
				e.preventDefault();
				lastPxX = undefined;
				lastPxY = undefined;
				if (drawing) {
					drawing = false;
					// 最終画像を callback
					const imageData = new ImageData(anvil.getBufferData().slice(), width, height);
					onCommit(imageData);
				}
			},
			{ passive: false }
		);

		canvas.addEventListener(
			'pointerout',
			(e) => {
				if (!e.isPrimary) return;
				// pointer capture中は描画継続させるためここでは停止しない
			},
			{ passive: true }
		);
		canvas.addEventListener(
			'pointercancel',
			(e) => {
				if (!e.isPrimary) return;
				e.preventDefault();
				drawing = false;
				lastPxX = undefined;
				lastPxY = undefined;
			},
			{ passive: false }
		);

		return () => {
			window.removeEventListener('resize', resizeHandler);
		};
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
		/* border: 1px solid black; */
		margin-bottom: 12px;
	}

	#canvas {
		transform-origin: 0 0;
		/* スマホでのスクロール/ピンチによる干渉を防ぐ */
		touch-action: none;
		-webkit-user-select: none;
		user-select: none;
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
