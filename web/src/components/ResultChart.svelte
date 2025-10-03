<script lang="ts">
	interface Props {
		result: number[];
		labels?: string[];
	}

	let { result, labels }: Props = $props();
</script>

<div class="chart_container">
	{#each result as item, index}
		<div class="chart_item">
			<p class="chart_item_label">{labels ? labels[index] : index}</p>
			<div class="chart_item_bar_container">
				<div
					class="chart_item_bar"
					style="height: {item * 100}%; background-color: {item === Math.max(...result)
						? 'magenta'
						: 'blue'}"
				></div>
			</div>
			<p class="chart_item_value">{Math.round(item * 100) / 100}</p>
		</div>
	{/each}
</div>

<style>
	.chart_container {
		display: flex;
		flex-direction: row;
		gap: 12px;
		height: 180px;
		/* margin: 12px; */
	}

	.chart_item {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: end;
	}
	.chart_item_label {
		font-size: 8px;
		margin-bottom: 12px;
		text-align: center;
		word-break: break-all;
		line-height: 1.2;
	}
	.chart_item_bar_container {
		display: flex;
		flex-direction: column;
		justify-content: end;
		width: 20px;
		flex-grow: 1;
		background-color: #ffffff10;
	}
	.chart_item_bar {
		width: 100%;
		background-color: blue;
	}
	.chart_item_value {
		font-size: 8px;
		margin-top: 4px;
	}
</style>
