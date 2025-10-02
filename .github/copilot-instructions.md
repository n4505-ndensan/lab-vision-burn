This is the project that implements computer vision model works on websites, using burn->wasm and sveltekit.

## website styles

- In default, paragraph, link, span, and any other components in the website uses 04B bitmap font as font-family, which the base fontsize is 8px.
- So, if there's no any specify instructions, use 8,16,24,... as fontsize.
- Also, the font-weight is not available in 04B font. So normally use "normal" weight and don't use "bold". Avoid using headline(h1, h2, ...) elements in the same reason.

- If the content is "only" in Japanese, you can use any font sizes.
