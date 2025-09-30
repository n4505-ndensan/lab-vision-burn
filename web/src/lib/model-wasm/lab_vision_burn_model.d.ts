/* tslint:disable */
/* eslint-disable */
export function start(): void;
/**
 * Mnist structure that corresponds to JavaScript class.
 * See:[exporting-rust-struct](https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html)
 */
export class Mnist {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Constructor called by JavaScripts with the new keyword.
   */
  constructor();
  /**
   * Returns the inference results.
   *
   * This method is called from JavaScript via generated wrapper code by wasm-bindgen.
   *
   * # Arguments
   *
   * * `input` - A f32 slice of input 28x28 image
   *
   * See bindgen support types for passing and returning arrays:
   * * [number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/number-slices.html)
   * * [boxed-number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/boxed-number-slices.html)
   */
  inference(input: Float32Array): Promise<Array<any>>;
  /**
   * 明示的に学習済みモデルをロード (二度目以降は何もしない)
   */
  load(): Promise<void>;
  /**
   * モデルがロード済みか
   */
  isLoaded(): boolean;
  /**
   * Top-1 クラス (0-9) を返す簡易推論 API
   */
  inferenceTop1(input: Float32Array): Promise<number>;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_mnist_free: (a: number, b: number) => void;
  readonly mnist_new: () => number;
  readonly mnist_inference: (a: number, b: number, c: number) => any;
  readonly mnist_load: (a: number) => any;
  readonly mnist_isLoaded: (a: number) => number;
  readonly mnist_inferenceTop1: (a: number, b: number, c: number) => any;
  readonly start: () => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_6: WebAssembly.Table;
  readonly closure2793_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure2805_externref_shim: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
