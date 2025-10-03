/* tslint:disable */
/* eslint-disable */
export function start(): void;
/**
 * MNIST専用の推論クラス
 */
export class MnistModel {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * コンストラクタ
   */
  constructor();
  /**
   * モデルをロード
   */
  load(): Promise<void>;
  /**
   * 推論実行（確率配列を返す）
   */
  inference(input: Float32Array): Promise<Array<any>>;
  /**
   * Top-1予測クラスのみ返す
   */
  inferenceTop1(input: Float32Array): Promise<number>;
  /**
   * モデルがロード済みか確認
   */
  isLoaded(): boolean;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_mnistmodel_free: (a: number, b: number) => void;
  readonly mnistmodel_new: () => number;
  readonly mnistmodel_load: (a: number) => any;
  readonly mnistmodel_inference: (a: number, b: number, c: number) => any;
  readonly mnistmodel_inferenceTop1: (a: number, b: number, c: number) => any;
  readonly mnistmodel_isLoaded: (a: number) => number;
  readonly start: () => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_6: WebAssembly.Table;
  readonly closure2803_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure2815_externref_shim: (a: number, b: number, c: any, d: any) => void;
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
