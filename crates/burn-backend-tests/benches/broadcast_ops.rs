//! Benchmarks for **broadcast** binary operations.
//!
//! `binary_ops` covers same-shape (and transposed) operands. This bench covers
//! the other half: operands of different shapes, where one side is a strided
//! view produced by `expand` / `unsqueeze` / `swap_dims` rather than a dense
//! buffer. Those views are what a backend's broadcast fast paths key off, and a
//! view that misses them costs an order of magnitude — so each group below pairs
//! a spelling that a fast path can recognise with one that computes the *same*
//! values through a shape the detector may not.
//!
//! The groups, and what each is asking:
//!
//! | Group | Question |
//! |-------|----------|
//! | `shared_row`   | `[N, C] ∘ [1, C]` — the layer-norm scale/shift shape |
//! | `per_row_scalar` | `[N, C] ∘ [N, 1]` — the layer-norm centre/scale shape |
//! | `operand_order` | is `dense ∘ broadcast` priced the same as `broadcast ∘ dense`? |
//! | `unit_leading_dim` | does a size-1 dim carrying a stride from `expand` / `swap_dims` cost anything? |
//! | `interleaved` | a broadcast dim *between* two dense ones — `[B, H, 1, R]` against `[B, H, P, R]` |
//! | `ownership` | does a uniquely-owned dense operand get written in place? |
//! | `add` / `non_commutative` | the same questions for `+`, and for the ops that cannot swap operands |
//!
//! Within a group every case does the same arithmetic on the same number of
//! elements, so the cases are directly comparable **to each other**; a large
//! spread inside one group is a missing fast path rather than a property of the
//! maths. That is deliberate — it makes the benchmark readable without a
//! baseline to diff against, and comparable across machines in a way absolute
//! timings are not.
//!
//! Reading the allocation columns matters as much as the timings here: several
//! of these shapes can be computed either by writing into an operand or by
//! allocating an output, and at these sizes that choice dominates.
//!
//! Run with:
//! ```bash
//! cargo bench --bench broadcast_ops --features flex-simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Tensor, TensorData};
use divan::Bencher;

#[cfg(not(feature = "bench-disable-alloc"))]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    println!("Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

// Two working-set sizes, matching `binary_ops`' vocabulary. MEDIUM is the size
// a decode step of a small language model actually touches per layer; LARGE
// checks the same shapes once the operands no longer fit in cache.
const ROWS: usize = 256;
const COLS: usize = 256; // MEDIUM: 64K elements
const ROWS_LARGE: usize = 1024;
const COLS_LARGE: usize = 1024; // LARGE: 1M elements

fn tensor_2d(rows: usize, cols: usize) -> Tensor<2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn tensor_3d(a: usize, b: usize, c: usize) -> Tensor<3> {
    let data: Vec<f32> = (0..a * b * c).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [a, b, c]), &Default::default())
}

fn tensor_4d(a: usize, b: usize, c: usize, d: usize) -> Tensor<4> {
    let data: Vec<f32> = (0..a * b * c * d)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [a, b, c, d]),
        &Default::default(),
    )
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            /// `[N, C] * [1, C]`: one contiguous row reused for every output row.
            #[divan::bench_group(name = "shared_row")]
            mod shared_row {
                use super::*;

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = tensor_2d(ROWS, COLS);
                    let b = tensor_2d(1, COLS);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(1, COLS_LARGE);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }
            }

            /// `[N, C] * [N, 1]`: one scalar per output row.
            #[divan::bench_group(name = "per_row_scalar")]
            mod per_row_scalar {
                use super::*;

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = tensor_2d(ROWS, COLS);
                    let b = tensor_2d(ROWS, 1);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(ROWS_LARGE, 1);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }
            }

            /// The same two products with the operands the other way round.
            /// `*` is commutative, so `dense_lhs` and `broadcast_lhs` must
            /// produce identical values — and ought to cost the same.
            #[divan::bench_group(name = "operand_order")]
            mod operand_order {
                use super::*;

                #[divan::bench]
                fn dense_lhs_row(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(1, COLS_LARGE);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn broadcast_lhs_row(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(1, COLS_LARGE);
                    bencher.bench_synced(|| b.clone() * a.clone());
                }

                #[divan::bench]
                fn dense_lhs_scalar(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(ROWS_LARGE, 1);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn broadcast_lhs_scalar(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(ROWS_LARGE, 1);
                    bencher.bench_synced(|| b.clone() * a.clone());
                }
            }

            /// A leading dim of extent 1 never advances, so its stride cannot
            /// change which elements are read — but `expand` writes 0 there and
            /// `swap_dims` can leave a stale value. All three cases below walk
            /// exactly the same dense buffer.
            #[divan::bench_group(name = "unit_leading_dim")]
            mod unit_leading_dim {
                use super::*;

                #[divan::bench]
                fn reshaped(bencher: Bencher) {
                    let a = tensor_3d(1, ROWS_LARGE, COLS);
                    let b = tensor_2d(ROWS_LARGE, COLS).reshape([1, ROWS_LARGE, COLS]);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn expanded(bencher: Bencher) {
                    let a = tensor_3d(1, ROWS_LARGE, COLS);
                    let b = tensor_2d(ROWS_LARGE, COLS)
                        .unsqueeze::<3>()
                        .expand([1, ROWS_LARGE, COLS]);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                #[divan::bench]
                fn swapped(bencher: Bencher) {
                    let a = tensor_3d(1, ROWS_LARGE, COLS);
                    // `[N, 1, C]` -> `[1, N, C]`: the same element order, but the
                    // leading stride is now whatever dim 1 carried.
                    let b = tensor_3d(ROWS_LARGE, 1, COLS).swap_dims(0, 1);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }
            }

            /// A broadcast dim between two dense ones: `[B, H, 1, R]` against
            /// `[B, H, P, R]`, i.e. each `(b, h)` row of length `R` reused for
            /// all `P`. This is the recurrent-SSM readout shape, and it is
            /// neither "every outer dim is broadcast" nor "the inner dims are a
            /// scalar" — the two shapes a two-case detector tends to cover.
            #[divan::bench_group(name = "interleaved")]
            mod interleaved {
                use super::*;

                const B: usize = 1;
                const H: usize = 24;
                const P: usize = 64;
                const R: usize = 128;

                #[divan::bench]
                fn broadcast_middle_dim(bencher: Bencher) {
                    let a = tensor_4d(B, H, P, R);
                    let b = tensor_3d(B, H, R).unsqueeze_dim::<4>(2);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                /// Control: the same output size with the broadcast dim
                /// innermost, which the per-row-scalar shape already covers.
                #[divan::bench]
                fn broadcast_inner_dim(bencher: Bencher) {
                    let a = tensor_4d(B, H, P, R);
                    let b = tensor_3d(B, H, P).unsqueeze_dim::<4>(3);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }

                /// Control: no broadcast at all, same element count.
                #[divan::bench]
                fn dense(bencher: Bencher) {
                    let a = tensor_4d(B, H, P, R);
                    let b = tensor_4d(B, H, P, R);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }
            }

            /// `add` alongside `mul`, since the two share a dispatch path and
            /// only the commutative ones can have their operands swapped.
            #[divan::bench_group(name = "add")]
            mod add {
                use super::*;

                #[divan::bench]
                fn dense_lhs(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(1, COLS_LARGE);
                    bencher.bench_synced(|| a.clone() + b.clone());
                }

                #[divan::bench]
                fn broadcast_lhs(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(1, COLS_LARGE);
                    bencher.bench_synced(|| b.clone() + a.clone());
                }

                #[divan::bench]
                fn interleaved(bencher: Bencher) {
                    let a = tensor_4d(1, 24, 64, 128);
                    let b = tensor_3d(1, 24, 128).unsqueeze_dim::<4>(2);
                    bencher.bench_synced(|| a.clone() + b.clone());
                }
            }

            /// Whether the dense operand is *uniquely* owned decides whether a
            /// backend may write the result into it or must allocate an output,
            /// which changes the memory traffic of the op — so a broadcast
            /// implementation can win on one of these and lose on the other.
            ///
            /// Both occur in ordinary eager code: `a.clone() * b` leaves `a`
            /// shared, while a temporary in the middle of an expression
            /// (`a * b + c * d`) is uniquely owned by the time the next operator
            /// sees it. A chain of such expressions is entirely `unique_dense`.
            ///
            /// The two cases differ *only* in ownership — same shapes, same
            /// arithmetic, same output size — so the allocation columns should
            /// read 8 MB for `shared_dense` and ~nothing for `unique_dense` on a
            /// backend that writes in place. Compare the pair, not the absolute
            /// numbers.
            #[divan::bench_group(name = "ownership")]
            mod ownership {
                use super::*;

                const S: usize = 256;
                const N: usize = 4096;

                /// Dense operand freshly built per iteration, so the op receives
                /// it uniquely owned. `with_inputs` keeps that construction out
                /// of the timed region.
                #[divan::bench]
                fn unique_dense(bencher: Bencher) {
                    let b = tensor_3d(1, 1, N);
                    bencher
                        .with_inputs(|| tensor_3d(2, S, N))
                        .bench_values(|a| {
                            let r = a * b.clone();
                            common::sync();
                            r
                        });
                }

                /// The same product with the dense operand shared, which is what
                /// every `a.clone() op b.clone()` in this file does.
                #[divan::bench]
                fn shared_dense(bencher: Bencher) {
                    let a = tensor_3d(2, S, N);
                    let b = tensor_3d(1, 1, N);
                    bencher.bench_synced(|| a.clone() * b.clone());
                }
            }

            /// `sub` and `div` do not commute, so an operand swap is not
            /// available to them; these cases exist to keep that visible.
            #[divan::bench_group(name = "non_commutative")]
            mod non_commutative {
                use super::*;

                #[divan::bench]
                fn sub_dense_lhs(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(ROWS_LARGE, 1);
                    bencher.bench_synced(|| a.clone() - b.clone());
                }

                #[divan::bench]
                fn sub_broadcast_lhs(bencher: Bencher) {
                    let a = tensor_2d(ROWS_LARGE, COLS_LARGE);
                    let b = tensor_2d(ROWS_LARGE, 1);
                    bencher.bench_synced(|| b.clone() - a.clone());
                }

                #[divan::bench]
                fn div_interleaved(bencher: Bencher) {
                    let a = tensor_4d(1, 24, 64, 128);
                    let b = tensor_3d(1, 24, 128).unsqueeze_dim::<4>(2);
                    bencher.bench_synced(|| a.clone() / b.clone());
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
