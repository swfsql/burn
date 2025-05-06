use burn_tensor::{Float, Tensor, TensorKind, backend::Backend};

use crate::sharding::ShardingSpec;

/// Represents a tensor that is distributed (sharded or replicated) across multiple devices.
///
/// A `ShardedTensor` contains a local shard of a global tensor, along with metadata
/// describing how the full tensor is partitioned across devices. The distribution is
/// defined by a `ShardingSpec`, which specifies per-dimension distribution strategies
/// (e.g., sharded or replicated) and the associated device mesh.
pub struct ShardedTensor<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    /// Local shard of the tensor
    local_shard: Tensor<B, D, K>,
    /// Specification of how this tensor is sharded
    sharding_spec: ShardingSpec<B::Device>,
    /// Global shape of the tensor (across all devices)
    global_shape: Vec<usize>,
}
