use super::{DeviceMesh, MeshDim};

/// Specifies how a single dimension is distributed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimDistribution {
    /// Dimension is sharded across a specific mesh dimension.
    Sharded(MeshDim),
    /// Dimension is replicated (not sharded).
    Replicated,
}

/// Describes how a tensor is distributed across devices
#[derive(Clone, Debug)]
pub struct ShardingSpec<T> {
    /// Distribution pattern for each tensor dimension
    dim_distributions: Vec<DimDistribution>,
    /// Description of the device mesh
    device_mesh: DeviceMesh<T>,
}
