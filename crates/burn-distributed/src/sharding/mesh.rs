use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

/// Represents a logical mesh dimension, identified by a unique name.
///
/// Mesh dimensions are used to define the logical structure of a device mesh.
/// They help organize and partition parallel computation workloads, such as data,
/// tensor, or pipeline parallelism, across the mesh. Each dimension is named
/// to provide unambiguous mapping during sharding operations.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MeshDim {
    /// A name that uniquely identifies the mesh dimension.
    name: String,
}

impl MeshDim {
    /// Constructs a new [`MeshDim`] with the given name.
    pub fn new<S: Into<String>>(name: S) -> Self {
        MeshDim { name: name.into() }
    }
}

/// Represents a logical arrangement of devices used for parallel computation.
///
/// A `DeviceMesh` defines a structured, N-dimensional topology over a set of physical devices,
/// where each dimension can be given a unique name (e.g., `"data"`, `"model"`, `"pipeline"`).
/// This logical mesh provides a basis for specifying tensor sharding and data distribution
/// strategies across devices. Mesh dimensions do not need to correspond to physical layout
/// and are primarily used to organize parallelism (e.g., data, tensor, or pipeline parallelism).
///
/// For example, a 2D mesh with shape `[2, 4]` and dimensions `["dp", "tp"]` represents
/// a logical grid of devices for 2-way data parallelism and 4-way tensor parallelism.
///
/// The mesh must use unique names for each dimension to allow unambiguous mapping
/// between tensor dimensions and mesh dimensions during sharding.
#[derive(Clone, Debug)]
pub struct DeviceMesh<T> {
    /// Physical devices in an n-dimensional logical arrangement
    devices: Vec<T>,
    /// Shape of the logical mesh
    shape: Vec<usize>,
    /// Maps dimension names to their indices in the mesh
    dims: HashMap<MeshDim, usize>,
}

/// Represents errors that can occur when constructing a `DeviceMesh`.
///
/// These errors are typically related to invalid dimension mappings or mismatches
/// between the device count and the mesh shape.
#[derive(Debug)]
pub enum DeviceMeshError {
    /// An invalid dimension was specified.
    InvalidDimension(String),
    /// The mesh configuration is invalid.
    InvalidMesh(String),
}

/// A builder for constructing a [`DeviceMesh`].
///
/// This builder allows you to define a logical mesh structure with a shape and
/// specific dimension mappings. Once all dimensions are mapped, the builder can
/// be used to construct the [`DeviceMesh`] instance.
#[derive(Clone, Debug)]
pub struct DeviceMeshBuilder<T> {
    /// Physical devices in an n-dimensional logical arrangement.
    devices: Vec<T>,
    /// Shape of the logical mesh.
    shape: Vec<usize>,
    /// Maps dimension names to their indices in the mesh.
    dims: HashMap<MeshDim, usize>,
}

impl<T> DeviceMeshBuilder<T> {
    /// Creates a new [`DeviceMeshBuilder`] with the given devices and shape.
    ///
    /// # Arguments
    /// * `devices` - A vector of devices to be arranged in the mesh.
    /// * `shape` - A vector representing the shape (dimensions) of the mesh.
    pub fn new<S: Into<Vec<usize>>>(devices: Vec<T>, shape: S) -> Self {
        Self {
            devices,
            shape: shape.into(),
            dims: HashMap::new(),
        }
    }

    // TODO: check docstring, change example for batch dim 0 maybe?

    /// Adds a dimension to the logical mesh, associating it with a specific index.
    ///
    /// The `with_dim` method allows you to map a mesh dimension (`MeshDim`) to a given index
    /// in the mesh shape. Each dimension represents a logical axis of parallelism and should
    /// correspond to a unique index in the shape of the mesh. This function ensures that the
    /// dimension index is valid and that no other dimension has already been assigned to that
    /// index. This can be called multiple times to map various dimensions to indices in the mesh.
    ///
    /// # Arguments
    /// * `idx` - The index of the dimension in the mesh shape.
    /// * `dim` - The mesh dimension to be assigned.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mesh = DeviceMeshBuilder::new(vec![0, 1, 2, 3], [2, 2])
    ///     .with_dim(0, MeshDim::new("data_parallel"))
    ///     .with_dim(1, MeshDim::new("tensor_parallel"))
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// In the example above, the first dimension (`"data_parallel"`) is assigned to index 0
    /// and the second dimension (`"tensor_parallel"`) is assigned to index 1 in the 2D mesh shape
    /// `[2, 2]`. This allows for specifying how the tensor is sharded across devices based on these
    /// logical mesh dimensions.
    pub fn with_dim(mut self, idx: usize, dim: MeshDim) -> Self {
        self.dims.insert(dim, idx);
        self
    }

    /// Builds a [`DeviceMesh`] from the current configuration.
    ///
    /// This method performs validation to ensure that all dimensions are correctly
    /// mapped and that the number of devices matches the expected shape. If the
    /// validation fails, an error is returned.
    ///
    /// # Returns
    /// A `DeviceMesh` if the configuration is valid, or a `DeviceMeshError` if invalid.
    pub fn build(self) -> Result<DeviceMesh<T>, DeviceMeshError> {
        let ndim = self.shape.len();
        let mut seen_indices = HashSet::new();

        for &idx in self.dims.values() {
            if idx >= ndim {
                return Err(DeviceMeshError::InvalidDimension(format!(
                    "Index {} exceeds mesh shape {:?}",
                    idx, self.shape
                )));
            }
            if !seen_indices.insert(idx) {
                return Err(DeviceMeshError::InvalidDimension(format!(
                    "Dimension {} already mapped",
                    idx
                )));
            }
        }

        if seen_indices.len() != ndim {
            return Err(DeviceMeshError::InvalidMesh(format!(
                "Not all mesh dimensions are mapped. Got {}, expected {}",
                seen_indices.len(),
                ndim
            )));
        }

        let expected_devices = self.shape.iter().product::<usize>();
        if self.devices.len() != expected_devices {
            return Err(DeviceMeshError::InvalidMesh(format!(
                "Device count ({}) doesn't match mesh shape {:?}",
                self.devices.len(),
                self.shape,
            )));
        }

        Ok(DeviceMesh {
            devices: self.devices,
            shape: self.shape,
            dims: self.dims,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_device_mesh_2x2() {
        let mesh = DeviceMeshBuilder::new(vec![0, 1, 2, 3], [2, 2])
            .with_dim(0, MeshDim::new("x"))
            .with_dim(1, MeshDim::new("y"))
            .build();

        assert!(mesh.is_ok());
    }

    #[test]
    #[should_panic = "InvalidDimension(\"Index 2 exceeds mesh shape [2, 2]\")"]
    fn test_device_mesh_dim_should_be_in_bound() {
        let _mesh = DeviceMeshBuilder::new(vec![0, 1, 2, 3], [2, 2])
            .with_dim(0, MeshDim::new("x"))
            .with_dim(2, MeshDim::new("y")) // out of bounds
            .build()
            .unwrap();
    }

    #[test]
    #[should_panic = "InvalidDimension(\"Dimension 0 already mapped\")"]
    fn test_device_mesh_dim_should_be_unique() {
        let _mesh = DeviceMeshBuilder::new(vec![0, 1, 2, 3], [2, 2])
            .with_dim(0, MeshDim::new("x"))
            .with_dim(0, MeshDim::new("y")) // already mapped
            .build()
            .unwrap();
    }

    #[test]
    #[should_panic = "InvalidMesh(\"Not all mesh dimensions are mapped."]
    fn test_device_mesh_dim_should_be_all_mapped() {
        let _mesh = DeviceMeshBuilder::new(vec![0, 1, 2, 3], [2, 2])
            .with_dim(0, MeshDim::new("x"))
            // .with_dim(0, MeshDim::new("y")) // already mapped
            .build()
            .unwrap();
    }

    #[test]
    #[should_panic = "InvalidMesh(\"Not all mesh dimensions are mapped."]
    fn test_device_mesh_dim_name_should_be_unique() {
        let _mesh = DeviceMeshBuilder::new(vec![0, 1, 2, 3], [2, 2])
            .with_dim(0, MeshDim::new("x"))
            .with_dim(1, MeshDim::new("x")) // same name is invalid
            .build()
            .unwrap();
    }

    #[test]
    #[should_panic = "InvalidMesh(\"Device count (4) doesn't match mesh shape [3, 2]"]
    fn test_device_mesh_devices_should_match_shape() {
        let _mesh = DeviceMeshBuilder::new(vec![0, 1, 2, 3], [3, 2])
            .with_dim(0, MeshDim::new("x"))
            .with_dim(1, MeshDim::new("y"))
            .build()
            .unwrap();
    }
}
