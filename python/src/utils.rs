// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow::compute::concat;
use arrow::datatypes::Float32Type;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{cast::AsArray, Array, FixedSizeListArray, Float32Array, UInt32Array};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance::Result;
use lance::{datatypes::Schema, io::ObjectStore};
use lance_arrow::FixedSizeListArrayExt;
use lance_file::writer::FileWriter;
use lance_index::scalar::IndexWriter;
use lance_index::vector::hnsw::{builder::HnswBuildParams, HNSW};
use lance_index::vector::kmeans::{
    compute_partitions, KMeans as LanceKMeans, KMeansAlgoFloat, KMeansParams,
};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_linalg::distance::DistanceType;
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use pyo3::intern;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyIterator,
    IntoPyObjectExt,
};

use crate::RT;

#[pyclass(name = "_KMeans")]
pub struct KMeans {
    /// Number of clusters
    k: usize,

    /// Metric type
    metric_type: DistanceType,

    max_iters: u32,

    /// A trained KMean model. This is set after calling `fit`.
    trained_kmeans: Option<LanceKMeans>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (k, metric_type="l2", max_iters=50, centroids_arr=None))]
    fn new(
        k: usize,
        metric_type: &str,
        max_iters: u32,
        centroids_arr: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let trained_kmeans = if let Some(arr) = centroids_arr {
            let data = ArrayData::from_pyarrow_bound(arr)?;
            if !matches!(data.data_type(), DataType::FixedSizeList(_, _)) {
                return Err(PyValueError::new_err("Must be a FixedSizeList"));
            }
            let fixed_size_arr = FixedSizeListArray::from(data);
            let params = KMeansParams {
                distance_type: metric_type.try_into().unwrap(),
                max_iters,
                ..Default::default()
            };
            let kmeans =
                LanceKMeans::new_with_params(&fixed_size_arr, k, &params).map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Error initialing KMeans from existing centroids: {}",
                        e
                    ))
                })?;
            Some(kmeans)
        } else {
            None
        };
        Ok(Self {
            k,
            metric_type: metric_type.try_into().unwrap(),
            max_iters,
            trained_kmeans,
        })
    }

    /// Train the model
    fn fit(&mut self, _py: Python, arr: &Bound<PyAny>) -> PyResult<()> {
        let data = ArrayData::from_pyarrow_bound(arr)?;
        if !matches!(data.data_type(), DataType::FixedSizeList(_, _)) {
            return Err(PyValueError::new_err("Must be a FixedSizeList"));
        }
        let fixed_size_arr = FixedSizeListArray::from(data);
        let params = KMeansParams {
            distance_type: self.metric_type,
            max_iters: self.max_iters,
            ..Default::default()
        };
        let kmeans = LanceKMeans::new_with_params(&fixed_size_arr, self.k, &params)
            .map_err(|e| PyRuntimeError::new_err(format!("Error training KMeans: {}", e)))?;
        self.trained_kmeans = Some(kmeans);
        Ok(())
    }

    fn predict(&self, py: Python, array: &Bound<PyAny>) -> PyResult<PyObject> {
        let Some(kmeans) = self.trained_kmeans.as_ref() else {
            return Err(PyRuntimeError::new_err("KMeans must fit (train) first"));
        };
        let data = ArrayData::from_pyarrow_bound(array)?;
        if !matches!(data.data_type(), DataType::FixedSizeList(_, _)) {
            return Err(PyValueError::new_err("Must be a FixedSizeList"));
        }
        let fixed_size_arr = FixedSizeListArray::from(data);
        if kmeans.dimension != fixed_size_arr.value_length() as usize {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: kmean model {} != data {}",
                kmeans.dimension,
                fixed_size_arr.value_length()
            )));
        };
        if !matches!(fixed_size_arr.value_type(), DataType::Float32) {
            return Err(PyValueError::new_err("Must be a FixedSizeList of Float32"));
        };
        let values = fixed_size_arr.values().as_primitive();
        let centroids = kmeans.centroids.as_primitive();
        let cluster_ids = UInt32Array::from(
            compute_partitions::<Float32Type, KMeansAlgoFloat<Float32Type>>(
                centroids,
                values,
                kmeans.dimension,
                kmeans.distance_type,
            )
            .0,
        );
        cluster_ids.into_data().to_pyarrow(py)
    }

    fn centroids(&self, py: Python) -> PyResult<PyObject> {
        if let Some(kmeans) = self.trained_kmeans.as_ref() {
            let centroids: Float32Array = kmeans.centroids.as_primitive().clone();
            let fixed_size_arr =
                FixedSizeListArray::try_new_from_values(centroids, kmeans.dimension as i32)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Error converting centroids to FixedSizeListArray: {}",
                            e
                        ))
                    })?;
            fixed_size_arr.into_data().to_pyarrow(py)
        } else {
            Ok(py.None())
        }
    }
}

#[pyclass(name = "_Hnsw")]
pub struct Hnsw {
    hnsw: lance_index::vector::hnsw::HNSW,
    vectors: Arc<dyn Array>,
}

#[pymethods]
impl Hnsw {
    #[staticmethod]
    #[pyo3(signature = (
        vectors_array,
        max_level=7,
        m=20,
        ef_construction=100,
        distance_type="l2",
    ))]
    fn build(
        vectors_array: &Bound<PyIterator>,
        max_level: u16,
        m: usize,
        ef_construction: usize,
        distance_type: &str,
    ) -> PyResult<Self> {
        let params = HnswBuildParams::default()
            .max_level(max_level)
            .num_edges(m)
            .ef_construction(ef_construction);

        let mut data: Vec<Arc<dyn Array>> = Vec::new();
        for vectors in vectors_array {
            let vectors = ArrayData::from_pyarrow_bound(&vectors?)?;
            if !matches!(vectors.data_type(), DataType::FixedSizeList(_, _)) {
                return Err(PyValueError::new_err("Must be a FixedSizeList"));
            }
            data.push(Arc::new(FixedSizeListArray::from(vectors)));
        }
        let array_refs = data.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        let vectors = concat(&array_refs).map_err(|e| PyIOError::new_err(e.to_string()))?;
        std::mem::drop(data);

        let dt = DistanceType::try_from(distance_type)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let hnsw = RT
            .runtime
            .block_on(params.build(vectors.clone(), dt))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { hnsw, vectors })
    }

    #[pyo3(signature = (file_path))]
    fn to_lance_file(&self, py: Python, file_path: &str) -> PyResult<()> {
        let object_store = ObjectStore::local();
        let path = Path::parse(file_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mut writer = RT
            .block_on(
                Some(py),
                FileWriter::<ManifestDescribing>::try_new(
                    &object_store,
                    &path,
                    Schema::try_from(HNSW::schema().as_ref())
                        .map_err(|e| PyIOError::new_err(e.to_string()))?,
                    &Default::default(),
                ),
            )?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        RT.block_on(Some(py), async {
            let batch = self.hnsw.to_batch()?;
            let metadata = batch.schema_ref().metadata().clone();
            writer.write_record_batch(batch).await?;
            writer.finish_with_metadata(&metadata).await?;
            Result::Ok(())
        })?
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    fn vectors(&self, py: Python) -> PyResult<PyObject> {
        self.vectors.to_data().to_pyarrow(py)
    }
}

/// A newtype wrapper for a Lance type.
///
/// This is used for types that have a corresponding dataclass in Python.
pub struct PyLance<T>(pub T);

/// Extract a Vec of PyLance types from a Python object.
pub fn extract_vec<'a, T>(ob: &Bound<'a, PyAny>) -> PyResult<Vec<T>>
where
    PyLance<T>: FromPyObject<'a>,
{
    ob.extract::<Vec<PyLance<T>>>()
        .map(|v| v.into_iter().map(|t| t.0).collect())
}

/// Export a Vec of Lance types to a Python object.
pub fn export_vec<'a, T>(py: Python<'a>, vec: &'a [T]) -> PyResult<Vec<PyObject>>
where
    PyLance<&'a T>: IntoPyObject<'a>,
{
    vec.iter()
        .map(|t| PyLance(t).into_py_any(py))
        .collect::<std::result::Result<Vec<_>, _>>()
}

pub fn class_name(ob: &Bound<'_, PyAny>) -> PyResult<String> {
    let full_name: String = ob
        .getattr(intern!(ob.py(), "__class__"))?
        .getattr(intern!(ob.py(), "__name__"))?
        .extract()?;
    match full_name.rsplit_once('.') {
        Some((_, name)) => Ok(name.to_string()),
        None => Ok(full_name),
    }
}
