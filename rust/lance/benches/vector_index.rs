// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
#![allow(clippy::print_stdout)]

use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, cast::as_primitive_array,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;

use lance::dataset::{Dataset, WriteMode, WriteParams, builder::DatasetBuilder};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::{FixedSizeListArrayExt, as_fixed_size_list_array};
use lance_index::{
    IndexType,
    vector::{ivf::IvfBuildParams, pq::PQBuildParams},
};
use lance_linalg::distance::MetricType;

fn bench_ivf_pq_index(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        create_file(std::path::Path::new("./vec_data.lance"), WriteMode::Create).await
    });
    let dataset = rt.block_on(async { Dataset::open("./vec_data.lance").await.unwrap() });
    let first_batch = rt.block_on(async {
        dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_next()
            .await
            .unwrap()
            .unwrap()
    });

    let mut rng = rand::rng();
    let vector_column = first_batch.column_by_name("vector").unwrap();
    let value =
        as_fixed_size_list_array(&vector_column).value(rng.random_range(0..vector_column.len()));
    let q: &Float32Array = as_primitive_array(&value);

    c.bench_function(
        format!("Flat_Index(d={},top_k=10,nprobes=10)", q.len()).as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let results = dataset
                    .scan()
                    .nearest("vector", q, 10)
                    .unwrap()
                    .minimum_nprobes(10)
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert!(!results.is_empty());
            })
        },
    );

    c.bench_function(
        format!("Ivf_PQ_Refine(d={},top_k=10,nprobes=10, refine=2)", q.len()).as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let results = dataset
                    .scan()
                    .nearest("vector", q, 10)
                    .unwrap()
                    .minimum_nprobes(10)
                    .refine(2)
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert!(!results.is_empty());
            })
        },
    );

    // reopen with no index caching to test IO overhead
    let dataset = rt.block_on(async {
        DatasetBuilder::from_uri("./vec_data.lance")
            .with_index_cache_size_bytes(0)
            .load()
            .await
            .unwrap()
    });

    c.bench_function(
        format!(
            "Ivf_PQ_NoCache(d={},top_k=10,nprobes=32, refine=1)",
            q.len()
        )
        .as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let results = dataset
                    .scan()
                    .nearest("vector", q, 10)
                    .unwrap()
                    .minimum_nprobes(32)
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert!(!results.is_empty());
            })
        },
    );
}

fn bench_batch_flat_knn(c: &mut Criterion) {
    const DIM: i32 = 512;
    const K: usize = 10;
    const NUM_ROWS: usize = 1_000_000;
    const BATCH_SIZE: usize = 10_000;
    const QUERY_COUNT: usize = 10;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let uri = std::env::temp_dir()
        .join(format!(
            "batch_flat_vec_data_{}.lance",
            rand::random::<u64>()
        ))
        .to_string_lossy()
        .to_string();
    let dataset = rt.block_on(async {
        create_flat_file(&uri, WriteMode::Create, NUM_ROWS, BATCH_SIZE, DIM).await
    });
    let first_batch = rt.block_on(async {
        dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_next()
            .await
            .unwrap()
            .unwrap()
    });
    let vector_column = first_batch.column_by_name("vector").unwrap();
    let vectors = as_fixed_size_list_array(vector_column);
    let query_values = (0..QUERY_COUNT)
        .flat_map(|query_index| {
            let values = vectors.value(query_index);
            as_primitive_array::<arrow_array::types::Float32Type>(&values)
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let queries =
        FixedSizeListArray::try_new_from_values(Float32Array::from(query_values.clone()), DIM)
            .unwrap();

    let mut group = c.benchmark_group("batch_flat_knn");
    group.bench_function(BenchmarkId::new("separate_queries", QUERY_COUNT), |b| {
        b.to_async(&rt).iter(|| async {
            for query_index in 0..QUERY_COUNT {
                let query = Float32Array::from(
                    query_values[query_index * DIM as usize..(query_index + 1) * DIM as usize]
                        .to_vec(),
                );
                let results = dataset
                    .scan()
                    .nearest("vector", &query, K)
                    .unwrap()
                    .use_index(false)
                    .project::<&str>(&[])
                    .unwrap()
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert!(!results.is_empty());
            }
        })
    });
    group.bench_function(BenchmarkId::new("batch_query", QUERY_COUNT), |b| {
        b.to_async(&rt).iter(|| async {
            let results = dataset
                .scan()
                .nearest("vector", &queries, K)
                .unwrap()
                .project::<&str>(&[])
                .unwrap()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert!(!results.is_empty());
        })
    });
    group.finish();
}

async fn create_file(path: &std::path::Path, mode: WriteMode) {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            128,
        ),
        false,
    )]));

    let num_rows = 100_000;
    let batch_size = 10000;
    let batches: Vec<RecordBatch> = (0..(num_rows / batch_size))
        .map(|_| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        create_float32_array(num_rows * 128),
                        128,
                    )
                    .unwrap(),
                )],
            )
            .unwrap()
        })
        .collect();

    let test_uri = path.to_str().unwrap();
    std::fs::remove_dir_all(test_uri).map_or_else(|_| println!("{} not exists", test_uri), |_| {});
    let write_params = WriteParams {
        max_rows_per_file: num_rows as usize,
        max_rows_per_group: batch_size as usize,
        mode,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    let ivf_params = IvfBuildParams {
        num_partitions: Some(32),
        ..Default::default()
    };
    let pq_params = PQBuildParams {
        num_bits: 8,
        num_sub_vectors: 16,
        ..Default::default()
    };
    let m_type = MetricType::L2;
    let params = VectorIndexParams::with_ivf_pq_params(m_type, ivf_params, pq_params);
    dataset
        .create_index(
            vec!["vector"].as_slice(),
            IndexType::Vector,
            Some("ivf_pq_index".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();
}

async fn create_flat_file(
    uri: &str,
    mode: WriteMode,
    num_rows: usize,
    batch_size: usize,
    dim: i32,
) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            dim,
        ),
        false,
    )]));

    struct FlatVectorBatchIter {
        schema: Arc<ArrowSchema>,
        remaining_rows: usize,
        batch_size: usize,
        dim: i32,
    }

    impl Iterator for FlatVectorBatchIter {
        type Item = Result<RecordBatch, ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining_rows == 0 {
                return None;
            }
            let rows = self.remaining_rows.min(self.batch_size);
            self.remaining_rows -= rows;

            let values = create_float32_array(rows * self.dim as usize);
            Some(
                RecordBatch::try_new(
                    self.schema.clone(),
                    vec![Arc::new(
                        FixedSizeListArray::try_new_from_values(values, self.dim).unwrap(),
                    )],
                )
                .map_err(Into::into),
            )
        }
    }

    std::fs::remove_dir_all(uri).map_or_else(|_| println!("{} not exists", uri), |_| {});
    let write_params = WriteParams {
        max_rows_per_file: num_rows,
        max_rows_per_group: batch_size,
        mode,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(
        FlatVectorBatchIter {
            schema: schema.clone(),
            remaining_rows: num_rows,
            batch_size,
            dim,
        },
        schema,
    );
    Dataset::write(reader, uri, Some(write_params))
        .await
        .unwrap()
}

fn create_float32_array(num_elements: usize) -> Float32Array {
    // Generate random values on demand so large benchmark datasets do not need
    // to be fully materialized in memory before writing.
    let mut rng = rand::rng();
    let mut values = Vec::with_capacity(num_elements);
    for _ in 0..num_elements {
        values.push(rng.random_range(0.0..1.0));
    }
    Float32Array::from(values)
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_ivf_pq_index, bench_batch_flat_knn);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_ivf_pq_index, bench_batch_flat_knn);
criterion_main!(benches);
