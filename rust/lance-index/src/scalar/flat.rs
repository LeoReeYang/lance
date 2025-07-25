// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::{any::Any, ops::Bound, sync::Arc};

use arrow_array::{
    cast::AsArray, types::UInt64Type, ArrayRef, BooleanArray, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;

use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_physical_expr::expressions::{in_list, lit, Column};
use deepsize::DeepSizeOf;
use lance_core::cache::LanceCache;
use lance_core::utils::address::RowAddress;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use snafu::location;

use super::{btree::BTreeSubIndex, IndexStore, ScalarIndex};
use super::{AnyQuery, MetricsCollector, SargableQuery, SearchResult};
use crate::frag_reuse::FragReuseIndex;
use crate::{Index, IndexType};

/// A flat index is just a batch of value/row-id pairs
///
/// The batch always has two columns.  The first column "values" contains
/// the values.  The second column "row_ids" contains the row ids
///
/// Evaluating a query requires O(N) time where N is the # of rows
#[derive(Debug)]
pub struct FlatIndex {
    data: Arc<RecordBatch>,
    has_nulls: bool,
}

impl DeepSizeOf for FlatIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.data.get_array_memory_size()
    }
}

impl FlatIndex {
    fn values(&self) -> &ArrayRef {
        self.data.column(0)
    }

    fn ids(&self) -> &ArrayRef {
        self.data.column(1)
    }
}

fn remap_batch(batch: RecordBatch, mapping: &HashMap<u64, Option<u64>>) -> Result<RecordBatch> {
    let row_ids = batch.column(1).as_primitive::<UInt64Type>();
    let val_idx_and_new_id = row_ids
        .values()
        .iter()
        .enumerate()
        .filter_map(|(idx, old_id)| {
            mapping
                .get(old_id)
                .copied()
                .unwrap_or(Some(*old_id))
                .map(|new_id| (idx, new_id))
        })
        .collect::<Vec<_>>();
    let new_ids = Arc::new(UInt64Array::from_iter_values(
        val_idx_and_new_id.iter().copied().map(|(_, new_id)| new_id),
    ));
    let new_val_indices = UInt64Array::from_iter_values(
        val_idx_and_new_id
            .into_iter()
            .map(|(val_idx, _)| val_idx as u64),
    );
    let new_vals = arrow_select::take::take(batch.column(0), &new_val_indices, None)?;
    Ok(RecordBatch::try_new(
        batch.schema(),
        vec![new_vals, new_ids],
    )?)
}

/// Trains a flat index from a record batch of values & ids by simply storing the batch
///
/// This allows the flat index to be used as a sub-index
#[derive(Debug)]
pub struct FlatIndexMetadata {
    schema: Arc<Schema>,
}

impl DeepSizeOf for FlatIndexMetadata {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.schema.metadata.deep_size_of_children(context)
            + self
                .schema
                .fields
                .iter()
                // This undercounts slightly because it doesn't account for the size of the
                // field data types
                .map(|f| {
                    std::mem::size_of::<Field>()
                        + f.name().deep_size_of_children(context)
                        + f.metadata().deep_size_of_children(context)
                })
                .sum::<usize>()
    }
}

impl FlatIndexMetadata {
    pub fn new(value_type: DataType) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", value_type, true),
            Field::new("row_ids", DataType::UInt64, true),
        ]));
        Self { schema }
    }
}

#[async_trait]
impl BTreeSubIndex for FlatIndexMetadata {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    async fn train(&self, batch: RecordBatch) -> Result<RecordBatch> {
        // The data source may not call the columns "values" and "row_ids" so we need to replace
        // the schema
        Ok(RecordBatch::try_new(
            self.schema.clone(),
            vec![batch.column(0).clone(), batch.column(1).clone()],
        )?)
    }

    async fn load_subindex(&self, serialized: RecordBatch) -> Result<Arc<dyn ScalarIndex>> {
        let has_nulls = serialized.column(0).null_count() > 0;
        Ok(Arc::new(FlatIndex {
            data: Arc::new(serialized),
            has_nulls,
        }))
    }

    async fn remap_subindex(
        &self,
        serialized: RecordBatch,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RecordBatch> {
        remap_batch(serialized, mapping)
    }

    async fn retrieve_data(&self, serialized: RecordBatch) -> Result<RecordBatch> {
        Ok(serialized)
    }
}

#[async_trait]
impl Index for FlatIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "FlatIndex is not vector index".into(),
            location: location!(),
        })
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    async fn prewarm(&self) -> Result<()> {
        // There is nothing to pre-warm
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "num_values": self.data.num_rows(),
        }))
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = self
            .ids()
            .as_primitive::<UInt64Type>()
            .iter()
            .map(|row_id| RowAddress::from(row_id.unwrap()).fragment_id())
            .collect::<Vec<_>>();
        frag_ids.sort();
        frag_ids.dedup();
        Ok(RoaringBitmap::from_sorted_iter(frag_ids).unwrap())
    }
}

#[async_trait]
impl ScalarIndex for FlatIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        metrics.record_comparisons(self.data.num_rows());
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        // Since we have all the values in memory we can use basic arrow-rs compute
        // functions to satisfy scalar queries.
        let mut predicate = match query {
            SargableQuery::Equals(value) => {
                if value.is_null() {
                    arrow::compute::is_null(self.values())?
                } else {
                    arrow_ord::cmp::eq(self.values(), &value.to_scalar()?)?
                }
            }
            SargableQuery::IsNull() => arrow::compute::is_null(self.values())?,
            SargableQuery::IsIn(values) => {
                let mut has_null = false;
                let choices = values
                    .iter()
                    .map(|val| {
                        has_null |= val.is_null();
                        lit(val.clone())
                    })
                    .collect::<Vec<_>>();
                let in_list_expr = in_list(
                    Arc::new(Column::new("values", 0)),
                    choices,
                    &false,
                    &self.data.schema(),
                )?;
                let result_col = in_list_expr.evaluate(&self.data)?;
                let predicate = result_col
                    .into_array(self.data.num_rows())?
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("InList evaluation should return boolean array")
                    .clone();

                // Arrow's in_list does not handle nulls so we need to join them in here if user asked for them
                if has_null && self.has_nulls {
                    let nulls = arrow::compute::is_null(self.values())?;
                    arrow::compute::or(&predicate, &nulls)?
                } else {
                    predicate
                }
            }
            SargableQuery::Range(lower_bound, upper_bound) => match (lower_bound, upper_bound) {
                (Bound::Unbounded, Bound::Unbounded) => {
                    panic!("Scalar range query received with no upper or lower bound")
                }
                (Bound::Unbounded, Bound::Included(upper)) => {
                    arrow_ord::cmp::lt_eq(self.values(), &upper.to_scalar()?)?
                }
                (Bound::Unbounded, Bound::Excluded(upper)) => {
                    arrow_ord::cmp::lt(self.values(), &upper.to_scalar()?)?
                }
                (Bound::Included(lower), Bound::Unbounded) => {
                    arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar()?)?
                }
                (Bound::Included(lower), Bound::Included(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt_eq(self.values(), &upper.to_scalar()?)?,
                )?,
                (Bound::Included(lower), Bound::Excluded(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt(self.values(), &upper.to_scalar()?)?,
                )?,
                (Bound::Excluded(lower), Bound::Unbounded) => {
                    arrow_ord::cmp::gt(self.values(), &lower.to_scalar()?)?
                }
                (Bound::Excluded(lower), Bound::Included(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt_eq(self.values(), &upper.to_scalar()?)?,
                )?,
                (Bound::Excluded(lower), Bound::Excluded(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt(self.values(), &upper.to_scalar()?)?,
                )?,
            },
            SargableQuery::FullTextSearch(_) => return Err(Error::invalid_input(
                "full text search is not supported for flat index, build a inverted index for it",
                location!(),
            )),
        };
        if self.has_nulls && matches!(query, SargableQuery::Range(_, _)) {
            // Arrow's comparison kernels do not return false for nulls.  They consider nulls to
            // be less than any value.  So we need to filter out the nulls manually.
            let valid_values = arrow::compute::is_not_null(self.values())?;
            predicate = arrow::compute::and(&valid_values, &predicate)?;
        }
        let matching_ids = arrow_select::filter::filter(self.ids(), &predicate)?;
        let matching_ids = matching_ids
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Result of arrow_select::filter::filter did not match input type");
        Ok(SearchResult::Exact(RowIdTreeMap::from_iter(
            matching_ids.values(),
        )))
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        true
    }

    // Note that there is no write/train method for flat index at the moment and so it isn't
    // really possible for this method to be called.  If there was we assume it will write all
    // data as a single batch named data.lance
    async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        _index_cache: LanceCache,
    ) -> Result<Arc<Self>> {
        let batches = store.open_index_file("data.lance").await?;
        let num_rows = batches.num_rows();
        let mut batch = batches.read_range(0..num_rows, None).await?;
        if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            batch = frag_reuse_index_ref.remap_row_ids_record_batch(batch, 1)?;
        }
        let has_nulls = batch.column(0).null_count() > 0;
        Ok(Arc::new(Self {
            data: Arc::new(batch),
            has_nulls,
        }))
    }

    // Same as above, this is dead code at the moment but should work
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let remapped = remap_batch((*self.data).clone(), mapping)?;
        let mut writer = dest_store
            .new_index_file("data.lance", remapped.schema())
            .await?;
        writer.write_record_batch(remapped).await?;
        writer.finish().await?;
        Ok(())
    }

    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // If this was desired, then you would need to merge new_data and data and write it back out
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::metrics::NoOpMetricsCollector;

    use super::*;
    use arrow_array::types::Int32Type;
    use datafusion_common::ScalarValue;
    use lance_datagen::{array, gen, RowCount};

    fn example_index() -> FlatIndex {
        let batch = gen()
            .col(
                "values",
                array::cycle::<Int32Type>(vec![10, 100, 1000, 1234]),
            )
            .col("ids", array::cycle::<UInt64Type>(vec![5, 0, 3, 100]))
            .into_batch_rows(RowCount::from(4))
            .unwrap();

        FlatIndex {
            data: Arc::new(batch),
            has_nulls: false,
        }
    }

    async fn check_index(query: &SargableQuery, expected: &[u64]) {
        let index = example_index();
        let actual = index.search(query, &NoOpMetricsCollector).await.unwrap();
        let SearchResult::Exact(actual_row_ids) = actual else {
            panic! {"Expected exact search result"}
        };
        let expected = RowIdTreeMap::from_iter(expected);
        assert_eq!(actual_row_ids, expected);
    }

    #[tokio::test]
    async fn test_equality() {
        check_index(&SargableQuery::Equals(ScalarValue::from(100)), &[0]).await;
        check_index(&SargableQuery::Equals(ScalarValue::from(10)), &[5]).await;
        check_index(&SargableQuery::Equals(ScalarValue::from(5)), &[]).await;
    }

    #[tokio::test]
    async fn test_range() {
        check_index(
            &SargableQuery::Range(
                Bound::Included(ScalarValue::from(100)),
                Bound::Excluded(ScalarValue::from(1234)),
            ),
            &[0, 3],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Unbounded, Bound::Excluded(ScalarValue::from(1000))),
            &[5, 0],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Included(ScalarValue::from(0)), Bound::Unbounded),
            &[5, 0, 3, 100],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Included(ScalarValue::from(100000)), Bound::Unbounded),
            &[],
        )
        .await;
    }

    #[tokio::test]
    async fn test_is_in() {
        check_index(
            &SargableQuery::IsIn(vec![
                ScalarValue::from(100),
                ScalarValue::from(1234),
                ScalarValue::from(3000),
            ]),
            &[0, 100],
        )
        .await;
    }

    #[tokio::test]
    async fn test_remap() {
        let index = example_index();
        // 0 -> 2000
        // 3 -> delete
        // Keep remaining as is
        let mapping = HashMap::<u64, Option<u64>>::from_iter(vec![(0, Some(2000)), (3, None)]);
        let metadata = FlatIndexMetadata::new(DataType::Int32);
        let remapped = metadata
            .remap_subindex((*index.data).clone(), &mapping)
            .await
            .unwrap();

        let expected = gen()
            .col("values", array::cycle::<Int32Type>(vec![10, 100, 1234]))
            .col("ids", array::cycle::<UInt64Type>(vec![5, 2000, 100]))
            .into_batch_rows(RowCount::from(3))
            .unwrap();
        assert_eq!(remapped, expected);
    }

    // It's possible, during compaction, that an entire page of values is deleted.  We just serialize
    // it as an empty record batch.
    #[tokio::test]
    async fn test_remap_to_nothing() {
        let index = example_index();
        let mapping = HashMap::<u64, Option<u64>>::from_iter(vec![
            (5, None),
            (0, None),
            (3, None),
            (100, None),
        ]);
        let metadata = FlatIndexMetadata::new(DataType::Int32);
        let remapped = metadata
            .remap_subindex((*index.data).clone(), &mapping)
            .await
            .unwrap();
        assert_eq!(remapped.num_rows(), 0);
    }
}
