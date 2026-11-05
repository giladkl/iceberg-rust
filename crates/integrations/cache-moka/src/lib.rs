// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::hash::Hash;
use std::sync::Arc;

use iceberg::cache::{MemorySize, ObjectCache, ObjectCacheProvide};
use iceberg::spec::{Manifest, ManifestList};

const DEFAULT_CACHE_SIZE_BYTES: u64 = 32 * 1024 * 1024; // 32MiB

struct MokaObjectCache<K, V>(moka::sync::Cache<K, V>);

impl<K, V> MokaObjectCache<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: MemorySize + Clone + Send + Sync + 'static,
{
    fn bounded_by_bytes(max_bytes: u64) -> Self {
        Self(
            moka::sync::Cache::builder()
                .weigher(|_, value: &V| u32::try_from(value.memory_size()).unwrap_or(u32::MAX))
                .max_capacity(max_bytes)
                .build(),
        )
    }
}

impl<K, V> ObjectCache<K, V> for MokaObjectCache<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        self.0.get(key)
    }

    fn set(&self, key: K, value: V) {
        self.0.insert(key, value);
    }
}

/// A cache provider that uses Moka for caching objects.
pub struct MokaObjectCacheProvider {
    manifest_cache: MokaObjectCache<String, Arc<Manifest>>,
    manifest_list_cache: MokaObjectCache<String, Arc<ManifestList>>,
}

impl Default for MokaObjectCacheProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MokaObjectCacheProvider {
    /// Creates a new `MokaObjectCacheProvider` with default cache sizes.
    pub fn new() -> Self {
        let manifest_cache = MokaObjectCache::bounded_by_bytes(DEFAULT_CACHE_SIZE_BYTES);
        let manifest_list_cache = MokaObjectCache::bounded_by_bytes(DEFAULT_CACHE_SIZE_BYTES);

        Self {
            manifest_cache,
            manifest_list_cache,
        }
    }

    /// Set the cache for manifests.
    pub fn with_manifest_cache(mut self, cache: moka::sync::Cache<String, Arc<Manifest>>) -> Self {
        self.manifest_cache = MokaObjectCache(cache);
        self
    }

    /// Set the cache for manifest lists.
    pub fn with_manifest_list_cache(
        mut self,
        cache: moka::sync::Cache<String, Arc<ManifestList>>,
    ) -> Self {
        self.manifest_list_cache = MokaObjectCache(cache);
        self
    }
}

impl ObjectCacheProvide for MokaObjectCacheProvider {
    fn manifest_cache(&self) -> &dyn ObjectCache<String, Arc<Manifest>> {
        &self.manifest_cache
    }

    fn manifest_list_cache(&self) -> &dyn ObjectCache<String, Arc<ManifestList>> {
        &self.manifest_list_cache
    }
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestContentType,
        ManifestEntry, ManifestMetadata, ManifestStatus, NestedField, PartitionSpec, PrimitiveType,
        Schema, Struct, Type,
    };

    use super::*;

    fn manifest_with_entries(entry_count: usize) -> Manifest {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .unwrap(),
        );
        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(PartitionSpec::unpartition_spec())
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();
        let entries = (0..entry_count)
            .map(|i| {
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("s3://bucket/{i}.parquet"))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(100)
                            .record_count(1)
                            .partition(Struct::empty())
                            .build()
                            .unwrap(),
                    )
                    .build()
            })
            .collect();
        Manifest::new(metadata, entries)
    }

    #[test]
    fn test_default_manifest_cache_weighs_entries_by_memory_size() {
        let provider = MokaObjectCacheProvider::new();
        let manifest = Arc::new(manifest_with_entries(100));

        provider
            .manifest_cache()
            .set("manifest".to_string(), manifest.clone());
        provider.manifest_cache.0.run_pending_tasks();

        assert_eq!(
            provider.manifest_cache.0.weighted_size(),
            manifest.memory_size() as u64
        );
    }

    #[test]
    fn test_byte_bound_cache_evicts_when_entries_outweigh_capacity() {
        let manifest = Arc::new(manifest_with_entries(100));
        let cache = MokaObjectCache::<String, Arc<Manifest>>::bounded_by_bytes(
            manifest.memory_size() as u64 * 2 - 1,
        );

        cache.set("a".to_string(), manifest.clone());
        cache.set("b".to_string(), manifest.clone());
        cache.0.run_pending_tasks();

        assert_eq!(cache.0.entry_count(), 1);
    }
}
