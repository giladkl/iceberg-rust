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

//! Cache management for Iceberg.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::mem::size_of_val;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};

use bimap::BiHashMap;
use serde_bytes::ByteBuf;

use crate::spec::{Manifest, ManifestList};

/// Bytes a value keeps resident in memory, for weighing it in a cache.
///
/// `Arc` and `Box` count their pointee as if this value owned it alone, so
/// a value shared between several holders is counted once per holder.
pub trait MemorySize {
    /// Bytes this value owns outside its own inline footprint.
    fn heap_size(&self) -> usize;

    /// Inline footprint plus [`heap_size`](Self::heap_size).
    fn memory_size(&self) -> usize {
        size_of_val(self) + self.heap_size()
    }
}

macro_rules! impl_memory_size_no_heap {
    ($($t:ty),* $(,)?) => {
        $(
            impl MemorySize for $t {
                fn heap_size(&self) -> usize {
                    0
                }
            }
        )*
    };
}

impl_memory_size_no_heap!(
    bool, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, usize, f32, f64
);

impl MemorySize for String {
    fn heap_size(&self) -> usize {
        self.capacity()
    }
}

impl MemorySize for ByteBuf {
    fn heap_size(&self) -> usize {
        self.capacity()
    }
}

impl<T: MemorySize> MemorySize for Vec<T> {
    fn heap_size(&self) -> usize {
        self.capacity() * size_of::<T>() + self.iter().map(T::heap_size).sum::<usize>()
    }
}

impl<T: MemorySize> MemorySize for Option<T> {
    fn heap_size(&self) -> usize {
        self.as_ref().map_or(0, T::heap_size)
    }
}

impl<T: MemorySize> MemorySize for Box<T> {
    fn heap_size(&self) -> usize {
        self.as_ref().memory_size()
    }
}

// An `Arc` allocation holds the strong and weak counts ahead of the value.
impl<T: MemorySize> MemorySize for Arc<T> {
    fn heap_size(&self) -> usize {
        2 * size_of::<AtomicUsize>() + self.as_ref().memory_size()
    }
}

impl<T: MemorySize> MemorySize for OnceLock<T> {
    fn heap_size(&self) -> usize {
        self.get().map_or(0, T::heap_size)
    }
}

impl<A: MemorySize, B: MemorySize> MemorySize for (A, B) {
    fn heap_size(&self) -> usize {
        self.0.heap_size() + self.1.heap_size()
    }
}

/// Bytes a hash map's table occupies: hashbrown stores one control byte per
/// bucket alongside the `(K, V)` slots. The entries' own heap is not included.
pub(crate) fn hash_table_size<K, V>(capacity: usize) -> usize {
    capacity * (size_of::<(K, V)>() + 1)
}

impl<K: MemorySize, V: MemorySize, S> MemorySize for HashMap<K, V, S> {
    fn heap_size(&self) -> usize {
        hash_table_size::<K, V>(self.capacity())
            + self
                .iter()
                .map(|(k, v)| k.heap_size() + v.heap_size())
                .sum::<usize>()
    }
}

impl<T: MemorySize, S> MemorySize for HashSet<T, S> {
    fn heap_size(&self) -> usize {
        hash_table_size::<T, ()>(self.capacity()) + self.iter().map(T::heap_size).sum::<usize>()
    }
}

// `BiHashMap` keeps each side in an `Rc` and indexes it from two hash maps.
impl<L: MemorySize + Eq + Hash, R: MemorySize + Eq + Hash> MemorySize for BiHashMap<L, R> {
    fn heap_size(&self) -> usize {
        2 * hash_table_size::<usize, usize>(self.capacity())
            + self
                .iter()
                .map(|(l, r)| 4 * size_of::<usize>() + l.memory_size() + r.memory_size())
                .sum::<usize>()
    }
}

/// A trait for caching in-memory objects of given type.
///
/// # Notes
///
/// ObjectCache will store deeply nested objects, such as `Manifest`,
/// which contains `Schema`. Please ensure that the cache stores the
/// object in memory as-is, without attempting to serialize it, as
/// serialization could be extremely expensive.
pub trait ObjectCache<K, V>: Send + Sync {
    /// Gets an object from the cache by its key.
    fn get(&self, key: &K) -> Option<V>;
    /// Sets an object in the cache with the given key and value.
    fn set(&self, key: K, value: V);
}

/// A trait for caching different in-memory objects used by iceberg.
///
/// # Notes
///
/// ObjectCache will store deeply nested objects, such as `Manifest`,
/// which contains `Schema`. Please ensure that the cache stores the
/// object in memory as-is, without attempting to serialize it, as
/// serialization could be extremely expensive.
pub trait ObjectCacheProvide: Send + Sync {
    /// Gets a cache for manifests.
    fn manifest_cache(&self) -> &dyn ObjectCache<String, Arc<Manifest>>;
    /// Gets a cache for manifest lists.
    fn manifest_list_cache(&self) -> &dyn ObjectCache<String, Arc<ManifestList>>;
}

/// CacheProvider is a type alias for a thread-safe reference-counted pointer to a CacheProvide trait object.
pub type ObjectCacheProvider = Arc<dyn ObjectCacheProvide>;

#[cfg(test)]
mod tests {
    use super::*;

    struct _TestDynCompatibleForObjectCache(Arc<dyn ObjectCache<String, Arc<Manifest>>>);
    struct _TestDynCompatibleForObjectCacheProvider(ObjectCacheProvider);

    #[test]
    fn test_vec_counts_its_buffer_and_each_element_heap() {
        let strings = vec![String::from("abc"), String::with_capacity(10)];

        assert_eq!(
            strings.heap_size(),
            strings.capacity() * size_of::<String>() + 3 + 10
        );
        assert_eq!(
            strings.memory_size(),
            size_of::<Vec<String>>() + strings.heap_size()
        );
    }

    #[test]
    fn test_arc_and_box_count_their_pointee() {
        let text = String::from("hello");
        let inline = size_of::<String>() + 5;

        assert_eq!(Box::new(text.clone()).heap_size(), inline);
        assert_eq!(
            Arc::new(text).heap_size(),
            2 * size_of::<AtomicUsize>() + inline
        );
    }

    #[test]
    fn test_option_and_once_lock_count_only_a_present_value() {
        assert_eq!(None::<String>.heap_size(), 0);
        assert_eq!(Some(String::from("ab")).heap_size(), 2);
        assert_eq!(OnceLock::<String>::new().heap_size(), 0);
    }

    #[test]
    fn test_hash_map_counts_table_and_entry_heap() {
        let mut map = HashMap::new();
        map.insert(1i32, String::from("abcd"));

        assert_eq!(
            map.heap_size(),
            map.capacity() * (size_of::<(i32, String)>() + 1) + 4
        );
    }
}
