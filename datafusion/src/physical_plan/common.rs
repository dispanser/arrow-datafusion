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

//! Defines common code used in execution plans

use std::fs;
use std::fs::metadata;
use std::path::Path;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::{RecordBatchStream, SendableRecordBatchStream};
use crate::error::{DataFusionError, Result};

use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use futures::{Stream, TryStreamExt};

/// Stream of record batches
pub struct SizedRecordBatchStream {
    schema: SchemaRef,
    batches: Vec<Arc<RecordBatch>>,
    index: usize,
}

impl SizedRecordBatchStream {
    /// Create a new RecordBatchIterator
    pub fn new(schema: SchemaRef, batches: Vec<Arc<RecordBatch>>) -> Self {
        SizedRecordBatchStream {
            schema,
            index: 0,
            batches,
        }
    }
}

impl Stream for SizedRecordBatchStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Poll::Ready(if self.index < self.batches.len() {
            self.index += 1;
            Some(Ok(self.batches[self.index - 1].as_ref().clone()))
        } else {
            None
        })
    }
}

impl RecordBatchStream for SizedRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Create a vector of record batches from a stream
pub async fn collect(stream: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
    stream
        .try_collect::<Vec<_>>()
        .await
        .map_err(DataFusionError::from)
}

/// Recursively build a list of files in a directory with a given extension
pub fn build_file_list(dir: &str, filenames: &mut Vec<String>, ext: &str) -> Result<()> {
    let metadata = metadata(dir)?;
    if metadata.is_file() {
        if dir.ends_with(ext) {
            filenames.push(dir.to_string());
        }
    } else {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(path_name) = path.to_str() {
                if path.is_dir() {
                    build_file_list(path_name, filenames, ext)?;
                } else if path_name.ends_with(ext) {
                    filenames.push(path_name.to_string());
                }
            } else {
                return Err(DataFusionError::Plan("Invalid path".to_string()));
            }
        }
    }
    Ok(())
}

/// A partition with the files located inside the partition leaf directory.
/// TODO: the actual partition data is not yet modelled.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct PartitionData {
    /// sequence of partition values (in order of occurrence on the path to the file
    partition_values: Vec<String>,

    /// files belonging to the same partition
    files: Vec<String>,
}

fn split_partition(name: &str) -> Option<(&str, &str)> {
    if let Some(idx) = name.find('=') {
        Some((&name[0..idx], &name[idx + 1..]))
    } else {
        None
    }
}

/// result of scanning a directory tree for files
pub struct PartitionScan {
    /// the schema contains information *only* for the partition columns,
    /// not for the data actually sitting inside the parquet files.
    schema: SchemaRef,

    /// the discovered (relevant) partitions
    partitions: Vec<PartitionData>,

    /// table root directory / root of scan operation.
    root_dir: String,
}

/// recursively scan a directory tree, extracting the partition schema and related files
pub fn scan_partition_tree(dir: &Path, ext: &str) -> Result<PartitionScan> {
    let metadata = metadata(dir)?;
    if metadata.is_file() {
        let file_names = dir
            .file_name()
            .and_then(|f| f.to_str())
            .map(|f| vec![f.to_string()])
            .unwrap_or_default();
        let partition_data = PartitionData {
            partition_values: vec![],
            files: file_names,
        };
        Ok(PartitionScan {
            schema: Arc::new(Schema::empty()),
            partitions: vec![partition_data],
            // may be OK because we already know it's a file.
            root_dir: dir
                .parent()
                .and_then(|f| f.to_str())
                .unwrap_or("")
                .to_string(),
        })
    } else {
        let mut matching_files: Vec<String> = vec![];
        for entry in fs::read_dir(dir)? {
            let entry = &entry?;
            let file_type: &fs::FileType = &entry.file_type()?;
            match file_type {
                file_type if file_type.is_dir() => println!("dir: {:?}", entry),
                file_type
                    if file_type.is_file()
                        && entry.path().to_str().unwrap().ends_with(ext) =>
                {
                    matching_files.push(
                        entry
                            .path()
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_string(),
                    )
                }
                _ => println!("unknown: {:?}", entry),
            }
        }
        println!("{:?}", matching_files);
        Ok(PartitionScan {
            schema: Arc::new(Schema::empty()),
            partitions: vec![PartitionData {
                partition_values: vec![],
                files: matching_files,
            }],
            root_dir: dir.to_str().map_or("".to_string(), |f| f.to_string()),
        })
    }
}

// #[cfg(test)]
mod tests {

    use super::*;
    use arrow::datatypes::{DataType, Field};

    #[test]
    fn scan_flat_parquet() {
        // no partitions, so we expect an empty column schema, and a single partition w/ all files
        let root_dir = Path::new(
            "/home/data/study/rust/arrow/partitioned-dataframe-testdata/unpartitioned",
        );
        if let Ok(partition_scan) = scan_partition_tree(&root_dir, ".parquet") {
            assert_eq!(*partition_scan.schema, Schema::empty());
            let partition_values = &partition_scan.partitions[0].partition_values;
            assert!(partition_values.is_empty());
            let expected_files = (1..10)
                .map(|i| format!("{}.parquet", i))
                .collect::<Vec<String>>();
            let file_names = &mut partition_scan.partitions[0].files.clone();
            file_names.sort();
            assert_eq!(file_names, &expected_files);
        } else {
            panic!("scan of parquet directory failed");
        }
    }

    #[test]
    fn scan_single_file() {
        let root_dir =
            "/home/data/study/rust/arrow/partitioned-dataframe-testdata/unpartitioned";
        let file_name = "3.parquet";
        let file_path = format!("{}/{}", root_dir, file_name);
        if let Ok(partition_scan) = scan_partition_tree(Path::new(&file_path), ".parquet")
        {
            assert_eq!(*partition_scan.schema, Schema::empty());
            let expected_partition = PartitionData {
                partition_values: vec![],
                files: vec![file_name.to_string()],
            };
            assert_eq!(vec![expected_partition], partition_scan.partitions);
            assert_eq!(partition_scan.root_dir, root_dir);
        } else {
            panic!("scan of single parquet file failed");
        }
    }

    #[test]
    fn scan_single_partition_layer() {
        let root_dir = "/home/data/study/rust/arrow/partitioned-dataframe-testdata/month-partitioned";
        if let Ok(partition_scan) = scan_partition_tree(Path::new(&root_dir), ".parquet") {
            panic!("scan of partitioned directory tree failed totally");
        } else {
            panic!("scan of partitioned directory tree failed");
        }
        // assert_eq!(*partition_scan.schema, Schema::new(vec![Field::new("date", DataType::Utf8, true)]));
    }

    #[test]
    fn scan_contradicting_schemas() {}
}
