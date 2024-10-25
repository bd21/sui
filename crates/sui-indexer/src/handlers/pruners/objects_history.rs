// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use diesel_async::RunQueryDsl;

use crate::{database::Connection, handlers::pruner::PrunableTable};

use super::{get_partition_sql, PartitionedTable, Prunable};

pub struct ObjectsHistory;

#[async_trait::async_trait]
impl Prunable for ObjectsHistory {
    const NAME: PrunableTable = PrunableTable::ObjectsHistory;

    const CHUNK_SIZE: u64 = 1;

    async fn data_lo(conn: &mut Connection<'_>) -> anyhow::Result<u64> {
        println!("trying to get data_lo");
        let result = diesel::sql_query(get_partition_sql(Self::NAME.as_ref()))
            .get_result::<PartitionedTable>(conn)
            .await
            .map(|entry| entry.first_partition as u64)
            .context("failed to get first partition");

        println!("result: {:?}", result);
        println!("got data_lo");
        result
    }

    async fn prune(
        prune_lo: u64,
        _prune_hi: u64,
        conn: &mut Connection<'_>,
    ) -> anyhow::Result<usize> {
        diesel::sql_query("CALL drop_partition($1, $2)")
            .bind::<diesel::sql_types::Text, _>(Self::NAME.to_string())
            .bind::<diesel::sql_types::BigInt, _>(prune_lo as i64)
            .execute(conn)
            .await
            .context("failed to drop partition")
    }
}

// TODO: I think for pruner we just need to know FANOUT, and how many to delete at once
// I think generally, unpartitioned tables can have CHUNK_SIZE=100k, and partitioned tables would be 1 at a time
