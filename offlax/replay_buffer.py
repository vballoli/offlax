from typing import Dict, List

import numpy as np
from petastorm.codecs import CompressedImageCodec, NdarrayCodec, ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType


class ReplayBuffer:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

    def get_schema(self):
        return Unischema(
            "replay_buffer_schema",
            [
                UnischemaField(
                    "observations",
                    self.observations.dtype,
                    self.observations.shape[1:],
                    NdarrayCodec(),
                    False,
                ),
                UnischemaField(
                    "actions",
                    self.actions.dtype,
                    self.actions.shape[1:],
                    NdarrayCodec(),
                    False,
                ),
                UnischemaField(
                    "rewards",
                    self.rewards.dtype,
                    self.rewards.shape[1:],
                    NdarrayCodec(),
                    False,
                ),
                UnischemaField(
                    "dones",
                    self.dones.dtype,
                    self.dones.shape[1:],
                    NdarrayCodec(),
                    False,
                ),
            ],
        )

    def dump(self, filepath: str):
        schema = self.get_schema()

        spark = SparkSession.builder.master("local[2]").getOrCreate()
        sc = spark.sparkContext

        with materialize_dataset(spark, filepath, schema):
            rows = (
                sc.parallelize(range(self.observations.shape[0]))
                .map(self.query)
                .map(lambda x: dict_to_spark_row(schema, x))
            )

            spark.createDataFrame(rows, schema.as_spark_schema()).coalesce(
                10
            ).write.mode("overwrite").parquet(filepath)

    def query(self, idx):
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }
