from functools import partial

import luigi
from luigi.util import requires

import numpy as np
import pandas as pd

from sklearn.datasets import dump_svmlight_file


FIELD_TYPES = {
    'id': np.uint32,

    'Semana': np.uint8,
    'Agencia_ID': np.uint16,
    'Canal_ID': np.uint8,
    'Ruta_SAK': np.uint16,
    'Cliente_ID': np.uint32,
    'Producto_ID': np.uint16,

    'Demanda_uni_equil': np.uint32
}

TRAIN_LOAD_COLUMNS = ["Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID", "Demanda_uni_equil"]
TEST_LOAD_COLUMNS = ["id", "Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID"]


read_csv = partial(pd.read_csv, dtype=FIELD_TYPES)


class ProcessTrainData(luigi.Task):
    """
    Save training data to hdf
    """
    def run(self):
        train_data = read_csv("data/input/train_small.csv", usecols=TRAIN_LOAD_COLUMNS)

        # rename for clarity
        train_data.rename(columns={'Demanda_uni_equil': 'target'}, inplace=True)
        train_data['target'] = np.log1p(train_data['target']).astype(np.float32, copy=False)

        train_data.to_hdf(self.output().path, 'train_data')

    def output(self):
        return luigi.LocalTarget("./data/processed/train.h5")


class ProcessTestData(luigi.Task):
    """
    Save test data to hdf
    """
    def run(self):
        test_data = read_csv("data/input/test.csv", usecols=TEST_LOAD_COLUMNS)
        test_data.to_hdf(self.output().path, 'test_data')

    def output(self):
        return luigi.LocalTarget("./data/processed/test.h5")


class ProcessTrainAndTestData(luigi.Task):
    """
    Join both train and test datasets into one for easier processing on some tasks
    """
    def requires(self):
        return {
            'train': ProcessTrainData(),
            'test': ProcessTestData()
        }

    def run(self):
        df_train = pd.read_hdf(self.input()['train'].path)
        df_test = pd.read_hdf(self.input()['test'].path)
        all_data = df_train.append(df_test)
        all_data.to_hdf(self.output().path, 'train_test_data')

    def output(self):
        return luigi.LocalTarget("./data/processed/train_and_test.h5")


@requires(ProcessTrainData)
class ConvertTrainAndTestDataToSVM(luigi.Task):
    """
    Converts data to LibSVM format
    """
    def run(self):
        all_data = pd.read_hdf(self.input().path)

        X = all_data[np.setdiff1d(all_data.columns, ['target'])]  # NOQA
        y = all_data['target']
        dump_svmlight_file(X, y, self.output().path)

    def output(self):
        return luigi.LocalTarget("./data/processed/train.svm")


@requires(ProcessTrainData)
class ConvertTrainAndTestDataToParquet(luigi.Task):
    """
    Converts data to parquet for later reuse
    """
    def run(self):
        """
        Runs only on an instance with Spark installed
        """
        from pyspark.sql import SparkSession
        from pyspark.sql.types import StructType, StructField, ByteType, ShortType, IntegerType

        spark = SparkSession\
            .builder\
            .appName("BimboConverter")\
            .config("spark.executor.memory", "8g")\
            .getOrCreate()

        schema = StructType([
            StructField('Semana', ByteType(), True),
            StructField('Agencia_ID', ShortType(), True),
            StructField('Canal_ID', ShortType(), True),
            StructField('Ruta_SAK', IntegerType(), True),
            StructField('Cliente_ID', IntegerType(), True),
            StructField('Producto_ID', IntegerType(), True),
            StructField('Demanda_uni_equil', IntegerType(), True)
        ])

        all_data = spark.read.csv("./data/input/train.csv", header=True, schema=schema)
        all_data = all_data.withColumnRenamed("Demanda_uni_equil", "target")
        all_data.write.parquet(self.output().path)

    def output(self):
        return luigi.LocalTarget("./data/processed/train.parquet")
