import logging

import luigi
from luigi.util import requires

import pandas as pd

from bimbo.preprocess.load_transform import ProcessTrainData

from .utils import float_to_int


logger = logging.getLogger(__name__)


# TODO: functions instead of classes?
@requires(ProcessTrainData)
class BuildProductFeatures(luigi.Task):

    def run(self):
        df_train = pd.read_hdf(self.input().path)
        product_features = df_train[['Producto_ID', 'target']].groupby(['Producto_ID']).mean().reset_index()
        product_features.columns = [product_features.columns[0], 'target_per_product_mean']
        product_features['target_per_product_mean'] = float_to_int(product_features['target_per_product_mean'])
        product_features.to_hdf(self.output().path, 'product_features_data')

    def output(self):
        """
        The task's output
        """
        return luigi.LocalTarget("./data/processed/product_features.h5")


@requires(ProcessTrainData)
class BuildClientFeatures(luigi.Task):

    def run(self):
        df_train = pd.read_hdf(self.input().path)
        client_features = df_train[['Cliente_ID', 'target']].groupby(['Cliente_ID']).mean().reset_index()
        client_features.columns = [client_features.columns[0], 'target_per_client_mean']
        client_features['target_per_product_mean'] = float_to_int(client_features['target_per_client_mean'])
        client_features.to_hdf(self.output().path, 'client_features_data')

    def output(self):
        """
        The task's output
        """
        return luigi.LocalTarget("./data/processed/client_features.h5")


@requires(ProcessTrainData)
class BuildClientProductFeatures(luigi.Task):

    def run(self):
        df_train = pd.read_hdf(self.input().path)

        client_product_features = df_train[['Cliente_ID', 'Producto_ID', 'target']]\
            .groupby(['Cliente_ID', 'Producto_ID'])\
            .mean()\
            .reset_index()

        client_product_features.columns = [
            client_product_features.columns[0],
            client_product_features.columns[1],
            'target_per_client_product_mean'
        ]
        client_product_features['target_per_product_mean'] = \
            float_to_int(client_product_features['target_per_client_product_mean'])

        client_product_features.to_hdf(self.output().path, 'client_product_features_data')

    def output(self):
        """
        The task's output
        """
        return luigi.LocalTarget("./data/processed/client_product_features.h5")
