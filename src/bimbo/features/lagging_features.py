import gc
import logging

import luigi
from luigi.util import requires

import numpy as np
import pandas as pd

from bimbo.preprocess.load_transform import ProcessTrainAndTestData

from .utils import float_to_int


logger = logging.getLogger(__name__)


@requires(ProcessTrainAndTestData)
class BuildTargetLagFeatures(luigi.Task):
    """
    Creates lagging features out of target var for every week

    This task is quite memory intenisve, so a couple of gc.collect()'s are used across the code.
    """
    def run(self):
        data_df = pd.read_hdf(self.input().path)

        index_columns = ['Semana', 'Cliente_ID', 'Producto_ID']
        target_col_name = 'target'
        columns_mask_filter = index_columns + [target_col_name]

        data_df = data_df.loc[:, columns_mask_filter]

        # after filtering we have duplicate rows, need to get rid of them by averaging
        data_df = data_df.groupby(index_columns).mean().reset_index()

        # get back our precious float32
        data_df['Semana'] = data_df['Semana'].astype(np.uint8, copy=False)
        data_df['Cliente_ID'] = data_df['Cliente_ID'].astype(np.uint32, copy=False)
        data_df['Producto_ID'] = data_df['Producto_ID'].astype(np.uint16, copy=False)

        # converting NA to 0 to save some memory
        data_df['target'] = float_to_int(data_df['target'].fillna(0))

        week_start = 7
        week_end = 11
        base_df = data_df[data_df['Semana'] >= week_start]

        gc.collect()

        for lag in range(1, 6):

            logger.info("Lag %d started" % lag)

            agg_column_name = '%sl%s' % (target_col_name, lag)

            # get the view which moves like a window on the data
            shifted_stats = data_df[(week_start - lag <= data_df['Semana']) & (data_df['Semana'] <= week_end - lag)]

            # move data `lag` semanas forward for join
            shifted_stats['Semana'] += lag
            shifted_stats = shifted_stats.rename(columns={target_col_name: agg_column_name})

            # combine on indexes
            base_df = base_df.merge(shifted_stats, how='left', on=index_columns)
            base_df[agg_column_name] = base_df[agg_column_name].fillna(0).astype(np.uint32, copy=False)
            gc.collect()

        base_df.reset_index(inplace=True)
        base_df = base_df[['Semana', 'Cliente_ID', 'Producto_ID',
                           'targetl1', 'targetl2', 'targetl3', 'targetl4', 'targetl5']]
        base_df.to_hdf(self.output().path, 'client_product_semana_features_data')

    def output(self):
        return luigi.LocalTarget("./data/processed/client_product_semana_features.h5")
