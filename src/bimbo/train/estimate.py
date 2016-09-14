import luigi

import numpy as np
import pandas as pd


import xgboost as xgb

from bimbo.features.lagging_features import BuildTargetLagFeatures
from bimbo.preprocess.load_transform import ProcessTestData
from bimbo.train.learn import GBTLearn


DEFAULT_FEATURES = ["Semana", "Agencia_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID",
                    "targetl1", "targetl2", "targetl3", "targetl4", "targetl5"]


class GBTEstimate(luigi.Task):
    """
    Creates a submission
    """
    week_first = luigi.IntParameter(default=10, description="Defines first week for predictions (out of two)")
    week_second = luigi.IntParameter(default=10, description="Defines second week for predictions (out of two)")
    features = luigi.ListParameter(default=DEFAULT_FEATURES, description="Features to use for training")

    def requires(self):
        return {
            'model': GBTLearn(),
            'test': ProcessTestData(),
            'client_product_semana_features': BuildTargetLagFeatures()
        }

    def run(self):
        """
        Note that the following code is based on https://www.kaggle.com/bpavlyshenko's.
        """
        # model
        xgb_fit = xgb.Booster()
        xgb_fit.load_model(self.input()['model'].path)

        # features merge
        # TODO: easy way to experiment with additional features
        data_test = pd.read_hdf(self.input()['test'].path)
        data_lagged_features = pd.read_hdf(self.input()['client_product_semana_features'].path)
        data_test = data_test.merge(data_lagged_features, how='left', on=["Cliente_ID", "Producto_ID", "Semana"])

        # Predict 10th week
        data_test1 = data_test[data_test['Semana'] == self.week_first]
        pred = xgb_fit.predict(xgb.DMatrix(data_test1[self.features], missing=np.nan))
        res = np.expm1(pred)

        # Lagged features for 11th week
        data_test_lag1 = data_test1[['Cliente_ID', 'Producto_ID']]
        data_test_lag1['targetl1'] = res
        data_test_lag1 = data_test_lag1.groupby(['Cliente_ID', 'Producto_ID']).agg({'targetl1': np.mean})
        data_test_lag1 = pd.DataFrame(data_test_lag1.to_records())
        data_test_lag1.columns = ['Cliente_ID', 'Producto_ID', 'targetl1']

        # Semana 11
        results = pd.DataFrame(dict(id=data_test1['id'], Demanda_uni_equil=res))

        data_test2 = data_test[data_test['Semana'] == self.week_second].copy()
        data_test2.drop('targetl1', inplace=True, axis=1)

        # Merge lagged values
        data_test2 = pd.merge(data_test2, data_test_lag1, how='left', on=['Cliente_ID', 'Producto_ID'])
        pred = xgb_fit.predict(xgb.DMatrix(data_test2[self.features], missing=np.nan))
        res = np.expm1(pred)

        res_df = pd.DataFrame(dict(id=data_test2['id'], Demanda_uni_equil=res))
        results = results.append(res_df)
        results.loc[results['Demanda_uni_equil'] < 0, 'Demanda_uni_equil'] = 0

        results.to_csv(self.output().path,
                       float_format='%.5f',
                       index=False,
                       columns=['id', 'Demanda_uni_equil'])

    def output(self):
        return luigi.LocalTarget("./data/submissions/submission.csv")
