import luigi
from luigi.util import requires

import numpy as np
import pandas as pd

import xgboost as xgb

from bimbo.preprocess.load_transform import ProcessTrainData

DEFAULT_FEATURES = ["Semana", "Agencia_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID",
                    "targetl1", "targetl2", "targetl3", "targetl4", "targetl5"]

# keeping them here as I'm not planning to tune hyperparams in this experiment
XGB_PARAMS = {
    "objective": "reg:linear",
    "booster": "gbtree",
    "eta": 0.1,
    "max_depth": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "eval_metric": 'rmse',
    "seed": 1111,
    # "nthread": n_jobs
}


@requires(ProcessTrainData)
class GBTLearn(luigi.Task):
    """
    Train model on the training dataset and save the model afterwards.
    """
    n_evals = luigi.IntParameter(default=75, description="XGB rounds")
    desired_sample_size = luigi.IntParameter(default=30000, description="Sample size for XGB watchlist")
    features = luigi.ListParameter(default=DEFAULT_FEATURES, description="Features to use for training")

    def run(self):
        data_train = pd.read_hdf(self.input().path)

        # merge features.
        # TODO: easy way to attach them
        data_lagged_features = pd.read_hdf("./data/processed/client_product_semana_features.h5")
        data_train = data_train.merge(data_lagged_features, how='left', on=["Cliente_ID", "Producto_ID", "Semana"])

        # why? Well, when testing your model, you might use very small datasets that will make XGB fail
        # due to small size. Here it takes sample size higher if it's needed.
        max_sample_size = min(max(int(data_train.size * 0.005), 1), self.desired_sample_size)

        watchlist_sampled = data_train.sample(n=max_sample_size, random_state=1)
        non_watchlist_sample = data_train.drop(watchlist_sampled.index)

        watchlist = xgb.DMatrix(watchlist_sampled[self.features], label=watchlist_sampled['target'], missing=np.nan)
        evals = [(watchlist, 'eval')]

        dtrain = xgb.DMatrix(non_watchlist_sample[self.features], label=non_watchlist_sample['target'], missing=np.nan)
        xgb_fit = xgb.train(XGB_PARAMS, dtrain, self.n_evals,
                            evals=evals,
                            verbose_eval=1,
                            early_stopping_rounds=10,
                            maximize=False)

        xgb_fit.save_model(self.output().path)

    def output(self):
        return luigi.LocalTarget("./data/models/xgb.model")
