import logging
import sys

import luigi

import pandas as pd

from bimbo.features.lagging_features import BuildTargetLagFeatures
from bimbo.features.mean_features import BuildClientProductFeatures
from bimbo.preprocess.load_transform import ConvertTrainAndTestDataToParquet, ConvertTrainAndTestDataToSVM
from bimbo.train.estimate import GBTEstimate
from bimbo.train.learn import GBTLearn

__all__ = (
    BuildClientProductFeatures,
    BuildTargetLagFeatures,
    ConvertTrainAndTestDataToSVM,
    ConvertTrainAndTestDataToParquet,
    GBTEstimate,
    GBTLearn
)

pd.set_option('display.width', 256)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d-%m-%Y %I:%M:%S')

# TODO: logging without debug
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # anyway, it's for local purposes only
    sys.argv.append('--local-scheduler')
    luigi.run()
