import pickle
from decimal import Decimal

import numpy as np

from src.train_constants import TRAIN_COLUMNS


class RandomForestAllocation:
    MIN_SINGLE_APY = 3.5
    MIN_POOLS = 8

    def __init__(self):
        self._columns = list(TRAIN_COLUMNS)
        self._columns.remove('apy')
        with open('model.pkl', 'rb') as f:
            self._model = pickle.load(f)

    def predict_allocation(self, assets_and_pools):
        total_assets = Decimal(assets_and_pools['total_assets'])

        batch = np.array([[pool[column] for column in self._columns] for pool in assets_and_pools['pools'].values()])

        y = self._model.predict(batch)

        zero_pools = y < self.MIN_SINGLE_APY
        if sum(zero_pools) > (len(y) - self.MIN_POOLS):
            zero_pools[:] = False
        y[zero_pools] = Decimal('0')
        y = [Decimal(alc) for alc in y]
        sum_y = Decimal(sum(y))
