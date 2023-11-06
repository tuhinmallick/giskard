import pandas as pd

from ..datasets import Dataset
from .base import BaseSlicer
from .slice import Query, LowerThan, GreaterThan, QueryBasedSliceFunction


class BruteForceSlicer(BaseSlicer):
    def __init__(self, dataset: Dataset, features=None, target=None):
        self.dataset = dataset
        self.features = features
        self.target = target

    def find_slices(self, features, target=None):
        target = target or self.target
        data = self.dataset.df

        if len(features) > 1:
            raise NotImplementedError("Only single-feature slicing is implemented for now.")
        (feature,) = features

        # Quantile-based binning
        _, cut_bin = pd.qcut(data[feature], q=4, retbins=True, duplicates="drop", labels=False)

        result_df = [[cut_bin[i], cut_bin[i + 1]] for i in range(len(cut_bin) - 1)]
        clauses = [
            [
                GreaterThan(feature, interval[0], True),
                LowerThan(feature, interval[1], True),
            ]
            for interval in result_df
        ]
        return [QueryBasedSliceFunction(Query(clause)) for clause in clauses]
