import numpy as np
import pandas as pd
from pandas import DataFrame as df
from matplotlib import pyplot as plt

import seaborn as sns

from IPython import embed

if __name__ == "__main__":
    data = pd.read_csv('../data/machine_usage_bak.csv', error_bad_lines=False, sep="\t")

    ## 描述两变量相关性
    coeff = data.iloc[:, [1,2,3,6,7,8]].corr()  # 相关性
    embed()
    sns.heatmap(coeff)
    # embed()