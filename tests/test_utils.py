import pandas as pd
import numpy as np

from pyedpyper import utils as ut


def test_summary():
    df1 = ut.summary(a=np.arange(6), b=np.arange(6, 12))
    df2 = pd.DataFrame({"a": np.arange(6), "b": np.arange(6, 12)})
    pd.testing.assert_frame_equal(df1, df2)
