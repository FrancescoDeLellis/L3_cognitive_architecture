import numpy as np
import pandas as pd
from wrap_functions import wrap_to_2pi

class PhaseIndexer:

    def __init__(self, df_in: pd.DataFrame, columns_names=None) -> None:
        assert np.isclose(df_in.index[0], 0), "Phases must start with 0."
        assert np.isclose(df_in.index[-1], 2 * np.pi), "Phases must end with 2pi."
        assert np.all(np.diff(df_in.index) >= 0), "Phases must be non-decreasing."
        if columns_names:
            assert isinstance(columns_names, list) and all(isinstance(i, str) for i in columns_names), \
                "columns_names must be None or a list of strings"
        self.df = df_in
        if columns_names is None: self.columns_names = df_in.columns
        else:                     self.columns_names = columns_names

    def get_values_at_phase(self, target_phase: float, columns_in=None):
        if columns_in:
            assert isinstance(columns_in, list) and all(isinstance(i, str) for i in columns_in), \
                "columns_in must be None or a list of strings"

        target_phase = wrap_to_2pi(target_phase)
        phases_to_search = self.df.index.to_numpy()
        idx = np.searchsorted(phases_to_search, target_phase, side='left')  # finds first idx after the position smaller than target

        # Corrections:
        if idx >= len(phases_to_search):  idx = len(phases_to_search) - 1  # if selected the last position
        elif idx > 0 and (target_phase < (phases_to_search[idx - 1] + phases_to_search[idx])/2):  idx -= 1  # select actually closest value

        selected_row = self.df.iloc[idx]
        if columns_in: selected_columns = columns_in
        else:          selected_columns = self.columns_names
        return np.array(selected_row[selected_columns])