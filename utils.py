import numpy as np

def sequence_data(time_series, window) -> np.array:
    """
        :param time_series: the time series object
        :param window: the window length
        :return: a list containing the list of sequences
    """
    N = len(time_series)
    if window >= N:
        return np.array([time_series])
    else:
        seq_list = []
        for i in range(N-window+1):
            seq_list.append(time_series[i:i+window])
        return np.array(seq_list)

