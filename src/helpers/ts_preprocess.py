import numpy as np
from math import ceil

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))
    
def multivariate_data(dataset, target, start_index, end_index, history_size,
        target_size=1, stride=1, step=1, single_step=False, autoregressive=True):
    """ Train test split of multivariate data for Neural Network training """
    data, labels = [], []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index, stride):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if autoregressive: # DeepAR style
            labels.append(target[i-history_size:i])
        elif single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)
