import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
  """
  Calcualate confidence interval of given data
  Args:
    data (list): the data from which confidence interval is to be found
    kwargs:
      confidence (float, 0.95): the value of confidence
  Returns:
    m-h: lower bound of confidence interval
    m: mean
    m+h: higher bound of confidence interval
  """
  a = 1.0 * np.array(data)
  n = len(a)
  m, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
  return m-h, m, m+h

