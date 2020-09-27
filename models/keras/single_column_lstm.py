from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.keras.multi_column_lstm

class LSTM(models.keras.multi_column_lstm.LSTM):
  """CNN for sentimental analysis."""

  def __init__(self, seq_len, if_binary_classification):
    super().__init__(seq_len, 1, if_binary_classification)




