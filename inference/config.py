import numpy as np


class Config:
    dataset = "CurveLanes"
    backbone = "34"
    num_lanes = 10
    num_row = 72
    num_col = 81
    num_cell_row = 200
    num_cell_col = 100
    train_width = 1600
    train_height = 800
    crop_ratio = 0.8

    row_anchor = np.linspace(0.4, 1, num_row)
    col_anchor = np.linspace(0, 1, num_col)
