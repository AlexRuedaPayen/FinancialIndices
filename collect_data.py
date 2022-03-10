import pandas
import numpy as np

RUI_pa=pandas.read_csv(filepath_or_buffer='./Data/RUI.PA.csv')

print(RUI_pa.columns)

print(f'Data from {min(RUI_pa['Date'].values)} to {max(RUI_pa['Date'].values)}')