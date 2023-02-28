import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

df: DataFrame = pd.read_excel(r'DATA_Project_1.xlsx')
df['WEEKDAY'] = [i.day_of_week for i in df['DATE']]
assets_only = df[df.columns.difference(['DATE', 'WEEKDAY'])]
assets_byweek = df.loc[df['WEEKDAY'] == 4]
assets_only_byweek=assets_byweek[assets_byweek.columns.difference(['DATE', 'WEEKDAY'])]


# Compute daily and weekly simple returns, moving weekly or fridays only
def assets_returns(assets, period, compounded=True):
    if not compounded:
        asset_return = (assets/assets.shift(periods=period)) - 1
    elif compounded:
        asset_return = np.log(assets/assets.shift(periods=period))

    asset_return.drop(index=asset_return.index[0:abs(period)], axis=0, inplace=True)
    asset_return['DATE'] = df['DATE']
    asset_return.set_index('DATE', inplace=True)
    return asset_return

# def assets_returns_weekly(assets, period, compounded=True):


daily_simple = assets_returns(assets_only, 1, False)
weekly_simple = assets_returns(assets_only_byweek, 1, False)
daily_compounded = assets_returns(assets_only, 1, True)
weekly_compounded = assets_returns(assets_only_byweek, 1, True)


daily_diff = daily_simple - daily_compounded
weekly_diff = weekly_simple - weekly_compounded


# Plot here. Maybe with a plot showing the differences between simple & compounded too
# daily_compounded.plot()
# plt.show()
# daily_simple.plot()
# plt.show()
daily_diff.plot()
plt.show()


# Descriptive Statistics
daily_compounded.describe()
weekly_compounded.describe()