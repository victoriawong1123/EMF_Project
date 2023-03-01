import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import matplotlib.dates as mdates


df: DataFrame = pd.read_excel(r'DATA_Project_1.xlsx')
df['WEEKDAY'] = [i.day_of_week for i in df['DATE']]
assets_only = df[df.columns.difference(['DATE', 'WEEKDAY'])]
assets_byweek = df.loc[df['WEEKDAY'] == 4]
assets_only_byweek: object = assets_byweek[assets_byweek.columns.difference(['DATE', 'WEEKDAY'])]


# Compute daily and weekly simple returns, moving weekly or fridays only
# 1. Diagnostic for individual assets: basic observation
def assets_returns(assets, period, compounded=True):
    if not compounded:
        asset_return = (assets / assets.shift(periods=period)) - 1
    elif compounded:
        asset_return = np.log(assets / assets.shift(periods=period))

    asset_return.drop(index=asset_return.index[0:abs(period)], axis=0, inplace=True)
    asset_return['DATE'] = df['DATE']
    asset_return.set_index('DATE', inplace=True)
    return asset_return


# Getting daily/weekly simple and compounded returns
# 1a
daily_simple = assets_returns(assets_only, 1, False)
weekly_simple = assets_returns(assets_only_byweek, 1, False)
daily_compounded = assets_returns(assets_only, 1, True)
weekly_compounded = assets_returns(assets_only_byweek, 1, True)

# Compare the differences between log and simple returns
# 1a
daily_diff = daily_simple - daily_compounded
weekly_diff = weekly_simple - weekly_compounded


# Plot here. Maybe with a plot showing the differences between simple & compounded too
def log_vs_simple(x, y, series1, series2, title):
    plt.gca()
    fig, ax = plt.subplots(x, y, figsize=(7.5, 6), sharex='col')
    flat_axes = ax.flatten()
    for i, key in enumerate(series1.keys()):
        flat_axes[i].plot(series1.index, series1[key].values * 100,
                          ls='-', color='b', label='simple')
        flat_axes[i].plot(series2.index, series2[key].values * 100,
                          ls='', marker='o', color='r', ms=1, label='compounded')
        flat_axes[i].set_title(key)
        flat_axes[i].tick_params(axis='both', direction='in')

        fmt = mdates.DateFormatter("'%y")
        flat_axes[i].xaxis.set_major_formatter(fmt)

    flat_axes[1].legend(frameon=False, loc=3)
    fig.supylabel('Return [%]')
    fig.supxlabel('Year')
    fig.suptitle(title)

    fig.tight_layout()
    plt.show()


# Graphs for comparing the differences between simple and compounded returns
log_vs_simple(3, 2, daily_simple, daily_compounded, 'Daily')
log_vs_simple(3, 2, weekly_simple, weekly_compounded, 'Weekly')

# Descriptive Statistics of log return
daily_compounded.describe()
weekly_compounded.describe()


# 1c
def return_moments(series: object, moments: list) -> object:
    des_summary = pd.DataFrame()
    for moment in moments:
        if moment == 1:
            des_summary['Mean'] = series.mean()
        elif moment == 2:
            des_summary['Volatility'] = series.std()
        elif moment == 3:
            des_summary['Skewness'] = series.skew()
        elif moment == 4:
            des_summary['Kurtosis'] = series.kurtosis()
    return des_summary.transpose()


# Generating the mean, volatility, skewness and kurtosis of daily and weekly compounded returns
daily_res = return_moments(daily_compounded, [1, 2, 3, 4])
weekly_res = return_moments(weekly_compounded, [1, 2, 3, 4])

"""2. Diagnostic for individual assets: exploration"""


# Find the maximum and minimum returns of the S&P500
# 2a
def get_max(series, col_name: str, i):
    max_returns = series.nlargest(i, col_name, keep='first')
    col_max = max_returns[col_name]
    return col_max


def get_min(series, col_name: str, i):
    min_returns = series.nsmallest(i, col_name, keep='first')
    col_min = min_returns[col_name]
    return col_min


stock_max = get_max(daily_compounded, 'Stock', 5)
stock_min = get_min(daily_compounded, 'Stock', 5)

# 2c Testing normality with Jarque Bera test
# def jb_statistics()
