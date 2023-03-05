import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import matplotlib.dates as mdates
from scipy import stats

df: DataFrame = pd.read_excel(r'DATA_Project_1.xlsx')
df['WEEKDAY'] = [i.day_of_week for i in df['DATE']]
assets_only = df[df.columns.difference(['DATE', 'WEEKDAY'])]
assets_byweek = df.loc[df['WEEKDAY'] == 4]
assets_only_byweek: object = assets_byweek[assets_byweek.columns.difference(['DATE', 'WEEKDAY'])]

"""Diagnostic for individual assets: basic observation"""


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
# 1a. Compare simple return and log-returns in daily frequency.
daily_simple = assets_returns(assets_only, 1, False)
weekly_simple = assets_returns(assets_only_byweek, 1, False)
daily_compounded = assets_returns(assets_only, 1, True)
weekly_compounded = assets_returns(assets_only_byweek, 1, True)

# Compare the differences between log and simple returns
# 1a
daily_diff = daily_simple - daily_compounded
weekly_diff = weekly_simple - weekly_compounded


# Graphs for showing the differences between simple and compounded returns
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

"""1c. How are the above descriptive statistics changed when you change the 
frequency from daily to weekly? """

# Descriptive Statistics of log return
daily_compounded.describe()
weekly_compounded.describe()


# 1c. Computing the mean, volatility, skewness and kurtosis of compounded returns
def return_moments(series: object, moments: list) -> pd.DataFrame:
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


# 2a. Find the maximum and minimum returns of the S&P500
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


# 2b Test if the magnitude of the crashes and booms is consistent with the hypothesis of normality


# 2c Testing normality with Jarque Bera test, we first calculate the JB-Score
def jb_statistics(returns, skewness, kurtosis):
    sample_size = np.shape(returns)[0]
    jb_score = {}
    for i in skewness.columns:
        jb_score[i] = sample_size * ((skewness[i].values ** 2 / 6) + (kurtosis[i].values - 3) ** 2 / 24)
    return pd.DataFrame(jb_score, index=['Jarque Bera Score'])


# Getting the daily and weekly skewness and kurtosis of all assets
daily_skewness = pd.DataFrame(daily_res.loc['Skewness', :]).transpose()
daily_kurtosis = pd.DataFrame(daily_res.loc['Kurtosis', :]).transpose()
weekly_skewness = pd.DataFrame(weekly_res.loc['Skewness', :]).transpose()
weekly_kurtosis = pd.DataFrame(weekly_res.loc['Kurtosis', :]).transpose()


# Generating the Jarque Bera test statistics of all assets
jarque_scores_daily = jb_statistics(daily_compounded, daily_skewness, daily_kurtosis)
jarque_scores_weekly = jb_statistics(weekly_compounded, weekly_skewness, weekly_kurtosis)


# Testing normality
# H0: Skewness and excess kurtosis are independent of each other -> normality
# H1: Skewness and excess kurtosis are dependent of each other -> non-normal
def jb_test(jb_stats, dof, alpha):
    critical_val = stats.chi2.ppf(1-alpha, df=dof)
    for i in jb_stats.columns:
        if jb_stats[i].values >= critical_val:
            print(f'Reject H0')


"""3. Diagnostic for a portfolio"""
"""Portfolio using an equally weighted allocation, 
we are computing the portfolio return as the average of the daily (weekly) 
simple return of the six asset classes """


def equal_weight(assets):
    weights = []
    for i in range(len(assets.columns)):
        weights.append(round(1 / len(assets.columns), 2))
    weights = np.asarray(weights)
    return weights


def portfolio_return(weight, daily):
    daily_portfolio_return = pd.DataFrame(daily @ weight.T)
    return daily_portfolio_return


asset_weight = equal_weight(assets_only)
daily_portfolio = portfolio_return(asset_weight, daily_simple)
weekly_portfolio = portfolio_return(asset_weight, weekly_simple)

# 3a. Descriptive statistics of portfolio daily returns
daily_portfolio_char = return_moments(daily_portfolio, moments=[1, 2, 3, 4])
weekly_portfolio_char = return_moments(weekly_portfolio, moments=[1, 2, 3, 4])
