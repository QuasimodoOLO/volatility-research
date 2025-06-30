""" zid_project2_main.py

"""
from bdb import effective

from numpy.lib.format import EXPECTED_KEYS

from project1.zid_project1 import DATDIR
# ----------------------------------------------------------------------------
# Part 1: Read the documentation for the following methods:
#   – pandas.DataFrame.mean
#   - pandas.Series.concat
#   – pandas.Series.count
#   – pandas.Series.dropna
#   - pandas.Series.index.to_period
#   – pandas.Series.prod
#   – pandas.Series.resample
#   - ......
# Hint: you can utilize modules covered in our lectures, listed above and any others.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Part 2: import modules inside the project2 package
# ----------------------------------------------------------------------------
# Create import statements so that the module config.py and util.py (inside the project2 package)
# are imported as "cfg", and "util"
#
# <COMPLETE THIS PART>
from project2 import config as cfg
from project2 import util

# We've imported other needed scripts and defined aliases. Please keep using the same aliases for them in this project.
from project2 import zid_project2_etl as etl
from project2 import zid_project2_characteristics as cha
from project2 import zid_project2_portfolio as pf

import pandas as pd

# -----------------------------------------------------------------------------------------------
# Part 3: Follow the workflow in portfolio_main function
#         to understand how this project construct total volatility long-short portfolio
# -----------------------------------------------------------------------------------------------
def portfolio_main(tickers, start, end, cha_name, ret_freq_use, q):
    """
    Constructs equal-weighted portfolios based on the specified characteristic and quantile threshold.
    We focus on total volatility investment strategy in this project 2.
    We name the characteristic as 'vol'

    This function performs several steps to construct portfolios:
    1. Call `aj_ret_dict` function from etl script to generate a dictionary containing daily and
       monthly returns.
    2. Call `cha_main` function from cha script to generate a DataFrame containing stocks' monthly return
       and characteristic, i.e., total volatility, info.
    3. Call `pf_main` function from pf script to construct a DataFrame with
       equal-weighted quantile and long-short portfolio return series.

    Parameters
    ----------
    tickers : list
        A list including all tickers (can include lowercase and/or uppercase characters) in the investment universe

    start  :  str
        The inclusive start date for the date range of the price table imported from data folder
        For example: if you enter '2010-09-02', function in etl script will include price
        data of stocks from this date onwards.
        And make sure the provided start date is a valid calendar date.

    end  :  str
        The inclusive end date for the date range, which determines the final date
        included in the price table imported from data folder
        For example: if you enter '2010-12-20', function in etl script will encompass data
        up to and including December 20, 2010.
        And make sure the provided start date is a valid calendar date.

    cha_name : str
        The name of the characteristic. Here, it should be 'vol'

    ret_freq_use  :  list
        It identifies that which frequency returns you will use to construct the `cha_name`
        in zid_project2_characteristics.py.
        Set it as ['Daily',] when calculating stock total volatility here.

    q : int
        The number of quantiles to divide the stocks into based on their characteristic values.


    Returns
    -------
    dict_ret : dict
        A dictionary with two items, each containing a dataframe of daily and monthly returns
        for all stocks listed in the 'tickers' list.
        This dictionary is the output of `aj_ret_dict` in etl script.
        See the docstring there for a description of it.

    df_cha : df
        A DataFrame with a Monthly frequency PeriodIndex, containing rows for each year-month
        that include the stocks' monthly returns for that period and the characteristics,
        i.e., total volatility, from the previous year-month.
        This df is the output of `cha_main` function in cha script.
        See the docstring there for a description of it.

    df_portfolios : df
        A DataFrame containing the constructed equal-weighted quantile and long-short portfolios.
        This df is the output of `pf_cal` function in pf script.
        See the docstring there for a description of it.

    """

    # --------------------------------------------------------------------------------------------------------
    # Part 4: Complete etl scaffold to generate returns dictionary and to make ad_ret_dic function works
    # --------------------------------------------------------------------------------------------------------
    dict_ret = etl.aj_ret_dict(tickers, start, end)

    # ---------------------------------------------------------------------------------------------------------
    # Part 5: Complete cha scaffold to generate dataframe containing monthly total volatility for each stock
    #         and to make char_main function work
    # ---------------------------------------------------------------------------------------------------------
    df_cha = cha.cha_main(dict_ret, cha_name,  ret_freq_use)

    # -----------------------------------------------------------------------------------------------------------
    # Part 6: Read and understand functions in pf scaffold. You will need to utilize functions there to
    #         complete some of the questions in Part 7
    # -----------------------------------------------------------------------------------------------------------
    df_portfolios = pf.pf_main(df_cha, cha_name, q)

    util.color_print('Portfolio Construction All Done!')

    return dict_ret, df_cha, df_portfolios

# ----------------------------------------------------------------------------
# Part 7: Complete the auxiliary functions
# ----------------------------------------------------------------------------
def get_avg(df: pd.DataFrame, year):
    """ Returns the average value of all columns in the given df for a specified year.

    This function will calculate the column average for all columns
    from a data frame `df`, for a given year `year`.
    The data frame `df` must have a DatetimeIndex or PeriodIndex index.

    Missing values will not be included in the calculation.

    Parameters
    ----------
    df : data frame
        A Pandas data frame with a DatetimeIndex or PeriodIndex index.

    year : int
        The year as a 4-digit integer.

    Returns
    -------
    ser
        A series with the average value of columns for the year `year`.

    Example
    -------
    For a data frame `df` containing the following information:

        |            | tic1 | tic2  |
        |------------+------+-------|
        | 1999-10-13 | -1   | NaN   |
        | 1999-10-14 | 1    | 0.032 |
        | 2020-10-15 | 0    | -0.02 |
        | 2020-10-16 | 1    | -0.02 |

        >> res = get_avg(df, 1999)
        >> print(res)
        tic1      0.000
        tic2      0.032
        dtype: float64

    """
    # <COMPLETE THIS PART>
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("The input data must contain “DatetimeIndex or PeriodIndex index”.")
    if not isinstance(year, int) or len(str(year)) != 4:
        raise ValueError("The year must be a 4-digit integer.")

    try:
        # Filter by year
        mask = df.index.year == year

        result = df.loc[mask].mean()
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to calculate average: {str(e)}") from e


def get_cumulative_ret(df):
    """ Returns cumulative returns for input DataFrame.

    Given a df with return series, this function will return the
    buy-and-hold return over the entire period.

    Parameters
    ----------
    df : DataFrame
        A Pandas DataFrame containing monthly portfolio returns
        with a PeriodIndex index.
        - df.columns: portfolio names

    Returns
    -------
    ser : Series
        A series containing portfolios' buy-and-hold return over the entire period.
        - ser.index: portfolio names

    Notes
    -----
    The buy and hold cumulative return will be computed as follows:

        (1 + r1) * (1 + r2) *....* (1 + rN) - 1
        where r1, ..., rN represents monthly returns

    """
    # <COMPLETE THIS PART>
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input data must be a Pandas DataFrame.")

    if not isinstance(df.index, pd.PeriodIndex):
        raise ValueError("The index must be PeriodIndex.")
    if df.empty:
        raise ValueError("The input data cannot be empty.")

    try:
        cumulative_ret = (df + 1).prod() - 1

        return cumulative_ret.rename(None)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate cumulative profit days: {str(e)}") from e

# ----------------------------------------------------------------------------
# Part 8: Answer questions
# ----------------------------------------------------------------------------
# NOTES:
#
# - THE SCRIPTS YOU NEED TO SUBMIT ARE
#   config.py, zid_project2_main.py, zid_project2_etl.py, and zid_project2_characteristics.py
#
# - Do not create any other functions inside the scripts you need to submit unless
#   we ask you to do so.
#
# - For this part of the project, only the answers provided below will be
#   marked. You are free to create any function you want (IN A SEPARATE
#   MODULE outside the scripts you need to submit).
#
# - All your answers should be strings. If they represent a number, include 4
#   decimal places unless otherwise specified in the question description
#
# - Here is an example of how to answer the questions below. Consider the
#   following question:
#
#   Q0: Which ticker included in config.TICMAP starts with the letter "C"?
#   Q0_answer = '?'
#
#   You should replace the '?' with the correct answer:
#   Q0_answer = 'CSCO'
#
#
#     To answer the questions below, you need to run portfolio_main function in this script
#     with the following parameter values:
#     tickers: all tickers included in the dictionary config.TICMAP define your team’s investment universe,
#     start: '2000-12-29',
#     end: '2021-08-31',
#     cha_name: 'vol'.
#     ret_freq_use: ['Daily',],
#     q: 3
#     Please name the three output files as DM_Ret_dict, Vol_Ret_mrg_df, EW_LS_pf_df.
#     You can utilize the three output files and auxiliary functions to answer the questions.

#     Since each team has a different investment universe,
#     whenever we refer to a specific stock by its position (e.g., "the second stock"),
#     we assume the tickers are sorted alphabetically.
#     For example, if your universe includes tickers <'AAPL', 'TSLA', and 'V'>,
#     then 'TSLA' would be considered the second stock.

# Q1: Which stock in your sample has the lowest average daily return for the
#     year 2008 (ignoring missing values)? Your answer should include the
#     ticker for this stock.
#     Use the output dictionary, DM_Ret_dict, and auxiliary function in this script
#     to do the calculation.
Q1_ANSWER = 'nvda'


# Q2: What is the daily average return of the stock in question 1 for the year 2008.
#     Use the output dictionary, DM_Ret_dict, and auxiliary function in this script
#     to do the calculation.
Q2_ANSWER = '-0.004240836570694617'


# Q3: Which stock in your sample has the highest average monthly return for the
#     year 2019 (ignoring missing values)? Your answer should include the
#     ticker for this stock.
#     Use the output dictionary, DM_Ret_dict, and auxiliary function in this script
#     to do the calculation.
Q3_ANSWER = 'roku'


# Q4: What is the average monthly return of the stock in question 3 for the year 2019.
#     Use the output dictionary, DM_Ret_dict, and auxiliary function in this script
#     to do the calculation.
Q4_ANSWER = '0.16442600938041302'


# Q5: What is the average monthly total volatility for the 10th stock of your investment universe
#     in the year 2010?
#     Use the output dataframe, Vol_Ret_mrg_df, and auxiliary function in this script
#     to do the calculation.
Q5_ANSWER = '0.020759'


# Q6: What is the ratio of the average monthly total volatility for the 20th stock of your investment universe
#     in the year 2008 to that in the year 2018? Keep 1 decimal places.
#     Use the output dataframe, Vol_Ret_mrg_df, and auxiliary function in this script
#     to do the calculation.
Q6_ANSWER = 'nan'


# Q7: How many effective year-month for the 30th stock in year 2010. An effective year-month
#     row means both monthly return and total volatility are not null.
#     Use the output dataframe, Vol_Ret_mrg_df, to do the calculation.
#     Answer should be an integer
Q7_ANSWER = '12'


# Q8: How many rows and columns in the EW_LS_pf_df data frame?
#     The answer string should only include two integers separating by a comma.
#     The first number represents number of rows.
#     Don't include any other signs or letters etc.
Q8_ANSWER = '245,4'


# Q9: What is the average equal weighted portfolio return of the quantile with the
#     lowest total volatility for the year 2019?
#     Use the output dataframe, EW_LS_pf_d, and auxiliary function in this script
#     to do the calculation.
Q9_ANSWER = '0.027629'


# Q10: What is the cumulative portfolio return of the total volatility long-short portfolio
#      over the whole sample period?
#      Use the output dataframe, EW_LS_pf_d, and auxiliary function in this script
#      to do the calculation.
Q10_ANSWER = '9.651685'


# ----------------------------------------------------------------------------
# Part 9: Add t_stat function
# ----------------------------------------------------------------------------
# We've outputted EW_LS_pf_df file and save the total volatility long-short portfolio
# in 'ls' column from Part 8.

# Please add an auxiliary function called ‘t_stat’ below.
# You can design the function.
# But make sure that when function get called, t_stat(EW_LS_pf_df),
# the output is a DataFrame with one row called 'ls' and three columns below:
#  1.ls_bar, the mean of 'ls' columns in EW_LS_pf_df, keep 4 decimal points
#  2.ls_t, the t stat of 'ls' columns in EW_LS_pf_df, keep 4 decimal points
#  3.n_obs, the number of observations of 'ls' columns in EW_LS_pf_df, save as integer

# Notes:
# Please add the function in zid_project2_main.py.
# The name of the function should be t_stat and including docstring.
# Please replace the '?' of ls_bar, ls_t and n_obs variables below
# with the respective values of the 'ls' column in EW_LS_pf_df from Part 8,
# keep 4 decimal places if it is not an integer:
ls_bar = '0.0115'
ls_t = '2.9372'
n_obs = '245'

# <ADD THE t_stat FUNCTION HERE>
def t_stat(ew_ls_pf_df):
    try:
        ls_returns = ew_ls_pf_df['ls'].dropna()

        mean = ls_returns.mean()
        n = len(ls_returns)

        t_value = mean / (ls_returns.std() / (n ** 0.5)) if n > 1 else float('nan')

        result = pd.DataFrame({
            'ls_bar': [round(mean,4)],
            'ls_t': [round(t_value,4)],
            'n_obs': [int(n)]
        }, index=['ls'])
        return result

    except Exception as e:
        raise ValueError(f"Calculation failed(t_stat): {str(e)}")
# ----------------------------------------------------------------------------
# Part 10: project presentation
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Part 11: project report
# ----------------------------------------------------------------------------
# Please refer to project2_desc.pdf for the instructions for Parts 10 and 11.

def _test_get_avg():
    """ Test function for `get_avg`
    """
    # Made-up data
    ret = pd.Series({
        '2019-01-01': 1.0,
        '2019-01-02': 2.0,
        '2020-10-02': 4.0,
        '2020-11-12': 4.0,
    })
    df = pd.DataFrame({'some_tic': ret})
    df.index = pd.to_datetime(df.index)

    msg = 'This is the test data frame `df`:'
    util.test_print(df, msg)

    res = get_avg(df,  2019)
    to_print = [
        "This means `res =get_avg(df, year=2019) --> 1.5",
        f"The value of `res` is {res}",
    ]
    util.test_print('\n'.join(to_print))


def _test_get_cumulative_ret():
    """ Test function for `get_cumulative_ret`

    """
    # Made-up data
    idx_m = pd.to_datetime(['2019-02-28',
                            '2019-03-31',
                            '2019-04-30',]).to_period('M')
    stock1_m = [0.063590, 0.034290, 0.004290]
    stock2_m = [None, 0.024390, 0.022400]
    monthly_ret_df = pd.DataFrame({'stock1': stock1_m, 'stock2': stock2_m, }, index=idx_m)
    monthly_ret_df.index.name = 'Year_Month'
    msg = 'This is the test data frame `monthly_ret_df`:'
    util.test_print(monthly_ret_df, msg)

    res = get_cumulative_ret(monthly_ret_df)
    to_print = [
        "This means `res =get_cumulative_ret(monthly_ret_df)",
        f"The value of `res` is {res}",
    ]
    util.test_print('\n'.join(to_print))


if __name__ == "__main__":
    # print(_test_get_avg())
    # print(_test_get_cumulative_ret())
    DM_Ret_dict, Vol_Ret_mrg_df, EW_LS_pf_df = portfolio_main(tickers=cfg.TICKERS, start='2000-12-29',end='2021-08-31', cha_name='vol', ret_freq_use=['Daily',], q = 3)
    # DM_Ret_dict['Daily'].to_csv(os.path.join(DATDIR, 'DM_Ret.csv'))
    # Vol_Ret_mrg_df.to_csv(os.path.join(DATDIR, 'Vol_Ret.csv'))
    # EW_LS_pf_df.to_csv(os.path.join(DATDIR, 'EW_LS_pf.csv'))

    DM_Ret_dict['Daily'].to_csv('DM_Ret_dict.csv')
    Vol_Ret_mrg_df.to_csv('Vol_Ret_mrg_df.csv')
    EW_LS_pf_df.to_csv('EW_LS_pf_df.csv')

    # Q1
    lowest_stock_2008 = get_avg(DM_Ret_dict['Daily'], 2008).idxmin()
    print(f"(Q1)the lowest average daily return for the year 2008 is {lowest_stock_2008}")

    # Q2
    lowest_return_2008 = get_avg(DM_Ret_dict['Daily'], 2008).min()
    print(f"(Q2)the daily average return of the stock in question 1 for the year 2008 is {lowest_return_2008}")

    # Q3
    highest_stock_2019 = get_avg(DM_Ret_dict['Monthly'], 2019).idxmax()
    print(f"(Q3)highest average monthly return for the year 2019 is {highest_stock_2019}")

    # Q4
    highest_return_2019 = get_avg(DM_Ret_dict['Monthly'], 2019).max()
    print(f"(q4)the average monthly total volatility for the 10th stock of your investment universe in the year 2010 is {highest_return_2019}")

    # Q5
    all_stocks = sorted([col for col in Vol_Ret_mrg_df.columns if '_vol' not in col])
    if len(all_stocks) < 10:
        raise ValueError("The number of stocks in the portfolio is less than 10.")
    tenth_stock = all_stocks[9]

    vol_col = tenth_stock + '_vol'

    try:
        vol_df = Vol_Ret_mrg_df[[vol_col]].copy()

        avg_vol = get_avg(vol_df, 2010)

        print(f"(Q5)The average monthly volatility of the 10th stock ({tenth_stock}) in 2010 was: {avg_vol.iloc[0]:.6f}")

    except Exception as e:
        print(f"Calculation failed(Q5): {str(e)}")
    # Q6
    all_stocks = sorted([col for col in Vol_Ret_mrg_df.columns if '_vol' not in col])
    if len(all_stocks) < 20:
        raise ValueError("The number of stocks in the portfolio is less than 20.")
    twentieth_stock = all_stocks[19]

    vol_col = twentieth_stock + '_vol'

    try:
        vol_df = Vol_Ret_mrg_df[[vol_col]].copy()
        avg_vol_2008 = get_avg(vol_df, 2008).iloc[0]
        avg_vol_2018 = get_avg(vol_df,2018).iloc[0]

        if avg_vol_2018 == 0:
            raise ValueError("Volatility was zero in 2018, so the ratio cannot be calculated.")
        ratio = avg_vol_2008 / avg_vol_2018

        print(f"(Q6)The ratio of the average monthly volatility of the 20th stock ({twentieth_stock}) in 2008 and 2018 is: {ratio:.1f}")

    except Exception as e:
        print(f"Calculation failed(Q6): {str(e)}")

    # Q7
    all_stocks = sorted([col for col in Vol_Ret_mrg_df.columns if '_vol' not in col])
    if len(all_stocks) < 30:
        raise ValueError("The number of stocks in the portfolio is less than 30.")
    thirtieth_stock = all_stocks[29]
    return_col = thirtieth_stock
    vol_col = thirtieth_stock + '_vol'
    year_2010_data = Vol_Ret_mrg_df.loc['2010', [return_col, vol_col]]

    effective_months = year_2010_data.dropna(how='any').shape[0]
    print(f"(Q7)The effective number of months for the 30th stock({thirtieth_stock}) in 2010 is: {effective_months}")
    # Q8
    rows, cols = EW_LS_pf_df.shape
    print(f"(Q8){rows}, {cols}")
    # Q9
    try:
        volatility ={
            'ewp_rank_1': EW_LS_pf_df['ewp_rank_1'].loc['2019'].std(),
            'ewp_rank_2': EW_LS_pf_df['ewp_rank_2'].loc['2019'].std(),
            'ewp_rank_3': EW_LS_pf_df['ewp_rank_3'].loc['2019'].std(),
        }
        min_vol_quantile = min(volatility, key=volatility.get)
        avg_return = get_avg(EW_LS_pf_df[[min_vol_quantile]], 2019)
        Q9_ANSWER = round(avg_return.iloc[0], 6)

    except Exception as e:
        print(f"Calculation failed(Q9): {str(e)}")
        Q9_ANSWER = None

    # Q10
    try:
        cumulative_ret = get_cumulative_ret(EW_LS_pf_df[['ls']])
        Q10_ANSWER = round(cumulative_ret.iloc[0], 6)

    except Exception as e:
        print(f"Calculation failed(Q10): {str(e)}")
        Q10_ANSWER = None

    print("=====result=====")
    print(f"answer of Q9: {Q9_ANSWER}")
    print(f"answer of Q10: {Q10_ANSWER}")

    ls_stats = t_stat(EW_LS_pf_df)
    ls_bar = ls_stats.at['ls', 'ls_bar']
    ls_t = ls_stats.at['ls', 'ls_t']
    n_obs = ls_stats.at['ls', 'n_obs']
    print(f"ls_t is {ls_t}")        #t value
    print(f"n_obs is {n_obs}")       #observation (unit: month)
    print(f"ls_bar is {ls_bar}")   # Average value of portfolio returns

    ########  Analysis  #######################
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import matplotlib.dates as mdates


    def calculate_ls_statistics(ew_ls_df):
        ls_returns = ew_ls_df['ls']
        mean_return = ls_returns.mean()
        std_dev = ls_returns.std()
        t_stat = mean_return / (std_dev / np.sqrt(len(ls_returns)))
        return mean_return, std_dev, t_stat


    def plot_cumulative_return(ew_ls_df):
        ew_ls_df = ew_ls_df.copy()
        ew_ls_df.index = pd.to_datetime(ew_ls_df.index, format='%Y-%m')

        cum_ret = (1 + ew_ls_df['ls']).cumprod()

        plt.figure(figsize=(10, 4))
        plt.plot(ew_ls_df.index, cum_ret, label='Long-Short Portfolio')

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.title('Cumulative Return of Long-Short Strategy')
        plt.ylabel('Portfolio Value')
        plt.xlabel('Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def hypothesis_test_ls_returns(ls_returns: pd.Series) -> dict:
        mean_ret = ls_returns.mean()
        t_stat, p_value = stats.ttest_1samp(ls_returns.dropna(), popmean=0)
        reject_null = p_value < 0.05

        return {
            'mean_return': mean_ret,
            't_statistic': t_stat,
            'p_value': p_value,
            'reject_null': reject_null
        }

    def plot_heatmap_correlation(vol_df):
        corr = vol_df.dropna().corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
        plt.title("Monthly Return Correlation Matrix")
        plt.tight_layout()
        plt.show()


    def detect_outliers_zscore(daily_ret_df, threshold=3):
        z_scores = np.abs((daily_ret_df - daily_ret_df.mean()) / daily_ret_df.std())
        outlier_mask = z_scores > threshold
        outlier_counts = outlier_mask.sum()
        return outlier_mask, outlier_counts


    def plot_outlier_distribution(daily_ret_df, outlier_mask):
        sample_stock = daily_ret_df.columns[0]
        plt.figure(figsize=(10, 4))
        sns.histplot(daily_ret_df[sample_stock], bins=100, kde=True, label="Return Dist")
        outliers = daily_ret_df[sample_stock][outlier_mask[sample_stock]]
        plt.scatter(outliers, np.zeros_like(outliers), color='red', label='Outliers', zorder=5)
        plt.title(f"Distribution of Returns for {sample_stock}")
        plt.xlabel("Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def remove_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
        cleaned_df = df.copy()
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            cleaned_df[col] = df[col].where((df[col] >= lower_bound) & (df[col] <= upper_bound))
        return cleaned_df


    def calculate_momentum_scores(daily_ret_df, lookback_months=3):
        monthly_returns = daily_ret_df.resample('ME').sum()
        momentum_scores = monthly_returns.rolling(window=lookback_months).sum()
        return momentum_scores


    def construct_momentum_ls_pf(momentum_scores, DM_Ret):
        ls_returns = {}
        for date, scores in momentum_scores.iterrows():
            if scores.isna().sum() > 0:
                continue
            top = scores.nlargest(10).index
            bottom = scores.nsmallest(10).index
            next_month = date + pd.offsets.MonthEnd(1)
            try:
                ret_next = DM_Ret.loc[next_month.strftime('%Y-%m')].resample('ME').sum().loc[next_month]
                long_ret = ret_next[top].mean()
                short_ret = ret_next[bottom].mean()
                ls_returns[next_month] = long_ret - short_ret
            except:
                continue
        return pd.Series(ls_returns, name="Momentum_LS_Returns")


    def plot_momentum_strategy(momentum_ls_returns):
        cumulative_return = (1 + momentum_ls_returns).cumprod()
        cumulative_return.plot(figsize=(10, 5), title='Momentum Long-Short Portfolio Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    DM_Ret = pd.read_csv('DM_Ret_dict.csv')
    DM_Ret.set_index('Date', inplace=True)
    Vol_Ret_mrg_df = pd.read_csv('Vol_Ret_mrg_df.csv')
    Vol_Ret_mrg_df.set_index('Year_Month', inplace=True)
    EW_LS_pf_df = pd.read_csv('EW_LS_pf_df.csv')
    EW_LS_pf_df.set_index('Year_Month', inplace=True)

    print(EW_LS_pf_df.index)
    mean_ret, std_dev, t_stat = calculate_ls_statistics(EW_LS_pf_df)
    print(mean_ret, std_dev, t_stat)
    plot_cumulative_return(EW_LS_pf_df)
    outlier_mask, counts = detect_outliers_zscore(DM_Ret)
    plot_outlier_distribution(DM_Ret, outlier_mask)

    DM_Ret.index = pd.to_datetime(DM_Ret.index)
    momentum_scores = calculate_momentum_scores(DM_Ret)
    momentum_ls_returns = construct_momentum_ls_pf(momentum_scores, DM_Ret)
    plot_momentum_strategy(momentum_ls_returns)
    pass


