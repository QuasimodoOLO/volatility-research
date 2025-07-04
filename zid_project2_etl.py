""" zid_project2_etl.py

"""

# ----------------------------------------------------------------------------
# Part 4.1: import needed modules
# ----------------------------------------------------------------------------
# Create import statements to import all modules you need in this script
# Note: please keep the aliases consistent throughout the project.
#       For details, review the import statements in zid_project2_main.py

# <COMPLETE THIS PART>
import pandas as pd
from project2 import config as cfg
from project2 import util
import yfinance as yf
from pathlib import Path
import time
from random import uniform
# ----------------------------------------------------------------------------
# Part 4.2: Complete the download_prc_to_csv function
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Part 4.2.1: Complete TICMAP in config.py
# ----------------------------------------------------------------------------
""" Define the TICMAP dictionary in config.py according to the instructions provided in the file """


# ----------------------------------------------------------------------------
# Part 4.2.2: Complete the download_prc_to_csv function
# ----------------------------------------------------------------------------
def download_prc_to_csv(start, end, tickers=cfg.TICKERS):
    """
    Downloads historical price data for a list of tickers and saves each as a separate CSV file
    to the project2 data folder.

    Parameters:
    -----------
    start : str
          The start date for the historical data in 'YYYY-MM-DD' format.

    end : str
          The end date for the historical data in 'YYYY-MM-DD' format.

    tickers : list, optional
          A list containing all the stock tickers that users intend to process through
          this function. Defaults to cfg.TICKERS.

    Output:
    -------
    For each ticker in `tickers`, a CSV file named '<ticker>_prc.csv' (e.g., 'aal_prc.csv')
    is created in the project2 data folder. The file name <ticker> should be in lower case.

    Each CSV file includes daily price data with the following columns:
          - Date: Trading date
          - Open: Opening price
          - High: Highest price of the day
          - Low: Lowest price of the day
          - Close: Closing price
          - Adj Close: Adjusted closing price
          - Volume: Trading volume

    Notes:
    ------
      - We recommend using the yfinance package to download the price data, but it is not mandatory
        as long as your implementation matches the expected output format.
      - The CSV files must be saved in the project2 data folder. Do not hard-code the directory path.
      - The saved files should be named using the format '<ticker>_prc.csv'.
      - To fully complete Part 4.2, ensure that the TICMAP dictionary is properly defined in config.py,
        following the instructions provided in that file.
      - You can test this function by calling the _test_download_prc_to_csv test function
        using a sample `TICMAP`.

    Hints:
    ------
      - An example CSV file has been provided in the data folder.
        Your output files should follow the same format and contain similar types of content.
        The order of the columns does not necessarily need to be the same.
    """
    # <COMPLETE THIS PART>
    # Ensure that the data directory exists
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Download stock data
    for i, ticker in enumerate(tickers, 1):
        try:
            file_name = f"{ticker.lower()}_prc.csv"
            file_path = data_dir / file_name

            if file_path.exists():
                print(f"[{i}/{len(tickers)}] ’{ticker}‘ --> the data file already exists. Skip download.")
                continue

            print(f"[{i}/{len(tickers)}] Downloading {ticker} ...")

            end_inclusive = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            data = yf.download(ticker,
                               start=start,
                               end=end_inclusive,
                               multi_level_index=False,
                               progress=False,
                               auto_adjust=False,
                               ignore_tz=True)

            print(data)
            if data is None or data.empty:
                print(f"Warning: No data found for {ticker}. Skipped.")
                continue

            data = data.reset_index()
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            data = data[required_cols]
            data.to_csv(file_path,index=False)
            print(f"[{i}/{len(tickers)}] {ticker} save to the {file_name}")

            time.sleep(uniform(0.5, 2.5))
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

# ----------------------------------------------------------------------------
# Part 4.3: Complete the read_prc_csv function
# ----------------------------------------------------------------------------
def read_prc_csv(tic, start, end, prc_col='Adj Close'):
    """ This function extracts and returns a Pandas Series of adjusted close prices
    for a specified ticker over a defined date range, sourced from a CSV file
    containing stock price information.

    Parameters
    ----------
    tic : str
        String with the ticker (can include lowercase and/or uppercase characters)

    start  :  str
        The inclusive start date for the date range of the output Pandas series
        For example: if you enter '2010-09-02', the function will include data from
        this date onwards. And make sure the provided start date is a valid
        calendar date.

    end  :  str
        The inclusive end date for the date range, which determines the final date
        included in the output Pandas series
        For example: if you enter '2010-12-20', the output series will encompass data
        up to and including December 20, 2010. And make sure the provided start date
        is a valid calendar date.

    prc_col: str, optional
        Column name of stock adjusted price in the imported CSV file

    Returns
    -------
    ser
        A Pandas Series comprising the adjusted close prices of a stock, identified
        by its ticker `tic`, for a specified date range. This range is inclusive,
        beginning from the `start` date and extending through to the `end` date.

        This data frame must meet the following criteria:
        - ser.index: `DatetimeIndex` with dates, matching the dates contained in
          the CSV file. The labels in the index must be `datetime` objects.

        - ser.columns: Rename the adjusted price column to `tic` in lowercase

        - ser.index.name: the index name remains the same as it is in the imported
          CSV file.

    Notes:
         - Ensure that the returned series does not contain any entries with null values.

    Examples
    --------
    IMPORTANT: The examples below are for illustration purposes. Your ticker/sample
    period may be different.

        >> tic = 'AAPL'
        >> ser = read_prc_csv(tic, '2010-01-04', '2010-12-31')
        >> util.test_print(ser)

    ----------------------------------------
    Date
    2010-01-04     6.604801
    2010-01-05     6.616219
                    ...
    2010-12-30     9.988830
    2010-12-31     9.954883
    Name: aapl, Length: 252, dtype: float64

    Obj type is: <class 'pandas.core.series.Series'>

    <class 'pandas.core.series.Series'>
    DatetimeIndex: 252 entries, 2010-01-04 to 2010-12-31
    Series name: aapl
    Non-Null Count  Dtype
    --------------  -----
    252 non-null    float64
    dtypes: float64(1)
    memory usage: 3.9 KB
    ----------------------------------------
     Hints
     -----
     - Remember that the ticker `tic` in `<tic>`_prc.csv is in lower case.
       File names in non-windows systems are case sensitive (so 'AAA.csv' and
       'aaa.csv' are different files).

    """
    # <COMPLETE THIS PART>
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        if start_date > end_date:
            raise ValueError("The start date cannot be later than the end date.")
    except Exception as e:
        raise ValueError(f"Invalid date format: {str(e)}") from e

    file_name = f"{tic.lower()}_prc.csv"
    file_path = Path(__file__).parent / "data" / file_name

    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date',encoding='utf-8')

    except FileNotFoundError:
        raise FileNotFoundError(f"can NOT found the data file: {file_path}")
    except Exception as e:
        raise ValueError(f"can NOT read the file: {str(e)}") from e

    if prc_col not in df.columns:
        raise ValueError(f"CSV file does not contain columns: {prc_col}")

    mask = (df.index >= start_date) & (df.index <= end_date)
    ser = df.loc[mask, prc_col].copy()
    # Check whether data within the date range is empty (to avoid returning empty Series)
    if ser.empty:
        raise ValueError(f"No data found for {tic} in the given date range: {start} to {end}")
    # missing value check
    if ser.isnull().any():
        null_dates = ser[ser.isnull()].index.tolist()
        raise ValueError(f"The data contains null values on these dates: {null_dates}")

    ser.name = tic.lower()
    ser.index.name = 'Date'
    return ser

# ----------------------------------------------------------------------------
# Part 4.4: Complete the daily_return_cal function
# ----------------------------------------------------------------------------
def daily_return_cal(prc):
    """ Create a pandas series containing daily returns for an individual stock,
    given a Pandas series with daily prices and datetime-formatted index.

    This function uses the `read_prc_csv` function in this module to read the
    price information for a stock.

    Parameters
    ----------
    prc : ser
        A Pandas series with adjusted closing price information for individual stocks.
        This series is the output of `read_prc_csv`.
        See the docstring of the `read_prc_csv` function for a description of this series.

    Returns
    -------
    ser,
        A series with daily return for a given ticker calculating from `prc` series.
        Daily returns are computed by dividing today's adjusted closing price to the
        previous trading day's adjusted closing price, minus 1.

        - ser.index: DatetimeIndex with dates. These dates should include all
          dates in the `prc` data frame excluding those with NaN daily return value.

        - ser.name: should be same as `prc` series

    Examples
    --------
    Note: The examples below are for illustration purposes. Your ticker/sample
    period may be different.

        >> ser_price = read_prc_csv(tic='AAPL', start='2020-09-03', end='2020-09-09')
        >> _test_daily_return_cal(made_up_data=False, ser_prc=ser_price)

        ----------------------------------------
        This is the test ser `prc`:

        Date
        2020-09-03    120.879997
        2020-09-04    120.959999
        2020-09-08    112.820000
        2020-09-09    117.320000
        Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        DatetimeIndex: 4 entries, 2020-09-03 to 2020-09-09
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        4 non-null      float64
        dtypes: float64(1)
        memory usage: 64.0 bytes
        ----------------------------------------

        ----------------------------------------
        This means `res_daily = daily_return_cal(prc)`, print out res_daily:

        Date
        2020-09-04    0.000662
        2020-09-08   -0.067295
        2020-09-09    0.039887
        Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        DatetimeIndex: 3 entries, 2020-09-04 to 2020-09-09
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        3 non-null      float64
        dtypes: float64(1)
         usage: 48.0 bytes
        ----------------------------------------

    Hints
     -----
     - Ensure that the returns do not contain any entries with null values.

    """
    # <COMPLETE THIS PART>
    if not isinstance(prc, pd.Series):
        raise TypeError("The parameter type must be pandas Series.")
    if not isinstance(prc.index, pd.DatetimeIndex):
        raise ValueError("The index of Series must be DatetimeIndex.")
    if prc.isnull().any():
        raise ValueError("Input price series contains null values.")

    prc = prc.dropna()

    if len(prc) < 2:
        raise ValueError("A price series requires at least two data points to calculate the yield.")

    daily_return = prc.pct_change().dropna()

    daily_return.name = prc.name

    if not isinstance(daily_return.index, pd.DatetimeIndex):
        raise TypeError("Yield rate sequence index anomaly.")
    return daily_return
# ----------------------------------------------------------------------------
# Part 4.5: Complete the monthly_return_cal function
# ----------------------------------------------------------------------------
def monthly_return_cal(prc):
    """ Create a pandas series containing monthly returns for an individual stock,
    given a Pandas series with daily prices and datetime-formatted index.

    This function uses the `read_prc_csv` function in this module to read the
    price information .

    Parameters
    ----------
    prc : ser
        A Pandas series with adjusted closing price information for individual stocks.
        This series is the output of `read_prc_csv`.
        See the docstring of the `read_prc_csv` function for a description of this series.

    Returns
    -------
    ser,
        A series with monthly return for a given ticker calculating from `prc`.
        Monthly returns are computed by dividing the current end-of-month
        adjusted price to the previous end-of-month adjusted price, minus 1.
        If the number of price entries in `prc` of a year-month is less than 18,
        the calculated monthly return should **NOT** be presented in resulting ser

        - ser.index: PeriodIndex with year-month. These year-month should include all
        year-month from the prc series, but exclude months with fewer than 18 price entries.
        This exclusion criteria means that any month with less than 18 entries in the 'prc'
        series should not be considered.

        - ser.index.name: rename the series index as 'Year_Month'

        - ser.name: should be same as `prc` series

    Notes:
        - There are different ways to achieve the function goal.
          For this project, please read the documentation for the method pandas.Series.resample to
          convert a time series from daily to monthly.
          Read to_period() documentations to convert DatetimeIndex to PeriodIndex

    Examples
    --------
    Note: The examples below are for illustration purposes. Your ticker/sample
    period may be different.

        >> ser_price = read_prc_csv(tic='AAPL', start='2020-08-31', end='2021-01-10')
        >> _test_monthly_return_cal(made_up_data=False, ser_prc=ser_price)

        prc series:
        ----------------------------------------
        This is the test ser `prc`:

        Date
        2020-08-31    129.039993
        2020-09-01    134.179993
        ...
        2020-09-30    115.809998
        ...
        2020-10-15    120.709999
        2020-10-16    119.019997
        Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        DatetimeIndex: 34 entries, 2020-08-31 to 2020-10-16
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        34 non-null     float64
        dtypes: float64(1)
        memory usage: 544.0 bytes
        ----------------------------------------

        ----------------------------------------
        This means `res_monthly = monthly_return_cal(prc)`, print out res_monthly:

        Year_Month
        2020-09   -0.102526
        Freq: M, Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        PeriodIndex: 1 entries, 2020-09 to 2020-09
        Freq: M
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        1 non-null      float64
        dtypes: float64(1)
        memory usage: 16.0 bytes
        ----------------------------------------

    Hints
     -----
     - Ensure that the returns do not contain any entries with null values.

    """
    # <COMPLETE THIS PART>
    if not isinstance(prc, pd.Series):
        raise TypeError("The parameter type must be pandas Series.")
    if not isinstance(prc.index, pd.DatetimeIndex):
        raise ValueError("The index of Series must be DatetimeIndex.")
    if prc.isnull().any():
        raise ValueError("Input price series contains null values.")

    monthly_last_trading_day = prc.resample('ME').last()

    # monthly rate of return
    monthly_return = monthly_last_trading_day.pct_change()

    #Number of trading days per month
    trading_days = prc.resample('ME').count()

    # Filter months with trading days >= 18 days
    valid_months = trading_days[trading_days >= 18].index

    # Only retain months with trading days >= 18 days.
    monthly_return = monthly_return[monthly_return.index.isin(valid_months)]

    #Convert index to PeriodIndex
    monthly_return.index = monthly_return.index.to_period('M')
    monthly_return.index.name = 'Year_Month'

    # Keep the Series name consistent with the input.
    monthly_return.name = prc.name
    # Remove any NaN values that may exist.
    monthly_return = monthly_return.dropna()

    if not isinstance(monthly_return.index, pd.PeriodIndex):
        raise TypeError("Output index is not a PeriodIndex.")

    monthly_return = monthly_return.sort_index()

    return monthly_return

# ----------------------------------------------------------------------------
# Part 4.6: Complete the aj_ret_dict function
# ----------------------------------------------------------------------------
def aj_ret_dict(tickers, start, end):
    """ Create a dictionary with two items, each containing a dataframe
    of daily and monthly returns for all stocks listed in `tickers`.

    The keys for these items should be named 'Daily' and 'Monthly', respectively,
    resulting in the following structure:
        {"Daily": daily_return_df,
         "Monthly": monthly_return_df}

    This function first uses `download_prc_to_csv` to download prices data to data folder.
    It then uses the `read_prc_csv` function from this module to retrieve
    price information over a given time range. Next, it applies daily_return_cal
    and monthly_return_cal functions to compute daily and monthly returns for each ticker
    listed in `tickers`. Finally, it converts the series of daily and monthly returns
    into dataframes and stores them in a dictionary, with 'Daily' and 'Monthly'
    serving as the keys for their respective dataframes.

    Parameters
    ----------
    tickers : list
        A list containing all the stock tickers that users intend to process through
        this function.
        String of the tickers inside the 'tickers' can include lowercase and/or
        uppercase characters
    start  :  str
        The inclusive start date for the date range when retrieve price information
        from CSV file. It is a parameter will be used in `read_prc_csv` function.
        For example: if you enter '2010-09-02', the `read_prc_csv` function will include
        data from this date onwards.
        And please make sure the provided start date is a valid calendar date.
    end  :  str
        The inclusive end date for the date range when retrieve price information
        from CSV file. It is a parameter will be used in `read_prc_csv` function.
        For example: if you enter '2010-12-20', the `read_prc_csv` function output series
        will encompass data up to and including December 20, 2010.
        And please make sure the provided start date is a valid calendar date.

    Returns
    -------
    dic,
        A dictionary with two items, each containing a dataframe of daily and monthly returns
        for all stocks listed in the 'tickers' list.

        - dic.keys(): The dictionary should have two keys: "Daily" and "Monthly".

        - dic.values(): The dictionary should have two {key:value} pairs.
          The "Daily"/"Monthly" key contains a dataframe of daily/monthly returns for all stocks
          listed in the 'tickers' list

        - dic['Daily'].index/dic['Monthly'].index: DatetimeIndex/PeriodIndex with dates/year-month.
          The dates/year-month should include all those in the return series outputted by
          the daily_return_cal/monthly_return_cal function, each generated for the respective
          tickers in the `tickers` list.

        - dic['Daily'].columns/dic['Monthly'].columns: should include all tickers from the
          tickers list, converted to lowercase.

        - dic['Daily'].index.name/dic['Monthly'].index.name: 'Date'/'Year_Month'

    Examples:
    --------
    Note: The examples below are for illustration purposes. Your ticker/sample
    period may be different.
        If we run _test_aj_ret_dict(['AAPL', 'TSLA'], start='2010-06-25', end='2010-08-05'),
        it will print out:

        ----------------------------------------
        This means `dict_ret = aj_ret_dict(tickers, start, end)`, print out dict_ret:

        {'Daily':                 aapl      tsla
        Date
        2010-06-28  0.006000       NaN
        2010-06-29 -0.045211       NaN
        2010-06-30 -0.018113 -0.002512
        2010-07-01 -0.012126 -0.078472
        ...
        2010-08-04  0.004009 -0.031435
        2010-08-05 -0.004867 -0.038100,
         'Monthly':                 aapl     tsla
        Year_Month
        2010-07     0.022741 -0.16324}

        Obj type is: <class 'dict'>

        the Key of the dictionary is Daily, value info:
        <class 'pandas.core.frame.DataFrame'>
        DatetimeIndex: 28 entries, 2010-06-28 to 2010-08-05
        Data columns (total 2 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   aapl    28 non-null     float64
         1   tsla    26 non-null     float64
        dtypes: float64(2)
        memory usage: 672.0 bytes

        the Key of the dictionary is Monthly, value info:
        <class 'pandas.core.frame.DataFrame'>
        PeriodIndex: 1 entries, 2010-07 to 2010-07
        Freq: M
        Data columns (total 2 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   aapl    1 non-null      float64
         1   tsla    1 non-null      float64
        dtypes: float64(2)
        memory usage: 24.0 bytes
        ----------------------------------------
    """
    # <COMPLETE THIS PART>
    # ret_dict = {
    #     'Daily': pd.DataFrame(),
    #     'Monthly': pd.DataFrame()
    # }

    if not isinstance(tickers, list) or len(tickers) == 0:
        raise ValueError("The parameter 'tickers' must be a non-empty list.")

    tickers = [tic.lower() for tic in tickers]

    try:
        download_prc_to_csv(start, end, tickers)
        daily_list = []
        monthly_list = []
        valid_tickers = []
        for tic in tickers:
            try:
                prc = read_prc_csv(tic, start, end)
                daily_ret = daily_return_cal(prc)
                monthly_ret = monthly_return_cal(prc)

                daily_list.append(daily_ret)
                monthly_list.append(monthly_ret)
                valid_tickers.append(tic)

            except Exception as e:
                print(f"Skip {tic.upper()}, processing failed: {str(e)}")
                continue

        df_daily = pd.concat(daily_list, axis=1)
        df_monthly = pd.concat(monthly_list, axis=1)

        if len(valid_tickers) == 0:
            raise ValueError("No valid tickers processed. Please check ticker list and time range.")

        df_daily.columns = valid_tickers
        df_monthly.columns = valid_tickers

        df_daily.index.name = 'Date'
        df_monthly.index.name = 'Year_Month'
        return {
            'Daily': df_daily,
            'Monthly': df_monthly
        }

    except Exception as e:
        raise RuntimeError(f"Failed to create yield dictionary: {str(e)}") from e

# ----------------------------------------------------------------------------
#   Test functions
# ----------------------------------------------------------------------------
def _test_download_prc_to_csv():
    """ Test function for `download_prc_to_csv`
    """
    tickers = ['AAPL', 'NVDA']
    download_prc_to_csv(start='2000-01-01', end='2025-01-01', tickers=tickers)


def _test_read_prc_csv():
    """ Test function for `read_prc_csv`
    """
    tic = 'AAPL'
    ser = read_prc_csv(tic, '2010-01-04', '2010-12-31')
    util.test_print(ser)


def _test_daily_return_cal(made_up_data=True, ser_prc=None):
    """ Test function for `daily_return_cal`
    """
    if made_up_data:
        # Made-up data
        prc = pd.Series({
             '2019-01-29': 1.0, '2019-01-30': 2.0, '2019-01-31': 1.5, '2019-02-01': 1.4,
             '2019-02-04': 1.6, '2019-02-05': 1.2, '2019-02-06': 1.4, '2019-02-07': 1.9, '2019-02-08': 1.7, })
        prc.name = 'comp_tic'
        prc.index = pd.to_datetime(prc.index)
        prc.index.name = 'Date'
    else:
        prc = ser_prc.copy()

    msg = 'This is the test ser `prc`:'
    util.test_print(prc, msg)

    res_daily = daily_return_cal(prc)
    msg = "This means `res_daily = daily_return_cal(prc)`, print out res_daily:"
    util.test_print(res_daily, msg)


def _test_monthly_return_cal(made_up_data=True, ser_prc=None):
    """ Test function for `monthly_return_cal`
    """
    if made_up_data:
        # Made-up data
        prc = pd.Series({
              '2019-01-28': 2.0, '2019-01-29': 2.0, '2019-01-30': 1.0, '2019-01-31': 1.5, '2019-02-01': 1.4,
              '2019-02-04': 1.6, '2019-02-05': 1.2, '2019-02-06': 1.4, '2019-02-07': 1.9, '2019-02-08': 1.7,
              '2019-02-11': 1.7, '2019-02-12': 1.5, '2019-02-13': 1.7, '2019-02-14': 1.2, '2019-02-15': 1.4,
              '2019-02-18': 0.9, '2019-02-19': 1.4, '2019-02-20': 1.7, '2019-02-21': 1.0, '2019-02-22': 1.2,
              '2019-02-25': 0.8, '2019-02-26': 1.7, '2019-02-27': 1.4, '2019-02-28': 1.0, '2019-03-01': 1.3, })
        prc.name = 'comp_tic'
        prc.index = pd.to_datetime(prc.index)
        prc.index.name = 'Date'
    else:
        prc = ser_prc.copy()

    msg = 'This is the test ser `prc`:'
    util.test_print(prc, msg)

    res_monthly = monthly_return_cal(prc)
    msg = "This means `res_monthly = monthly_return_cal(prc)`, print out res_monthly:"
    util.test_print(res_monthly, msg)


def _test_aj_ret_dict(tickers, start, end):
    """ Test function for `aj_ret_dict`
    """

    dict_ret = aj_ret_dict(tickers, start, end)

    msg = "This means `dict_ret = aj_ret_dict(tickers, start, end)`, print out dict_ret:"
    util.test_print(dict_ret, msg)

    return dict_ret


if __name__ == "__main__":
    pass
    # # test download_prc_to_csv function
    # _test_download_prc_to_csv()
    # #test read_prc_csv function
    # _test_read_prc_csv()

    # # use made-up series to test daily_return_cal function
    # _test_daily_return_cal()
    # # use apple prc series to test daily_return_cal function
    # ser_price = read_prc_csv(tic='AAPL', start='2020-09-03', end='2020-09-09')
    # _test_daily_return_cal(made_up_data=False, ser_prc=ser_price)
    #
    # # use made-up series to test daily_return_cal function
    # _test_monthly_return_cal()
    # # use apple prc series to test daily_return_cal function
    # ser_price = read_prc_csv(tic='AAPL', start='2010-08-31', end='2011-01-10')
    # _test_monthly_return_cal(made_up_data=False, ser_prc=ser_price)
    # # test aj_ret_dict function
    # _test_aj_ret_dict(['AAPL', 'TSLA'], start='2010-06-25', end='2010-08-05')

