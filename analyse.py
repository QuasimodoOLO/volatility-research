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


DM_Ret = pd.read_csv('DM_Ret.csv')
DM_Ret.set_index('Date', inplace=True)
Vol_Ret_mrg_df = pd.read_csv('Vol_Ret.csv')
Vol_Ret_mrg_df.set_index('Year_Month', inplace=True)
EW_LS_pf_df = pd.read_csv('EW_LS_pf.csv')
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
