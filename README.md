[🇺🇸 English](#english) | [🇨🇳 中文](#中文)

# English
## Trading Strategy Construction and Implementation
(1) Strategy Signal
The signal we used is the total return volatility (vol) of each stock in the previous period, which is measured by the standard deviation of daily returns within a month to calculate the monthly total volatility. 
This feature is calculated by vol_cal() in zid_project2_characteristics.py, output in cha_main(), and merged with the current month's revenue for subsequent sorting. In conclusion, 
the strategy ranks stocks based on the previous month's volatility and constructs an equally weighted long-short portfolio consisting of “high vol” and “low vol” stocks. 
It then tests whether the average return deviates significantly from 0 to assess whether the volatility factor generates excess returns.
The intuition behind this approach is based on the Low Volatility Anomaly, which contradicts the classical risk-return trade-off. While traditional finance suggests that higher risk yields higher returns, 
empirical studies show that low-volatility stocks often outperform high-volatility ones on a risk-adjusted basis.
The primary signal employed in this strategy is volatility-based ranking. Stocks are ranked monthly based on their rolling 20-day volatility. 
The idea is to go long on low-volatility stocks (believed to have stable upward returns) and short high-volatility stocks (often exhibiting overreaction or mean-reversion behavior). 
This is inspired by the Low Volatility Anomaly, which contradicts classical finance theory that higher risk yields higher return.
(2) Long-Short Portfolio Construction 
Each month, we construct a market-neutral, equal-weighted long-short portfolio as follows:
a) Calculate each stock’s volatility over the trailing 20 trading days at the start of the month.
b) Rank all stocks by their volatility.
c) Go long the bottom 20% (lowest volatility).
d) Go short the top 20% (highest volatility).
e) Assign equal weights to all positions on both long and short sides.
This results in a portfolio that is hedged against market direction but exposed to the volatility factor.

(3) Hypotheses Tested
We perform a t-test on the average return of the long-short portfolio to determine statistical significance.
Null hypothesis (H₀): The mean return of the long-short strategy is zero (μ = 0)
Alternative hypothesis (H₁): The mean return is not zero (μ ≠ 0)
We reject H₀ if the t-statistic exceeds ±1.96, indicating the strategy yields statistically significant alpha.
(4) Code Implementation
	aj_ret_dict() in zid_project2_etl.py retrieves and processes return data.
	vol_cal() in zid_project2_characteristics.py computes monthly volatilities.
	cha_main() merges volatility signals with monthly returns and shifts them by one period.
	pf_main() in zid_project2_portfolio.py constructs quantile-based portfolios and the long-short portfolio.
	t_stat() in zid_project2_main.py computes the statistical significance of the long-short strategy.
## Investment Universe Selection
Market Selected
We selected the U.S. equity market (NASDAQ) due to its high liquidity and data availability.
Institutional Features
	Electronic trading and high transparency
	Regulation by the SEC, ensuring fair disclosure
	Daily and intraday data access
Investment Universe
We selected approximately 80 stocks as our stock pool using the Global Industry Classification Standard (GICS). These 80 stocks cover 11 industries, which include all Level I industries. These 11 industries cover almost all LEVE 1 industries, such as technology, healthcare, and others. This helps us validate the “low volatility anomaly” strategy, as emerging and traditional industries tend to have higher volatility.
Our selected universe includes mid-to-large cap U.S. equities from diverse sectors, e.g., AAPL, MSFT, JNJ, KO, PG, XOM. 
## Results Interpretation and Evaluation
The performance statistics of the long-short portfolio highlight its strong investment potential. The average monthly return is approximately 1.15%, indicating that the strategy consistently delivers positive returns. With a standard deviation of 6.11%, the portfolio maintains a reasonable level of risk, suggesting a favorable balance between return and volatility. Most notably, the t-statistic of 2.94 exceeds the conventional threshold of 2, implying that the strategy’s returns are statistically significant and unlikely to have occurred by chance. This combination of a positive mean return, manageable risk, and strong statistical significance supports the conclusion that the strategy is both economically viable and statistically robust, effectively capturing the volatility anomaly within the selected investment universe. However, this strategy relies solely on a single factor—total return volatility—to rank stocks. While volatility is a well-documented anomaly, using it in isolation may lead to model instability, low diversification, and exposure to unintended risks, such as industry concentration or sensitivity to macroeconomic regimes.
In the future, volatility will be combined with other predictive signals (such as momentum, scale, value, and quality) to improve model robustness and reduce noise.

 ![image](https://github.com/user-attachments/assets/b804702d-099a-498a-8011-6aba3f35a80d)

Figure 1. Long-Short Portfolio

## Exploring Alternative Strategies
As an alternative to the volatility anomaly-based strategy, we explored a momentum-driven long-short portfolio. The core idea is to capitalize on the momentum effect, 
where stocks with strong recent performance tend to continue outperforming, while poorly performing stocks continue to underperform.
We computed momentum scores by summing past three months of monthly returns for each stock. At the start of each month, we ranked the stocks by their momentum score and took long positions in the top 10 and short positions in the bottom 10. 
This portfolio was rebalanced monthly.
This approach has several potential advantages. Firstly, momentum is a well-documented anomaly with persistent returns across markets and time periods. Secondly, the strategy is relatively easy to implement and interpret. 
Finally, it complements volatility-based strategies, offering diversification in both source of alpha and risk exposure.
 ![image](https://github.com/user-attachments/assets/4c640053-f843-4019-8f10-bbbe588bcb90)

Figure 2. Momentum Driven Long-Short Portfolio
## Outlier Detection in Stock Returns
To improve the robustness of our ETL pipeline and trading strategy, we implemented a data cleaning step focused on detecting and mitigating outliers in stock return data.
1. How do you detect outliers in stock return data?
We use the Z-score method to identify outliers in daily stock returns. For each stock, we compute the Z-score as:
![image](https://github.com/user-attachments/assets/d0d83691-5399-479e-a86e-8565d39f164c)

where x is the return on a given day, \mu is the mean of daily returns, and \sigma is the standard deviation. Any observation with ∣Z∣>3 is flagged as an outlier, following the empirical rule that ~99.7% of values should lie within ±3 standard deviations in a normal distribution.
The detect_outliers_zscore() function returns both a boolean mask and an outlier count per stock.
3. Why is this method appropriate for our investment universe?
Our investment universe consists of over 80 Nasdaq-listed stocks, which are generally liquid and widely followed, making their return distributions reasonably stable.
The Z-score method is simple, efficient, and interpretable, making it suitable for batch processing across large datasets and robust enough for initial anomaly detection in financial time series.
5. How do outliers affect financial data analysis and portfolio performance?
Outliers can inflate volatility estimates, distort mean return calculations, and create false factor signals. For example, one extreme event can cause a low-volatility stock to be wrongly categorized as high-volatility, affecting portfolio composition and alpha signals.
If left untreated, outliers introduce noise into risk models, reduce Sharpe ratios, and potentially lead to overfitting in backtests.
6. Does our investment universe contain outliers?
Yes. Based on the Z-score analysis, our dataset contains multiple stocks with days exhibiting ∣Z∣>3. These are likely due to market-wide events, firm-specific news, or data recording errors.
Detecting and flagging these anomalies is therefore essential for ensuring the integrity of the return series used in signal generation and portfolio construction.

 ![image](https://github.com/user-attachments/assets/5d945860-8e21-4988-af80-715b27a46bb2)

Figure 3. Example of the outlier detection
## Summary 
Based on 245 monthly observations, the long-short portfolio constructed by sorting stocks on past-month volatility generates an average return of 1.15% per month, with a t-statistic of 2.9375. This result is statistically significant at the 5% level, rejecting the null hypothesis that the portfolio return is zero.
The performance confirms the existence of a low-volatility anomaly, suggesting that stocks with lower volatility tend to outperform high-volatility stocks in a risk-adjusted, market-neutral setting. However, this analysis is based solely on a single factor (volatility).
Incorporating multiple signals—such as momentum or value—may further enhance robustness and explanatory power of the strategy.

# 中文
## 交易策略的构建与实现
1) 策略信号
本策略使用的核心信号是上一期个股的总收益波动率（vol），计算方法为：在每个月月初，统计上一整月（日度 20 个交易日）的日收益标准差，得到该月的总波动率。

具体代码：zid_project2_characteristics.py 中的 vol_cal() 负责计算；cha_main() 将该信号与当月收益对齐并滞后一期，供后续排序。

实施思路：将股票按上一期波动率从低到高排序，等权做多「低波动」组，等权做空「高波动」组，构造多空组合；检验该组合平均收益是否显著偏离 0，从而判断波动率因子是否产生超额收益。

理论动机：该做法基于 “低波动异常”（Low Volatility Anomaly）：与经典风险-收益正相关假说相反，实证研究发现低波动股票往往在风险调整后跑赢高波动股票。

2) 多空组合构建
每个月按以下步骤建立市场中性、等权重的多空组合：

步骤	内容
a	计算所有股票过去 20 日的波动率
b	按波动率升序排序
c	做多波动率最低 20% 的股票
d	做空波动率最高 20% 的股票
e	多头与空头内部均等权配置

该组合方向中性，仅暴露于「波动率因子」。

3) 假设检验
对多空组合的月度平均收益做 t 检验：

原假设 H₀：组合平均收益 μ = 0

备择假设 H₁：μ ≠ 0

当 |t 值| > 1.96（5% 显著性）则拒绝 H₀，说明策略收益具有统计显著性。

4) 代码流程
功能	                       文件 & 函数
拉取并处理收益数据	zid_project2_etl.py → aj_ret_dict()
计算月度波动率	zid_project2_characteristics.py → vol_cal()
合并信号与收益	cha_main()
构建分位 & 多空组合	zid_project2_portfolio.py → pf_main()
统计显著性	zid_project2_main.py → t_stat()

## 投资范围选择
市场选择
美国纳斯达克市场：流动性高，数据完备。

机构特征
电子交易、信息透明

SEC 监管，披露标准统一

易获得日频和高频数据

股票池
采用 GICS 行业分类，选取约 80 只中大盘股，涵盖 11 个一级行业（科技、医疗、消费等）。

行业多样化有助于检验低波动异常在不同板块的表现。

## 结果解读与评估
指标	数值
月均收益	1.15 %
月收益标准差	6.11 %
t-统计量	2.94

t 值 > 2，拒绝均值为 0 的原假设 → 策略收益统计显著。

月均 1.15 % + 合理波动 → 经济上具有吸引力。

局限：仅使用单一波动率因子，可能出现行业集中、宏观状态敏感等风险。未来可引入动量、规模、价值、质量等因子以提升稳健性。

![image](https://github.com/user-attachments/assets/5e57aa3a-209f-4c6b-95ce-92c7da839676)

图 1：低波动多空组合净值曲线

备选策略：动量因子
思路：利用动量效应，买入近 3 个月收益最好的 10 只股票，卖空最差的 10 只，月度调仓。

优点：动量异常经久不衰、实现简单，与波动率因子互补，提供 alpha 多样化。

![image](https://github.com/user-attachments/assets/95471c75-0fda-4732-9d5b-66312f355059)

图 2：动量驱动多空组合净值曲线

日收益异值检测
1. 检测方法
采用 Z-score：
![image](https://github.com/user-attachments/assets/f9fffe28-92ad-4189-a506-02ce1f364421)

|Z| > 3 视为异常点。

函数：detect_outliers_zscore() 返回布尔掩码及每只股票的异常计数。

2. 适用性
80+ 只纳斯达克股票流动性好，收益分布较稳定，Z-score 简单高效，适合批处理。

3. 异值影响
夸大波动率、扭曲平均收益、产生假信号；若不清洗，会降低夏普比率并导致过拟合。

![image](https://github.com/user-attachments/assets/f15316c8-5e13-46b5-a774-ca045c525fcb)

图 3：日收益异常点示例

## 总结
基于 245 个月样本，按前期波动率排序构建的多空组合月均收益 1.15 %，t 值 2.94，在 5% 水平上显著不为零，验证了低波动异常的存在。
然而单因子策略可能不稳定，未来将与动量、价值等因子结合，以提高模型鲁棒性并分散风险。
