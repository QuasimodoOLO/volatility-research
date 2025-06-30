""" config.py

Configuration options for the project2 package         

"""
# IMPORTANT: Please do NOT modify this file except for defining the TICMAP dictionary

import os
import toolkit_config as tcfg

ROOTDIR = os.path.join(tcfg.PRJDIR, 'project2')
DATADIR = os.path.join(ROOTDIR, 'data')


# ------------------------------------------------------------------------------
# Part 4.2.1: Define the TICMAP Dictionary in config.py
# ------------------------------------------------------------------------------

# Your first task is to choose a stock exchange market where your team will implement
# a volatility trading strategy.

#    One country may have multiple stock exchanges.
#    For example, the U.S. has NYSE, NASDAQ, and AMEX as its main exchanges.

#    You must select at least 50 stocks listed on a single stock exchange
#    within your chosen country.

#    The selected country can be part of either a developed or emerging market.
#    Refer to the MSCI market classification for guidance:
#       https://www.msci.com/our-solutions/indexes/market-classification

# ------------------------------------------------------------------------------
# Define the TICMAP dictionary below
# ------------------------------------------------------------------------------

# This dictionary should contain ticker-to-company name mappings in the format:
#     'TICKER': 'Company Name'

#    If you use yfinance to download price data, make sure the tickers match
#    those used on Yahoo Finance (e.g., 'AAPL' for Apple Inc.).

#    If you're using another data source, be aware that ticker symbols may differ
#    and should be adjusted accordingly.

# Example:
# TICMAP = {
#     'AAPL': 'Apple Inc.',
#     'GOOGL': 'Alphabet Inc.',
#     ...
# }

# This dictionary will be used throughout the project to download and organize
# price data for your selected investment universe.

# Hint:
# Consider why you chose this country and market—you'll be asked to reflect on
# your selection in a later part of Project 2.

############################################################################
#Select leading companies from 10 GICS industries for feature comparison#
############################################################################
TICMAP = {
    # Information Technology
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'NVDA': 'NVIDIA Corporation',
    'ADBE': 'Adobe Inc.',
    'INTC': 'Intel Corporation',
    'CSCO': 'Cisco Systems Inc.',
    'AVGO': 'Broadcom Inc.',
    'TXN': 'Texas Instruments Inc.',

    # Communication Services
    'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms Inc.',
    'NFLX': 'Netflix Inc.',
    'TMUS': 'T-Mobile US Inc.',
    'CHTR': 'Charter Communications Inc.',
    'TTWO': 'Take-Two Interactive Software Inc.',
    'ROKU': 'Roku Inc.',
    'DIS': 'The Walt Disney Company',

    # Consumer Discretionary
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'SBUX': 'Starbucks Corporation',
    'BKNG': 'Booking Holdings Inc.',
    'EBAY': 'eBay Inc.',
    'MAR': 'Marriott International Inc.',
    'MELI': 'MercadoLibre Inc.',
    'YUMC': 'Yum China Holdings Inc.',

    # Consumer Staples
    'PEP': 'PepsiCo Inc.',
    'COST': 'Costco Wholesale Corporation',
    'WMT': 'Walmart Inc.',
    'MNST': 'Monster Beverage Corporation',
    'KDP': 'Keurig Dr Pepper Inc.',
    'MDLZ': 'Mondelez International Inc.',
    'KR': 'The Kroger Co.',
    'WBA': 'Walgreens Boots Alliance Inc.',

    # Health Care
    'VRTX': 'Vertex Pharmaceuticals Inc.',
    'GILD': 'Gilead Sciences Inc.',
    'REGN': 'Regeneron Pharmaceuticals Inc.',
    'BIIB': 'Biogen Inc.',
    'ILMN': 'Illumina Inc.',
    'AMGN': 'Amgen Inc.',
    'MRNA': 'Moderna Inc.',
    'ISRG': 'Intuitive Surgical Inc.',

    # Financials
    'PYPL': 'PayPal Holdings Inc.',
    'INTU': 'Intuit Inc.',
    'SCHW': 'Charles Schwab Corporation',
    'BKNG': 'Booking Holdings Inc.',
    'MTCH': 'Match Group Inc.',
    'LPLA': 'LPL Financial Holdings Inc.',
    'NDAQ': 'Nasdaq Inc.',

    # Industrials
    'CPRT': 'Copart Inc.',
    'ODFL': 'Old Dominion Freight Line Inc.',
    'JBHT': 'J.B. Hunt Transport Services Inc.',
    'TT': 'Trane Technologies plc',
    'CSX': 'CSX Corporation',
    'CHRW': 'C.H. Robinson Worldwide Inc.',
    'FAST': 'Fastenal Company',
    'EXPD': 'Expeditors International of Washington Inc.',

    # Materials
    'ALB': 'Albemarle Corporation',
    'LIN': 'Linde plc',
    'ECL': 'Ecolab Inc.',
    'NEM': 'Newmont Corporation',
    'APD': 'Air Products and Chemicals Inc.',
    'CTVA': 'Corteva Inc.',
    'SHW': 'Sherwin-Williams Company',
    'DD': 'DuPont de Nemours Inc.',

    # Energy
    'FANG': 'Diamondback Energy Inc.',
    'CTRA': 'Coterra Energy Inc.',
    'OVV': 'Ovintiv Inc.',
    'HES': 'Hess Corporation',
    'APA': 'APA Corporation',
    'MUR': 'Murphy Oil Corporation',
    'DVN': 'Devon Energy Corporation',

    # Utilities
    'SRE': 'Sempra Energy',
    'WEC': 'WEC Energy Group Inc.',
    'NEE': 'NextEra Energy Inc.',
    'XEL': 'Xcel Energy Inc.',
    'AEP': 'American Electric Power Company Inc.',
    'PEG': 'Public Service Enterprise Group Inc.',
    'ED': 'Consolidated Edison Inc.',
    'ES': 'Eversource Energy',

    # Real Estate
    'EQIX': 'Equinix Inc.',
    'AMT': 'American Tower Corporation',
    'PLD': 'Prologis Inc.',
    'DLR': 'Digital Realty Trust Inc.',
    'PSA': 'Public Storage',
    'SBAC': 'SBA Communications Corporation',
    'VICI': 'VICI Properties Inc.',
    'O': 'Realty Income Corporation'
}

TICKERS = sorted(TICMAP.keys())


# -------------------------------------------------------- 
#   Aux function to process col names
# --------------------------------------------------------
def standardise_colnames(df):
    """ Renames the columns in `df` so that 
    - Names are lower case
    - Spaces are replaced with '_'

    Parameters
    ----------
    df : dataframe


    Notes
    -----
    - If column with the standardised name already exists, the new column will
      include a '_' prefix

    Examples
    -------

    >> df = pd.DataFrame([(1, 2), (3, 4)], columns=['A', 'B C'])
    >> print(df)

       A  B C
    0  1    2
    1  3    4

    >> df2 = standardise_colnames(df)
    >> print(df2)

       a  b_c
    0  1    2
    1  3    4

    """
    cols = set(df.columns)
    # You can define `local` functions
    def _parse_name(colname):
        # Processes the column name
        new_name = colname.lower().replace(' ', '_')
        # Decide what to do. The options are:
        # 1) column name is already properly formatted:
        #   => do nothing
        # 2) column name is not properly formatted but exists in the dataframe
        #   => Include '_' prefix
        # 3) Else: return formatted name
        if new_name == colname: 
            # Returns original column
            return colname
        elif new_name in cols:
            return '_' + new_name
        else:
            return new_name
    return df.rename(columns=_parse_name)




