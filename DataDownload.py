import wrds

def download_wrds_data():
    conn = wrds.Connection()

    # CRSP: Equity data for Merton model inputs
    crsp_query = f"""
    SELECT a.permno, a.date, a.ret, a.prc, a.shrout, 
           b.siccd, b.exchcd  -- Industry/Exchange codes
    FROM crsp.dsf AS a
    LEFT JOIN crsp.dse AS b
    ON a.permno = b.permno AND a.date = b.date
    WHERE a.date BETWEEN '2015-01-01' AND '2025-01-01'
    AND b.shrcd IN (10, 11)  -- Common stocks
    """
    crsp_data = conn.raw_sql(crsp_query)

    # Compustat: Balance sheet data for debt/equity
    compustat_query = """
    SELECT gvkey, datadate, at, lt, seq, sale  -- Assets, Liabilities, Equity
    FROM comp.funda
    WHERE indfmt = 'INDL' AND datafmt = 'STD' 
    AND datadate BETWEEN '2015-01-01' AND '2025-01-01'
    """
    compustat_data = conn.raw_sql(compustat_query)

    # OptionMetrics: Implied volatility (Heston input)
    # Requires OptionMetrics subscription
    all_options = db.raw_sql("""SELECT * FROM optionm_all.vsurfd2023 vol_surface 
    INNER JOIN optionm_all.hvold2023 realized_vol ON vol_surface.secid=realized_vol.secid AND vol_surface.date=realized_vol.date AND vol_surface.days=realized_vol.days
    INNER JOIN optionm_all.secprd2023 prices ON vol_surface.secid=prices.secid AND vol_surface.date=prices.date 
    WHERE vol_surface.date='2023-01-03' AND vol_surface.impl_volatility IS NOT NULL
    LIMIT 10000""")
    all_options


if __name__ == "__main__":
    crsp_data, compustat_data, optionmetrics_data = download_wrds_data()
    crsp_data.to_csv('crsp_data.csv', index=False)
    compustat_data.to_csv('compustat_data.csv', index=False)
    optionmetrics_data.to_csv('optionmetrics_data.csv', index=False)