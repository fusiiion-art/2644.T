import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
import re
import warnings
import yfinance as yf
import numpy as np

# SSL警告の抑制
warnings.simplefilter('ignore')

class AlternativeDataScraper:
    def __init__(self):
        # 403回避のための強力なヘッダー偽装
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
        }

    def fetch_tsmc_revenue(self) -> pd.DataFrame:
        """
        TSMCの月次売上レポートを取得し、YoYモメンタムを計算する
        失敗した場合はyfinanceからTSM(ADR)のデータを代替として取得する
        Source: https://investor.tsmc.com/english/monthly-revenue
        """
        logging.info("Fetching TSMC Monthly Revenue (Scraping)...")
        current_year = datetime.now().year
        years = range(current_year, current_year - 4, -1) # 過去4年分
        
        all_dfs = []
        
        # 1. 公式サイトからのスクレイピング試行
        try:
            for year in years:
                url = f"https://investor.tsmc.com/english/monthly-revenue/{year}"
                try:
                    # requestsでヘッダー付きGETリクエストを送る
                    response = requests.get(url, headers=self.headers, timeout=10)
                    response.raise_for_status() # 403エラーならここで例外発生
                    
                    # HTML文字列をpandasに渡す
                    dfs = pd.read_html(response.text, attrs={'class': 'cols-4'})
                    if dfs:
                        df_year = dfs[0]
                        df_year['Year'] = year
                        all_dfs.append(df_year)
                        logging.info(f" -> Successfully fetched TSMC data for {year}")
                except Exception as e:
                    logging.warning(f"Failed to fetch TSMC data for {year}: {e}")
        except Exception as e:
            logging.error(f"Critical error during scraping: {e}")

        # 2. データの加工 (成功した場合)
        if all_dfs:
            try:
                df = pd.concat(all_dfs, ignore_index=True)
                df.columns = [c.lower() for c in df.columns]
                
                month_map = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                
                df['month_num'] = df['month'].str.slice(0, 3).str.lower().map(month_map)
                df = df.dropna(subset=['month_num'])
                
                # 発表日を作成 (翌月10日)
                def get_release_date(row):
                    y = int(row['year'])
                    m = int(row['month_num'])
                    if m == 12:
                        y += 1
                        m = 1
                    else:
                        m += 1
                    return pd.Timestamp(year=y, month=m, day=10)

                df['Date'] = df.apply(get_release_date, axis=1)
                
                # 売上数値の抽出
                col_rev = [c for c in df.columns if 'revenue' in c][0]
                df['tsmc_revenue'] = df[col_rev].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
                
                df = df.sort_values('Date')
                df['tsmc_yoy_growth'] = df['tsmc_revenue'].pct_change(12)
                df['tsmc_mom_growth'] = df['tsmc_revenue'].pct_change(1)
                df['tsmc_growth_acceleration'] = df['tsmc_yoy_growth'].diff()
                
                result = df[['Date', 'tsmc_revenue', 'tsmc_yoy_growth', 'tsmc_growth_acceleration']].set_index('Date')
                logging.info(f"✅ Successfully scraped TSMC revenue data ({len(result)} records)")
                return result
            except Exception as e:
                logging.error(f"Error processing scraped data: {e}")

        # 3. バックアッププラン (yfinanceからTSM株価を取得)
        logging.warning("⚠️ Scraping failed or blocked. Switching to Backup: Fetching TSM (ADR) from yfinance.")
        return self._fetch_tsmc_proxy_from_yfinance()

    def _fetch_tsmc_proxy_from_yfinance(self) -> pd.DataFrame:
        """
        TSMCの米国ADR (Ticker: TSM) の価格と出来高を取得し、
        売上モメンタムの代用変数を作成する。
        """
        try:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
            df_tsm = yf.download("TSM", start=start_date, progress=False)
            
            if df_tsm.empty:
                return pd.DataFrame()

            # MultiIndexカラムの修正
            if isinstance(df_tsm.columns, pd.MultiIndex):
                df_tsm.columns = [c[0] for c in df_tsm.columns] # Price typeのみ残す

            df_tsm = df_tsm.reset_index()
            df_tsm.rename(columns={'Date': 'Date', 'Adj Close': 'Close'}, inplace=True)
            
            # 代替特徴量の作成
            # 1. Pseudo Revenue (擬似売上): 株価 * 出来高 (市場の注目度x評価)
            df_tsm['tsmc_revenue'] = df_tsm['Close'] * df_tsm['Volume']
            
            # 2. 移動平均で平滑化 (月次トレンドに近づける)
            df_tsm['tsmc_revenue'] = df_tsm['tsmc_revenue'].rolling(20).mean()
            
            # 3. 成長率
            df_tsm['tsmc_yoy_growth'] = df_tsm['Close'].pct_change(252) # 1年前比
            df_tsm['tsmc_growth_acceleration'] = df_tsm['tsmc_yoy_growth'].diff(20)
            
            result = df_tsm[['Date', 'tsmc_revenue', 'tsmc_yoy_growth', 'tsmc_growth_acceleration']].dropna().set_index('Date')
            logging.info(f"✅ Loaded TSM proxy data via yfinance ({len(result)} records).")
            return result
            
        except Exception as e:
            logging.error(f"Backup fetch failed: {e}")
            return pd.DataFrame()

    def get_all_alternative_data(self) -> pd.DataFrame:
        """
        全ソースを結合して日次データフレーム(ffill)として返す
        """
        df_tsmc = self.fetch_tsmc_revenue()
        
        if not df_tsmc.empty:
            # 日次へのリサンプリング
            df_daily = df_tsmc.resample('D').ffill().reset_index()
            return df_daily
        
        return pd.DataFrame()