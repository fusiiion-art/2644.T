"""
異常なActual値の調査スクリプト
2025年Q4に集中する極端な値の原因を特定
"""
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path(r"C:\AI_Project\logs\daily_2644t_walkforward\version_0\wf_results_daily_2644t.csv")
FEATURES_PATH = Path(r"C:\AI_Project\data\daily_2644t\features_daily_2644t.parquet")

def main():
    print("=" * 60)
    print("Extreme Actual Values Investigation")
    print("=" * 60)
    
    # Load walkforward results
    wf_df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
    wf_df.set_index('Date', inplace=True)
    
    # Find extreme values
    threshold = 0.3
    extreme_mask = np.abs(wf_df['actual']) > threshold
    extreme_dates = wf_df[extreme_mask].index
    
    print(f"\nExtreme values (|actual| > {threshold}): {len(extreme_dates)} cases")
    
    # Load original features if available
    if FEATURES_PATH.exists():
        features_df = pd.read_parquet(FEATURES_PATH)
        if 'Date' in features_df.columns:
            features_df.set_index('Date', inplace=True)
        
        # Check target column calculation
        target_cols = [c for c in features_df.columns if 'return' in c.lower() and 'semi' in c.lower()]
        print(f"\nTarget-related columns: {target_cols[:5]}")
        
        # Compare values on extreme dates
        print("\n" + "-" * 40)
        print("Comparison with original features:")
        print("-" * 40)
        
        for date in extreme_dates[:10]:
            wf_actual = wf_df.loc[date, 'actual']
            print(f"\n{date.strftime('%Y-%m-%d')}: WF actual = {wf_actual:.4f}")
            
            if date in features_df.index:
                for col in target_cols[:3]:
                    if col in features_df.columns:
                        feat_val = features_df.loc[date, col]
                        print(f"  {col}: {feat_val:.4f}")
    else:
        print(f"\nFeatures file not found: {FEATURES_PATH}")
    
    # Analyze patterns
    print("\n" + "=" * 60)
    print("Pattern Analysis")
    print("=" * 60)
    
    extreme_df = wf_df[extreme_mask].copy()
    
    # By month
    extreme_df['month'] = extreme_df.index.to_period('M')
    monthly = extreme_df.groupby('month').agg({
        'actual': ['count', 'mean', 'std', 'min', 'max']
    })
    print("\nMonthly breakdown:")
    print(monthly.to_string())
    
    # By regime
    print("\nBy regime:")
    regime_stats = extreme_df.groupby('regime')['actual'].describe()
    print(regime_stats.to_string())
    
    # Check for consecutive extreme values
    print("\n" + "-" * 40)
    print("Consecutive extreme values:")
    print("-" * 40)
    extreme_dates_list = list(extreme_dates)
    for i in range(len(extreme_dates_list) - 1):
        diff = (extreme_dates_list[i+1] - extreme_dates_list[i]).days
        if diff <= 3:
            print(f"{extreme_dates_list[i].strftime('%Y-%m-%d')} -> {extreme_dates_list[i+1].strftime('%Y-%m-%d')} ({diff} days)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    print("""
1. Check data source for 2025-10-01 ~ 2025-12-31 period:
   - Stock splits, dividends, trading halts?
   - Data provider issues?

2. Verify return calculation:
   - Is it log return or simple return?
   - Check for missing values in price data

3. Consider clipping actual values during training:
   actual_clipped = np.clip(actual, -0.3, 0.3)
   
4. Add robust loss function:
   - Huber loss instead of MSE
   - Quantile regression
""")
    
    print("\nInvestigation complete.")

if __name__ == "__main__":
    main()
