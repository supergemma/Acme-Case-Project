import polars as pl
import numpy as np
from scipy.optimize import linprog

# Generate fixed dataset


def generate_fixed_data():
    data = {
        'Product': ['Lipstick', 'Mascara', 'Toner', 'Bronzer'],
        'Brand': ['Bobbi Brown', 'Bobbi Brown', 'Elizabeth Arden', 'Elizabeth Arden'],
        'Current Revenue': [2400000, 3000000, 1000000, 4000000],
        'Margin': [10, 12, 9, 12],
        'Min Trend': [-1, -1, 1, 3],
        'Max Trend': [3, 3, 9, 12],
        'Min Contribution': [0.01, 0.02, 0.03, 0.04],
        'Max Contribution': [0.6, 0.7, 0.8, 0.9]
    }
    return pl.DataFrame(data)

# Calculate sales and profit


def calculate_sales_and_profit(df):
    df = df.with_columns([
        (pl.col('Current Revenue') * (1 + pl.col('Max Trend') / 100)).alias('Max Sales'),
        (pl.col('Current Revenue') * (1 + pl.col('Min Trend') / 100)).alias('Min Sales'),
        (pl.col('Current Revenue') * (1 + pl.col('Max Trend') / 100)
         * pl.col('Margin') / 100).alias('Max Profit'),
        (pl.col('Current Revenue') * (1 + pl.col('Min Trend') / 100)
         * pl.col('Margin') / 100).alias('Min Profit')
    ])
    return df

# Optimize contribution using linear programming


def optimize_contribution(df):
    c = -df.select('Max Sales').to_numpy().flatten()
    A_eq = [df.select('Min Contribution').to_numpy().flatten()]
    b_eq = [1]
    bounds = list(zip(df['Min Contribution'].to_numpy(),
                  df['Max Contribution'].to_numpy()))

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        df = df.with_columns(pl.Series('Optimized Contribution', res.x))
        df = df.with_columns((pl.col('Optimized Contribution')
                             * pl.col('Current Revenue')).alias('Optimized Sales'))
    else:
        print(f"Optimization failed: {res.message}")

    return df

# 5-Year Sales Forecast


def forecast_sales(df, growth_rate=0.05):
    projections = []
    for year in range(1, 6):
        year_df = df.select([
            pl.col('Product'),
            pl.col('Brand'),
            (pl.col('Max Sales') * (1 + growth_rate) ** year).alias('Sales')
        ])
        year_df = year_df.with_columns(pl.lit(year).alias('Year'))
        projections.append(year_df)
    return pl.concat(projections)


# export csv


def export_to_csv(df, absolute_path):
    try:
        df.write_csv(absolute_path)
        print(f"Data successfully exported to {absolute_path}")
    except Exception as e:
        print(f"Failed to export CSV: {e}")


def main():
    print("--- Fixed Data Calculation ---")
    fixed_df = generate_fixed_data()
    calculated_df = calculate_sales_and_profit(fixed_df)
    optimized_df = optimize_contribution(calculated_df)
    print(optimized_df)
    export_to_csv(
        optimized_df, "/Users/supergemma/Desktop/Optimized_Contribution_Report.csv")

    print("\n--- 5-Year Sales Forecast ---")
    forecast_df = forecast_sales(calculated_df)
    print(forecast_df)
    export_to_csv(
        forecast_df, "/Users/supergemma/Desktop/Sales_Forecast_Report.csv")


if __name__ == '__main__':
    main()
