import polars as pl
from matplotlib import pyplot as plt

from src.model import estimate_factor_returns

def main():
    # print("Hello from statistical-factor-model!")
    #
    asset_returns = pl.read_csv('data/russell_1000_returns.csv').fill_null(strategy='forward')

    factor_ret_df, residual_ret_df, loadings_df = estimate_factor_returns(asset_returns, n_factors=20)

    factor_ret_df.write_csv('data/factor_ret.csv')
    residual_ret_df.write_csv('data/residual_ret_df.csv')

    plt.plot(
        factor_ret_df['date'].to_list(),
        ((1 + factor_ret_df['factor_1']).cum_prod() - 1).to_numpy(),
        label='returns'
    )
    plt.savefig('plots/factor_1.png')

    plt.hist(
        factor_ret_df['factor_1'],
        bins=100,
        label='factor returns'
    )

if __name__ == "__main__":
    main()
