## Statistical Factor Model

This is full implementation of a statistical factor model. It was inspired by Gappy's Chapter 7 of EQI and the toraniko multi-factor model implementation.

The only data required to run this is symbol-by-symbol daily asset returns. In my case I'm using the current Russell 1000 constituents.
```
┌────────────┬────────┬───────────────┐
│ date       ┆ symbol ┆ asset_returns │
│ ---        ┆ ---    ┆ ---           │
│ date       ┆ str    ┆ f64           │
╞════════════╪════════╪═══════════════╡
│ 2010-01-05 ┆ A      ┆ -0.010863     │
│ 2010-01-05 ┆ AA     ┆ -0.031231     │
│ 2010-01-05 ┆ AAL    ┆ 0.113208      │
│ 2010-01-05 ┆ AAON   ┆ -0.029015     │
│ 2010-01-05 ┆ AAPL   ┆ 0.001729      │
│ …          ┆ …      ┆ …             │
│ 2026-01-23 ┆ ZG     ┆ -0.008977     │
│ 2026-01-23 ┆ ZION   ┆ -0.031951     │
│ 2026-01-23 ┆ ZM     ┆ 0.013349      │
│ 2026-01-23 ┆ ZS     ┆ 0.006289      │
│ 2026-01-23 ┆ ZTS    ┆ -0.002813     │
└────────────┴────────┴───────────────┘
```