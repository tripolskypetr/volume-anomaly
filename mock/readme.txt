Fixtures extracted from BTCUSDT-aggTrades-2025-03-01.csv (Binance, timestamps in µs).

Format per JSON: { label, ts_ms, historical, recent }
  historical — trades from the 15 minutes preceding the bucket start (capped 20k)
  recent     — trades of the 30-second bucket itself (capped 3000)

Selection: 30s buckets ranked by volume/count z-score over the day;
calm_baseline is the quietest 30s window with ≥100 trades.

Results with the z-channel detector (defaults, threshold 0.75):
  spike_1_vol_count  conf=1.000  anomaly  (zVol≈104, sell-side)
  spike_2_vol_count  conf=1.000  anomaly  (zVol≈190, neutral direction at trained threshold)
  spike_3_count_only conf≈0.93   anomaly  (zRate≈32 — HFT count burst, low volume)
  spike_4_vol        conf=1.000  anomaly  (zVol≈306, sell-side)
  calm_baseline      conf≈0.007  quiet
