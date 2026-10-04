"""Calendar-year panels and strictly chronological forecasting windows."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .common import load_players, load_performance, require_columns

STATS = ["market_value_in_eur", "goals", "assists", "minutes", "apps"]


def make_panel(players_path, appearances_path, valuations_path):
    players = load_players(players_path)
    perf = load_performance(appearances_path)
    values = pd.read_csv(valuations_path)
    require_columns(values, ["player_id", "date", "market_value_in_eur"], valuations_path)
    values["date"] = pd.to_datetime(values.date, errors="coerce")
    values["market_value_in_eur"] = pd.to_numeric(values.market_value_in_eur, errors="coerce")
    if values.date.isna().any() or values.player_id.isna().any():
        raise ValueError("Valuations have invalid dates or missing player IDs")
    if not np.isfinite(values.market_value_in_eur).all() or (values.market_value_in_eur < 0).any():
        raise ValueError("Valuations must be finite non-negative euro values")
    values["season"] = values.date.dt.year
    values = values.groupby(["player_id", "season"], as_index=False).market_value_in_eur.mean()
    panel = perf.merge(values, on=["player_id", "season"], validate="one_to_one")
    panel = panel[panel.player_id.isin(players.player_id)]
    # Never filter on last_season >= test year: that would condition on future survival.
    return panel.sort_values(["player_id", "season"]).reset_index(drop=True), players


def make_windows(panel, lookback):
    if lookback < 1:
        raise ValueError("lookback must be positive")
    require_columns(panel, ["player_id", "season"] + STATS, "panel")
    if panel.duplicated(["player_id", "season"]).any():
        raise ValueError("Panel has duplicate player/year rows")
    X, y, rows = [], [], []
    for player_id, frame in panel.groupby("player_id", sort=True):
        frame = frame.sort_values("season")
        history = {int(r.season): r for r in frame.itertuples(index=False)}
        for row in frame.itertuples(index=False):
            year = int(row.season)
            years = list(range(year - lookback, year))
            if not all(t in history for t in years):
                continue  # A missing calendar year is not a one-year lag.
            sequence = [[getattr(history[t], col) for col in STATS] for t in years]
            if not np.isfinite(sequence).all() or not np.isfinite(row.market_value_in_eur):
                continue
            X.append(sequence); y.append(row.market_value_in_eur)
            rows.append({"player_id": player_id, "season": year})
    if not X:
        raise ValueError(f"No complete {lookback}-year windows; check history or reduce --lookback")
    # Oldest to newest, so the final step is exactly the persistence baseline.
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1), pd.DataFrame(rows)


def split_masks(rows, train_end, val_year, test_year):
    if not train_end < val_year < test_year:
        raise ValueError("Require train_end < val_year < test_year")
    masks = [rows.season.to_numpy() <= train_end,
             rows.season.to_numpy() == val_year,
             rows.season.to_numpy() == test_year]
    if not all(mask.any() for mask in masks):
        raise ValueError("Train, validation and test must each have complete windows")
    return masks


def scale_sequences(X, y, train_mask):
    seq_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    seq_scaler.fit(X[train_mask].reshape(-1, X.shape[2]))
    y_scaler.fit(y[train_mask])
    return (seq_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape).astype(np.float32),
            y_scaler.transform(y).astype(np.float32), seq_scaler, y_scaler)
