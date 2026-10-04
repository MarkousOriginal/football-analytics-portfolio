"""LSTM training with disjoint years, early stopping, persistence and optional FIFA ablation."""
import copy
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from .common import load_fifa, fifa_snapshot, match_fifa, metrics, write_json, scatter
from .forecast_data import make_panel, make_windows, split_masks, scale_sequences, STATS

FIFA_FEATURES = ["overall", "attacking_finishing", "attacking_volleys", "movement_reactions", "pace",
    "shooting", "movement_acceleration", "passing", "skill_ball_control",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
    "physic", "mentality_interceptions"]


def fit_predict(X, static, y, masks, args, out, name):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .model import RNNWithFIFAAttention
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    train, val, test = masks
    xt = torch.from_numpy(X); st = torch.from_numpy(static); yt = torch.from_numpy(y)
    model = RNNWithFIFAAttention(X.shape[2], static.shape[1], args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(xt[train], st[train], yt[train]), batch_size=32, shuffle=True, generator=generator)
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for xb, sb, yb in loader:
            optimizer.zero_grad(); loss = loss_fn(model(xb, sb), yb)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xt[val], st[val]), yt[val]))
        if not np.isfinite(val_loss):
            raise ValueError("Training produced a non-finite validation loss")
        history.append({"epoch": epoch, "train_mse_scaled": total / int(train.sum()), "val_mse_scaled": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(xt[test], st[test]).numpy()
    pd.DataFrame(history).to_csv(out / f"{name}_training.csv", index=False)
    torch.save({"state_dict": best_state, "seq_input_dim": X.shape[2], "static_input_dim": static.shape[1],
                "hidden_dim": args.hidden_dim, "best_epoch": best_epoch}, out / f"{name}_model.pt")
    return pred, best_epoch


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.threads < 1 or args.hidden_dim < 1:
        raise ValueError("epochs, patience, threads and hidden_dim must be positive")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    panel, players = make_panel(args.players, args.appearances, args.valuations)
    X, y, rows = make_windows(panel, args.lookback)
    static = None
    static_features = []
    snapshot = None
    if args.fifa:
        if args.fifa_observed_year is None:
            raise ValueError("With --fifa, provide --fifa-observed-year: actual year this snapshot was available, not the edition number")
        # A snapshot published during year s is treated as available from Jan 1 of s+1.
        eligible = rows.season.to_numpy() > args.fifa_observed_year
        X, y, rows = X[eligible], y[eligible], rows.loc[eligible].reset_index(drop=True)
        if not len(rows):
            raise ValueError("No target year is later than the FIFA snapshot observation year")
        fifa = load_fifa(args.fifa, [f for f in FIFA_FEATURES if f != "overall"], args.fifa_version)
        # The edition number alone does not prove availability; check the recorded update dates.
        snapshot = fifa_snapshot(fifa, args.fifa_observed_year)
        matched, audit = match_fifa(players, fifa, args.match_threshold, args.match_margin)
        audit.to_csv(out / "match_audit.csv", index=False)
        static = rows.merge(matched[["player_id"] + FIFA_FEATURES], on="player_id", how="left", validate="many_to_one")[FIFA_FEATURES].to_numpy(dtype=float)
        static_features = FIFA_FEATURES
    masks = split_masks(rows, args.train_end, args.val_year, args.test_year)
    train, val, test = masks
    if static is not None and not np.isfinite(static[train]).any():
        raise ValueError("No historical FIFA matches in training; inspect match_audit.csv")
    X_scaled, y_scaled, seq_scaler, y_scaler = scale_sequences(X, y, train)
    # Persistence means the last observed annual market value, never the target year.
    baseline = X[test, -1, 0].copy()
    actual = y[test].reshape(-1)
    base_scores = metrics(actual, baseline)
    results = [{"variant": "persistence", **base_scores, "mae_improvement_pct": 0.0}]
    predictions = rows.loc[test].copy(); predictions["actual_eur"] = actual; predictions["persistence_eur"] = baseline
    variants = [("lstm_without_fifa", np.zeros((len(X), 1), dtype=np.float32))]
    preprocess = {"sequence_scaler": seq_scaler, "target_scaler": y_scaler, "sequence_features": STATS, "lookback": args.lookback}
    if static is not None:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputer.fit(static[train]); filled = imputer.transform(static)
        static_scaler = MinMaxScaler().fit(filled[train])
        variants.append(("lstm_fifa_attention", static_scaler.transform(filled).astype(np.float32)))
        preprocess.update({"static_imputer": imputer, "static_scaler": static_scaler, "static_features": static_features})
    # Both variants use exactly the same players and dates when FIFA is supplied.
    for name, inputs in variants:
        pred_scaled, best_epoch = fit_predict(X_scaled, inputs, y_scaled, masks, args, out, name)
        pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
        # Evaluate raw outputs; do not silently clip negative predictions.
        scores = metrics(actual, pred)
        improvement = 100 * (1 - scores["mae"] / base_scores["mae"]) if base_scores["mae"] else None
        results.append({"variant": name, **scores, "mae_improvement_pct": improvement, "best_epoch": best_epoch})
        predictions[name + "_eur"] = pred
        scatter(actual, pred, f"{name}: {args.test_year}", out / f"{name}_predictions.png")
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    row_split = rows.copy(); row_split["split"] = np.select(masks, ["train", "validation", "test"], default="unused")
    row_split.to_csv(out / "split_audit.csv", index=False)
    joblib.dump(preprocess, out / "preprocessing.joblib")
    # Splits are by year, not by player: test players may also appear in earlier training years.
    test_players = rows.loc[test, "player_id"]
    write_json(out / "run.json", {"seed": args.seed, "lookback": args.lookback, "train_end": args.train_end,
        "val_year": args.val_year, "test_year": args.test_year, "train_n": int(train.sum()), "val_n": int(val.sum()),
        "test_n": int(test.sum()), "test_players": int(test_players.nunique()),
        "test_players_also_in_training": int(test_players[test_players.isin(rows.loc[train, "player_id"])].nunique()),
        "fifa_observed_year": args.fifa_observed_year, "fifa_version": args.fifa_version, "fifa_snapshot": snapshot,
        "results": results, "target": "Mean observed market value in euros during the target calendar year",
        "population": "Players with complete consecutive histories; not a test of unseen-player generalization"})
    print(pd.DataFrame(results).to_string(index=False))
