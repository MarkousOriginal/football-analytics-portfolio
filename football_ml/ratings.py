"""Position-specific FIFA rating reconstruction with comparable feature ablations."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .common import (load_players, load_performance, load_fifa, fifa_snapshot, match_fifa,
                     map_fifa_position, metrics, write_json, scatter, require_columns)

EXTRA = {
    "Attacker": ["attacking_finishing", "attacking_volleys", "movement_reactions", "pace", "shooting"],
    "Midfielder": ["movement_acceleration", "skill_curve", "skill_fk_accuracy", "passing", "skill_ball_control"],
    "Defender": ["defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle", "physic", "mentality_interceptions"],
    "Goalkeeper": ["goalkeeping_diving", "goalkeeping_handling", "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed"],
}
BASE = ["goals", "assists", "goals_per90", "assists_per90", "minutes", "apps", "age"]


def run(args):
    from xgboost import XGBRegressor
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    players = load_players(args.players)
    require_columns(players, ["date_of_birth"], args.players)
    perf = load_performance(args.appearances)
    if args.performance_year is not None:
        perf = perf[perf.season == args.performance_year]
    birth_year = pd.to_datetime(players.date_of_birth, errors="coerce").dt.year
    players["birth_year"] = birth_year
    perf = perf.merge(players[["player_id", "birth_year"]], on="player_id", validate="many_to_one")
    perf["age"] = perf.season - perf.birth_year
    stats = perf.groupby("player_id", as_index=False).agg(
        goals=("goals", "sum"), assists=("assists", "sum"), minutes=("minutes", "sum"),
        apps=("apps", "sum"), age=("age", "mean"))
    stats = stats[stats.minutes > 0].copy()
    stats["goals_per90"] = stats.goals / stats.minutes * 90
    stats["assists_per90"] = stats.assists / stats.minutes * 90
    features = sorted({f for fs in EXTRA.values() for f in fs})
    fifa = load_fifa(args.fifa, features, args.fifa_version)
    matched, audit = match_fifa(players, fifa, args.match_threshold, args.match_margin)
    audit.to_csv(out / "match_audit.csv", index=False)
    # Model features come from observed performance; FIFA columns such as age are kept apart.
    matched = matched.rename(columns={c: "fifa_" + c for c in matched.columns if c in stats.columns and c != "player_id"})
    data = stats.merge(matched, on="player_id", validate="one_to_one")
    data["position_group"] = data.player_positions.map(map_fifa_position)
    data.to_csv(out / "matched_players.csv", index=False)
    results, predictions, skipped = [], [], {}
    for group, extra in EXTRA.items():
        subset = data[data.position_group == group].reset_index(drop=True)
        if len(subset) < 20:
            skipped[group] = f"Only {len(subset)} matched players; need at least 20"
            continue
        train, test = train_test_split(np.arange(len(subset)), test_size=.2, random_state=args.seed)
        # There is one row per player, so player identities cannot cross this split.
        y = subset.overall.to_numpy()
        pred = np.full(len(test), y[train].mean())
        results.append({"position": group, "variant": "training_mean", "train_n": len(train), "test_n": len(test), **metrics(y[test], pred)})
        for variant, cols in [("performance_only", BASE), ("fifa_augmented", BASE + extra)]:
            X = subset[cols].replace([np.inf, -np.inf], np.nan)
            # Tree models require no MinMaxScaler; missing values remain missing.
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=.1,
                                 random_state=args.seed, n_jobs=args.threads, objective="reg:squarederror")
            model.fit(X.iloc[train].to_numpy(), y[train]); pred = model.predict(X.iloc[test].to_numpy())
            scores = metrics(y[test], pred)
            results.append({"position": group, "variant": variant, "train_n": len(train), "test_n": len(test), **scores})
            predictions.append(pd.DataFrame({"player_id": subset.iloc[test].player_id.to_numpy(),
                "position": group, "variant": variant, "actual": y[test], "predicted": pred}))
            joblib.dump({"model": model, "features": cols}, out / f"{group}_{variant}.joblib")
            scatter(y[test], pred, f"{group}: {variant}", out / f"{group}_{variant}_predictions.png")
            importance = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
            importance.to_csv(out / f"{group}_{variant}_importance.csv", header=["importance"])
            fig, ax = plt.subplots(figsize=(8, 4)); importance.head(10).sort_values().plot.barh(ax=ax)
            ax.set(title=f"{group}: {variant}", xlabel="XGBoost feature importance")
            fig.tight_layout(); fig.savefig(out / f"{group}_{variant}_importance.png", dpi=160); plt.close(fig)
    if not results:
        raise ValueError("No position has enough matched players. Inspect match_audit.csv and source data.")
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    write_json(out / "run.json", {"seed": args.seed, "performance_year": args.performance_year,
        "fifa_version": args.fifa_version, "fifa_snapshot": fifa_snapshot(fifa), "skipped_groups": skipped,
        "task": "Contemporaneous rating reconstruction, not future football performance forecasting"})
    print(pd.DataFrame(results).to_string(index=False))
