import json
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def require_columns(df, columns, source):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def clean(name):
    if pd.isna(name):
        return ""
    text = unicodedata.normalize("NFKD", str(name).casefold())
    return "".join(c for c in text if c.isalnum())


def map_fifa_position(pos):
    groups = {
        "Attacker": {"ST", "CF", "RW", "LW", "RF", "LF"},
        "Midfielder": {"CM", "CAM", "CDM", "RM", "LM", "LAM", "RAM", "RCM", "LCM"},
        "Defender": {"CB", "LB", "RB", "LCB", "RCB", "LWB", "RWB"},
        "Goalkeeper": {"GK"},
    }
    # Read position tokens, not substrings: LWB must not match LW.
    for token in re.findall(r"[A-Z]+", str(pos).upper()):
        for group, positions in groups.items():
            if token in positions:
                return group
    return "Other"


def load_players(path):
    df = pd.read_csv(path)
    require_columns(df, ["player_id", "name"], path)
    if df.player_id.isna().any() or df.player_id.duplicated().any():
        raise ValueError("players.csv must have exactly one row per non-null player_id")
    return df


def load_performance(path):
    df = pd.read_csv(path)
    require_columns(df, ["player_id", "date", "game_id", "goals", "assists", "minutes_played"], path)
    if df.duplicated(["player_id", "game_id"]).any():
        raise ValueError("Duplicate player_id/game_id appearances; deduplicate source data first")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    if df.date.isna().any():
        raise ValueError("appearances.csv contains invalid dates")
    if df.player_id.isna().any() or df.game_id.isna().any():
        raise ValueError("appearances.csv contains missing player_id/game_id")
    for col in ["goals", "assists", "minutes_played"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"appearances.csv: invalid or missing {col}")
    # These are calendar years, matching the original scripts; not football seasons.
    df["season"] = df.date.dt.year
    return df.groupby(["player_id", "season"], as_index=False).agg(
        goals=("goals", "sum"), assists=("assists", "sum"),
        minutes=("minutes_played", "sum"), apps=("game_id", "count"))


def load_fifa(path, features, version=None):
    df = pd.read_csv(path, low_memory=False)
    require_columns(df, ["long_name", "overall", "player_positions"] + list(features), path)
    if "fifa_version" in df:
        if version is None and df.fifa_version.nunique() > 1:
            raise ValueError("FIFA CSV has multiple editions; pass --fifa-version (e.g. 24)")
        if version is not None:
            df = df[pd.to_numeric(df.fifa_version, errors="coerce") == float(version)].copy()
    elif version is not None:
        raise ValueError("No fifa_version column is available; supply a preselected snapshot and omit --fifa-version")
    if df.empty:
        raise ValueError("Selected FIFA edition has no rows")
    if "fifa_update" in df and df.fifa_update.nunique() > 1:
        raise ValueError("Selected FIFA edition has several updates; supply a single update per player")
    df = df.dropna(subset=["long_name", "overall"]).copy()
    df["clean_name"] = df.long_name.map(clean)
    df = df[df.clean_name != ""].drop_duplicates().copy()
    # Different players sharing a full name cannot be told apart by name, so exclude them all.
    shared = df.clean_name.duplicated(keep=False)
    if shared.any():
        print(f"Excluded {int(shared.sum())} FIFA rows whose full name is shared by several players")
        df = df[~shared].copy()
    for col in ["overall"] + list(features):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=["overall"]).reset_index(drop=True)


def match_fifa(players, fifa, threshold=90, margin=5):
    from rapidfuzz import process, fuzz
    names = fifa.clean_name.tolist()
    lookup = {name: i for i, name in enumerate(names)}
    audit = []
    for row in players.itertuples(index=False):
        name = clean(row.name)
        status, index, score, gap, candidate = "unmatched", None, None, None, None
        if name:
            hits = process.extract(name, names, scorer=fuzz.ratio, limit=2)
            if hits:
                candidate, score, _ = hits[0]
                gap = score - hits[1][1] if len(hits) > 1 else 100.0
                if score >= threshold and gap >= margin:
                    index, status = lookup[candidate], "accepted"
                elif score >= threshold:
                    status = "ambiguous"
        audit.append({"player_id": row.player_id, "name": row.name,
                      "candidate_name": candidate, "fifa_row": index, "match_score": score,
                      "match_gap": gap, "match_status": status})
    audit = pd.DataFrame(audit)
    accepted = audit[audit.match_status == "accepted"].copy()
    # Do not silently map multiple source identities to the same FIFA identity.
    collision_ids = accepted.loc[accepted.fifa_row.duplicated(keep=False), "player_id"]
    audit.loc[audit.player_id.isin(collision_ids), "match_status"] = "collision"
    accepted = audit[audit.match_status == "accepted"].copy()
    extra = fifa.drop(columns=["clean_name"])
    # FIFA files carry their own player_id; keep it, but never as the source identity.
    extra = extra.rename(columns={c: "fifa_" + c for c in extra.columns if c in audit.columns})
    extra = extra.reset_index(names="fifa_row")
    merged = accepted.merge(extra, on="fifa_row", validate="many_to_one")
    return merged, audit


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual).reshape(-1), np.asarray(predicted).reshape(-1)
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def scatter(actual, predicted, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=20, alpha=.65)
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    ax.plot([low, high], [low, high], "--", color="gray")
    ax.set(xlabel="Actual", ylabel="Predicted", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
