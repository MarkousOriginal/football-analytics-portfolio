"""Position-specific clustering, silhouettes, PCA views and heuristic scouting ranks."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from .common import require_columns, write_json

FEATURES = {
    "Goalkeeper": ["PasTotCmp%", "PasTotDist", "PasTotPrgDist", "PasShoCmp%", "PasMedCmp%", "PasLonCmp%", "Sw", "TB", "PasProg", "PasLive", "PasDead", "PasFK", "Touches", "TouDefPen", "TouDef3rd", "TouMid3rd", "AerWon", "AerLost", "AerWon%", "Err", "OG"],
    "Defender": ["Tkl", "TklWon", "Blocks", "BlkSh", "BlkPass", "Int", "Tkl+Int", "Clr", "Err", "TouDef3rd", "TouDefPen"],
    "Midfielder": ["Assists", "PasAss", "Pas3rd", "PasTotCmp%", "PasTotDist", "PasProg", "Carries", "CarPrgDist", "CarProg", "Touches", "Rec", "RecProg"],
    "Attacker": ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "Assists", "PasProg", "Pas3rd", "Carries", "CarProg", "Off", "PKatt", "ShoPK"],
}


def map_position(pos):
    # The first listed position determines the broad group.
    return {"GK": "Goalkeeper", "DF": "Defender", "MF": "Midfielder", "FW": "Attacker"}.get(str(pos).split("-")[0], "Other")


def rank_score(frame, group):
    weights = {
        "Goalkeeper": {"PasTotCmp%": 1.5, "PasTotPrgDist": .01, "PasLonCmp%": 1.2, "Touches": .1, "AerWon%": .5, "Err": -3, "OG": -2},
        "Defender": {"Tkl+Int": 3, "Clr": 2, "Blocks": 2, "Err": -2},
        "Midfielder": {"Assists": 3, "PasProg": 2, "Carries": 1, "CarProg": 1, "Rec": 1},
        "Attacker": {"Goals": 4, "Shots": 2, "SoT": 3, "PasProg": 1},
    }
    return sum(frame[col] * weight for col, weight in weights[group].items())


def cluster_position(frame, features, seed=42):
    X = StandardScaler().fit_transform(frame[features])
    upper = min(6, len(frame) - 1, len(np.unique(X, axis=0)))
    trials, best = [], None
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(X)
        if not 2 <= len(np.unique(labels)) <= len(X) - 1:
            continue
        score = float(silhouette_score(X, labels, sample_size=min(len(X), 5000), random_state=seed))
        trials.append({"k": k, "silhouette": score})
        if best is None or score > best[0]:
            best = score, model, labels
    if best is None:
        raise ValueError("Too few distinct rows for a valid silhouette comparison")
    return X, best, trials


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    players = pd.read_csv(args.stats, sep=args.separator, encoding=args.encoding)
    require_columns(players, ["Player", "Pos"], args.stats)
    players["Position_Group"] = players.Pos.map(map_position)
    all_top, summaries, skipped = [], [], {}
    for group, features in FEATURES.items():
        subset = players[players.Position_Group == group].copy()
        if subset.empty:
            skipped[group] = "No rows"; continue
        require_columns(subset, features, args.stats)
        for col in features:
            subset[col] = pd.to_numeric(subset[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        original_n = len(subset)
        subset = subset.dropna(subset=["Player"] + features).reset_index(drop=True)
        if len(subset) < 3:
            skipped[group] = "Fewer than 3 complete rows"; continue
        try:
            X, (score, model, labels), trials = cluster_position(subset, features, args.seed)
        except ValueError as error:
            skipped[group] = str(error); continue
        subset["Cluster"] = labels
        # Cluster IDs are arbitrary. Interpret profiles from means, not fixed ID labels.
        subset["Role"] = [f"{group} cluster {n}" for n in labels]
        subset["Score"] = rank_score(subset, group)
        subset["Position"] = group
        top = subset.sort_values(["Cluster", "Score"], ascending=[True, False]).groupby("Cluster").head(5)
        all_top.append(top)
        subset.to_csv(out / f"{group}_players.csv", index=False)
        means = subset.groupby("Cluster")[features].mean()
        means.to_csv(out / f"{group}_cluster_means.csv")
        pd.DataFrame(trials).to_csv(out / f"{group}_k_selection.csv", index=False)
        scaler = StandardScaler().fit(subset[features])
        joblib.dump({"scaler": scaler, "kmeans": model, "features": features}, out / f"{group}_clustering.joblib")
        for dims in [2, 3]:
            if min(X.shape) < dims:
                continue
            pca = PCA(n_components=dims).fit(X); coords = pca.transform(X)
            fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d" if dims == 3 else None)
            for label in sorted(set(labels)):
                points = coords[labels == label]
                ax.scatter(*[points[:, i] for i in range(dims)], label=f"Cluster {label}", alpha=.7, s=35)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            if dims == 3:
                ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
            ax.set_title(f"{group}: {dims}D PCA, silhouette {score:.3f}"); ax.legend()
            fig.tight_layout(); fig.savefig(out / f"{group}_pca{dims}.png", dpi=160); plt.close(fig)
            joblib.dump(pca, out / f"{group}_pca{dims}.joblib")
        normalized = pd.DataFrame(MinMaxScaler().fit_transform(means), index=means.index, columns=means.columns)
        fig, ax = plt.subplots(figsize=(12, 5)); sns.heatmap(normalized, annot=True, cmap="YlOrRd", ax=ax)
        ax.set_title(f"{group}: relative cluster means (scaled within each feature)")
        fig.tight_layout(); fig.savefig(out / f"{group}_heatmap.png", dpi=160); plt.close(fig)
        summaries.append({"position": group, "k": model.n_clusters, "silhouette": score,
                          "rows": len(subset), "incomplete_rows_dropped": original_n - len(subset)})
    if not all_top:
        raise ValueError(f"No group could be clustered: {skipped}")
    pd.concat(all_top, ignore_index=True).to_csv(out / "scouting_results.csv", index=False)
    write_json(out / "run.json", {"seed": args.seed, "groups": summaries, "skipped": skipped,
        "interpretation": "Exploratory clustering; scores use manually chosen weights, not learned quality labels"})
    print(pd.DataFrame(summaries).to_string(index=False))
