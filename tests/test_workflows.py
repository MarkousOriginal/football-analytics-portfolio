"""Correctness checks and integration runs use generated data, never portfolio metrics."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd

from football_ml.common import clean, map_fifa_position, load_fifa, fifa_snapshot, match_fifa
from football_ml.forecast_data import make_windows, scale_sequences, split_masks
from football_ml.ratings import EXTRA
from football_ml.scouting import FEATURES, cluster_position


def fixtures(root):
    rng = np.random.default_rng(17)
    all_features = sorted(set(sum(EXTRA.values(), [])))
    scouting_features = sorted(set(sum(FEATURES.values(), [])))
    players, fifa, appearances, valuations, scouting = [], [], [], [], []
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    for player_id in range(1, 101):
        name = "".join(rng.choice(alphabet, 18))
        position = ["ST", "CM", "CB", "GK"][(player_id - 1) // 25]
        players.append({"player_id": player_id, "name": name, "date_of_birth": "1980-01-01"})
        attributes = {feature: float(rng.uniform(20, 90)) for feature in all_features}
        # Real FIFA exports have their own player_id and age, which must not clash with the sources.
        fifa.append({"player_id": 900000 + player_id, "age": 30, "update_as_of": "2000-09-01", "long_name": name, "overall": float(rng.uniform(50, 90)), "player_positions": position, **attributes})
        fbref = {"ST": "FW", "CM": "MF", "CB": "DF", "GK": "GK"}[position]
        scouting.append({"Player": name, "Pos": fbref, **{feature: float(rng.uniform(1, 20)) for feature in scouting_features}})
        for year in range(2000, 2025):
            appearances.append({"player_id": player_id, "date": f"{year}-06-01", "game_id": f"{player_id}-{year}",
                "goals": int(rng.integers(0, 8)), "assists": int(rng.integers(0, 5)), "minutes_played": 90})
            valuations.append({"player_id": player_id, "date": f"{year}-06-01", "market_value_in_eur": 1e6 + player_id * 1e4 + (year - 2000) * 5e4 + float(rng.normal(0, 1e4))})
    for filename, rows in [("players.csv", players), ("male_players.csv", fifa), ("appearances.csv", appearances), ("player_valuations.csv", valuations)]:
        pd.DataFrame(rows).to_csv(root / filename, index=False)
    pd.DataFrame(scouting).to_csv(root / "stats.csv", index=False, sep=";")


class CorrectnessTests(unittest.TestCase):
    def test_name_normalization_and_position_tokens(self):
        self.assertEqual(clean("  Á. Player "), "aplayer")
        self.assertEqual(clean(None), "")
        self.assertEqual(map_fifa_position("LWB, LM"), "Defender")
        self.assertEqual(map_fifa_position("GK"), "Goalkeeper")

    def test_fuzzy_identity_collisions_are_rejected(self):
        players = pd.DataFrame({"player_id": [1, 2], "name": ["A Player", "A Player"]})
        fifa = pd.DataFrame({"clean_name": ["aplayer"], "overall": [80]})
        matched, audit = match_fifa(players, fifa)
        self.assertTrue(matched.empty)
        self.assertTrue((audit.match_status == "collision").all())

    def test_multiple_fifa_editions_cannot_merge_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fifa.csv"
            pd.DataFrame({"long_name": ["A", "A"], "overall": [80, 81], "player_positions": ["GK", "GK"], "fifa_version": [23, 24]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "multiple editions"):
                load_fifa(path, [])
            self.assertEqual(load_fifa(path, [], "24").overall.iloc[0], 81)
            with self.assertRaisesRegex(ValueError, "no rows"):
                load_fifa(path, [], "99")

    def test_fifa_update_dates_must_precede_observation_year(self):
        fifa = pd.DataFrame({"fifa_version": [22, 22], "update_as_of": ["2021-09-23", "2021-09-23"]})
        self.assertEqual(fifa_snapshot(fifa, 2021)["update_as_of_max"], "2021-09-23")
        # An edition labelled for one year can contain records dated later; those must be rejected.
        with self.assertRaisesRegex(ValueError, "after --fifa-observed-year"):
            fifa_snapshot(fifa.assign(update_as_of=["2021-09-23", "2022-02-01"]), 2021)
        with self.assertRaisesRegex(ValueError, "invalid dates"):
            fifa_snapshot(fifa.assign(update_as_of=["2021-09-23", None]), 2021)

    def test_windows_are_chronological_and_do_not_bridge_missing_years(self):
        panel = pd.DataFrame({"player_id": [1, 1, 1, 1], "season": [2024, 2020, 2023, 2021],
            "market_value_in_eur": [4., 0., 3., 1.], "goals": [0]*4, "assists": [0]*4, "minutes": [90]*4, "apps": [1]*4})
        X, y, rows = make_windows(panel, 1)
        self.assertEqual(rows.season.tolist(), [2021, 2024])
        self.assertEqual(X[:, -1, 0].tolist(), [0., 3.])
        self.assertEqual(y[:, 0].tolist(), [1., 4.])
        panel = pd.concat([panel, pd.DataFrame({"player_id": [1], "season": [2022], "market_value_in_eur": [2.], "goals": [0], "assists": [0], "minutes": [90], "apps": [1]})])
        X, y, rows = make_windows(panel, 3)
        self.assertEqual(X[0, :, 0].tolist(), [0., 1., 2.])
        self.assertEqual(int(rows.season.iloc[0]), 2023)

    def test_validation_is_disjoint_and_scalers_fit_only_training(self):
        rows = pd.DataFrame({"season": [2021, 2022, 2023, 2024]})
        masks = split_masks(rows, 2022, 2023, 2024)
        self.assertEqual(sum(mask.astype(int) for mask in masks).tolist(), [1, 1, 1, 1])
        X = np.array([[[1.]], [[2.]], [[100.]], [[200.]]], dtype=np.float32)
        y = np.array([[1.], [2.], [100.], [200.]], dtype=np.float32)
        scaled, _, xs, ys = scale_sequences(X, y, masks[0])
        self.assertEqual(float(xs.data_max_[0]), 2.)
        self.assertEqual(float(ys.data_max_[0]), 2.)
        self.assertGreater(float(scaled[-1, 0, 0]), 1.)
        with self.assertRaises(ValueError):
            split_masks(rows, 2023, 2023, 2024)

    def test_small_and_constant_clusters_fail_cleanly(self):
        with self.assertRaises(ValueError):
            cluster_position(pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]}), ["a", "b"])
        X, best, trials = cluster_position(pd.DataFrame({"a": [0, 1, 9, 10], "b": [1, 2, 9, 10]}), ["a", "b"])
        self.assertLessEqual(max(t["k"] for t in trials), 3)


class IntegrationTests(unittest.TestCase):
    def test_three_workflows_create_artifacts_and_comparable_metrics(self):
        from football_ml.ratings import run as ratings
        from football_ml.scouting import run as scouting
        from football_ml.forecast import run as forecast
        import torch
        from football_ml.model import RNNWithFIFAAttention
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixtures(root)
            common = dict(players=root / "players.csv", appearances=root / "appearances.csv", seed=42, threads=1,
                          fifa_version=None, match_threshold=90, match_margin=5)
            ratings(Namespace(**common, fifa=root / "male_players.csv", performance_year=2023, output=root / "ratings"))
            scored = pd.read_csv(root / "ratings" / "metrics.csv")
            self.assertEqual(len(scored), 12)  # Four positions times baseline and two variants.
            self.assertTrue((scored.groupby("position").test_n.nunique() == 1).all())
            scouting(Namespace(stats=root / "stats.csv", separator=";", encoding="latin1", seed=42, output=root / "scouting"))
            self.assertTrue((root / "scouting" / "scouting_results.csv").exists())
            forecast(Namespace(**common, valuations=root / "player_valuations.csv", fifa=root / "male_players.csv",
                fifa_observed_year=2000, lookback=3, train_end=2022, val_year=2023, test_year=2024,
                epochs=2, patience=2, hidden_dim=8, output=root / "forecast"))
            result = pd.read_csv(root / "forecast" / "metrics.csv")
            self.assertEqual(set(result.variant), {"persistence", "lstm_without_fifa", "lstm_fifa_attention"})
            audit = pd.read_csv(root / "forecast" / "split_audit.csv")
            self.assertTrue((audit.loc[audit.split == "train", "season"] <= 2022).all())
            self.assertTrue((audit.loc[audit.split == "validation", "season"] == 2023).all())
            run_info = json.loads((root / "forecast" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_info["fifa_snapshot"]["update_as_of_max"], "2000-09-01")
            # Splits are by year, so every generated test player also has training years.
            self.assertEqual(run_info["test_players_also_in_training"], run_info["test_players"])
            checkpoint = torch.load(root / "forecast" / "lstm_fifa_attention_model.pt", weights_only=True)
            model = RNNWithFIFAAttention(checkpoint["seq_input_dim"], checkpoint["static_input_dim"], checkpoint["hidden_dim"])
            model.load_state_dict(checkpoint["state_dict"])
            self.assertEqual(tuple(model(torch.zeros(2, 3, 5), torch.zeros(2, checkpoint["static_input_dim"])).shape), (2, 1))
            # The default performance-only CLI path must also work without a FIFA file.
            forecast(Namespace(**common, valuations=root / "player_valuations.csv", fifa=None,
                fifa_observed_year=None, lookback=3, train_end=2022, val_year=2023, test_year=2024,
                epochs=1, patience=1, hidden_dim=8, output=root / "forecast_without_fifa"))
            basic = pd.read_csv(root / "forecast_without_fifa" / "metrics.csv")
            self.assertEqual(set(basic.variant), {"persistence", "lstm_without_fifa"})
            # A future FIFA snapshot must never enter a historical forecast.
            with self.assertRaisesRegex(ValueError, "No target year"):
                forecast(Namespace(**common, valuations=root / "player_valuations.csv", fifa=root / "male_players.csv",
                    fifa_observed_year=2024, lookback=3, train_end=2022, val_year=2023, test_year=2024,
                    epochs=1, patience=1, hidden_dim=8, output=root / "future_snapshot"))


if __name__ == "__main__":
    unittest.main()
