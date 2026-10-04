import argparse


def main():
    parser = argparse.ArgumentParser(description="Football ratings, scouting and market-value forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    ratings = commands.add_parser("ratings", help="XGBoost rating reconstruction and feature ablations")
    ratings.add_argument("--players", required=True); ratings.add_argument("--appearances", required=True)
    ratings.add_argument("--fifa", required=True); ratings.add_argument("--performance-year", type=int)
    scouting = commands.add_parser("scouting", help="KMeans/PCA exploratory scouting")
    scouting.add_argument("--stats", required=True); scouting.add_argument("--separator", default=";")
    scouting.add_argument("--encoding", default="latin1")
    forecast = commands.add_parser("forecast", help="LSTM with temporal holdouts and persistence baseline")
    for flag in ["players", "appearances", "valuations"]:
        forecast.add_argument("--" + flag, required=True)
    forecast.add_argument("--fifa"); forecast.add_argument("--fifa-observed-year", type=int)
    forecast.add_argument("--lookback", type=int, default=10)
    forecast.add_argument("--train-end", type=int, default=2022)
    forecast.add_argument("--val-year", type=int, default=2023)
    forecast.add_argument("--test-year", type=int, default=2024)
    forecast.add_argument("--epochs", type=int, default=200); forecast.add_argument("--patience", type=int, default=20)
    forecast.add_argument("--hidden-dim", type=int, default=128)
    for sub, folder in [(ratings, "ratings"), (scouting, "scouting"), (forecast, "forecast")]:
        sub.add_argument("--output", default="outputs/" + folder)
        sub.add_argument("--seed", type=int, default=42)
    for sub in [ratings, forecast]:
        sub.add_argument("--fifa-version")
        sub.add_argument("--match-threshold", type=float, default=90)
        sub.add_argument("--match-margin", type=float, default=5)
        sub.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "ratings":
        from .ratings import run
    elif args.command == "scouting":
        from .scouting import run
    else:
        from .forecast import run
    try:
        run(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
