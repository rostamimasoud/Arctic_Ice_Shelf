"""Stage 4: emulators, their skill, and what they learned about distance.

The stage runs in three parts.

An ensemble of the spatial model is integrated across a range of greenhouse
forcings and stored as a supervised learning problem: given the state of the
whole domain at the start of a month and the forcing acting on it, predict how
the state changes over that month.

Four emulators are then fitted to it.  One sees each cell in isolation and knows
nothing of its neighbours, which makes it the control that shows how much skill
comes from spatial information.  The other three pass messages between
neighbouring cells, with weights that are fixed by the graph, learned, or
learned and additionally conditioned on the geometry of each connection.  Skill
is measured on ensemble members held out entirely, and compared against
predicting no change at all.

Finally the distance over which influence travels is measured three ways: by
perturbing the physical model, by perturbing a trained emulator in the same way,
and by composing the learned attention weights.  The first is the answer, since
the transport parameters of the model were prescribed and are therefore known.
Comparing the other two against it tests whether attention weights can be read
as physical connectivity, which is usually assumed and rarely checked.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import RUNS_DIR, SEED, ensure_dirs
from arcticgnn.interpret.connectivity import (attention_length_scale,
                                              choose_sources, compare,
                                              emulator_length_scale,
                                              physical_length_scale)
from arcticgnn.models.gnn import GraphTensors
from arcticgnn.models.train import set_threads, train_emulator
from arcticgnn.physics.seaice import SeaIceParams
from arcticgnn.spatial.ensemble import build_dataset, default_path
from arcticgnn.spatial.model import SpatialArcticModel

#: Greenhouse forcings spanning the ensemble, W m^-2.
FORCINGS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0])

#: Architectures fitted, in the order reported.
ARCHITECTURES = ("dense", "convolution", "attention", "edge")

#: Months at which the influence length scale is measured.  One in the middle of
#: the growth season and one at the summer minimum, because the question is
#: whether connectivity depends on the state of the ice.
SEASON_MONTHS = {"winter": 1.0 / 12.0, "summer": 8.0 / 12.0}


def spatial_params(calibration: dict) -> SeaIceParams:
    spatial = calibration["spatial"]
    return SeaIceParams().replace(
        longwave_offset=float(spatial["longwave_offset"]),
        longwave_annual=float(spatial["longwave_annual"]),
        albedo_scale=0.5, forcing=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()

    ensure_dirs()
    set_threads(arguments.threads)

    calibration = json.loads((RUNS_DIR / "calibration.json").read_text())
    column = spatial_params(calibration)

    path = default_path()
    if arguments.rebuild or not path.exists():
        print("running the ensemble")
        dataset, grid, graph = build_dataset(FORCINGS, years=arguments.years,
                                             column=column)
        dataset.save(path)
        print(f"  stored {dataset.n_samples} snapshots at {path}")
    else:
        from arcticgnn.spatial.ensemble import EmulationDataset
        from arcticgnn.spatial.grid import build_graph, build_grid
        dataset = EmulationDataset.load(path)
        grid = build_grid(100.0)
        graph = build_graph(grid)
        print(f"  reusing {dataset.n_samples} snapshots from {path}")

    tensors = GraphTensors.from_graph(graph)
    output = {"forcings": FORCINGS.tolist(), "n_samples": dataset.n_samples,
              "n_cells": int(grid.n_cells), "n_edges": int(graph.n_edges),
              "scores": [], "connectivity": {}}

    fitted = {}
    for architecture in ARCHITECTURES:
        print(f"fitting the {architecture} emulator")
        model, score = train_emulator(
            dataset, graph, architecture, seed=SEED,
            n_epochs=arguments.epochs, max_samples=arguments.max_samples)
        print("  " + score.describe())
        fitted[architecture] = model
        output["scores"].append({
            "architecture": architecture,
            "n_parameters": score.n_parameters,
            "variance_explained": score.variance_explained.tolist(),
            "rmse": score.rmse.tolist(),
            "rmse_persistence": score.rmse_persistence.tolist(),
            "thickness_variance_explained": score.thickness_variance_explained,
            "history": {"epoch": score.history.epoch,
                        "train_loss": score.history.train_loss,
                        "test_loss": score.history.test_loss}})

    # -- how far influence travels ----------------------------------------- #
    print("measuring the influence length scale")
    model_spatial = SpatialArcticModel(grid, graph, column=column)
    sources = choose_sources(grid, n_sources=48)

    train_index, _ = dataset.split(seed=SEED)
    mean = dataset.features[train_index].reshape(-1, dataset.n_features).mean(0)
    std = dataset.features[train_index].reshape(-1, dataset.n_features).std(0) + 1e-6

    for season, time in SEASON_MONTHS.items():
        # A snapshot of the control ensemble member at this point in the season.
        month = int(round(time * 12)) % 12 + 1
        candidates = np.where((dataset.month == month)
                              & (dataset.forcing == FORCINGS[0]))[0]
        if candidates.size == 0:
            continue
        sample = int(candidates[candidates.size // 2])
        features = dataset.features[sample]

        state = np.stack([
            -features[:, 0] * 9.7,          # thickness back to enthalpy
            features[:, 2]], axis=-1)

        truth = physical_length_scale(model_spatial, grid, state, sources,
                                      t0=time, label=season)
        entry = {"physical": truth.summary()}

        for architecture in ("attention", "edge"):
            model = fitted[architecture]
            learned = emulator_length_scale(model, tensors, grid, features,
                                            mean, std, sources, label=season)
            weights = attention_length_scale(model, tensors, grid, features,
                                             mean, std, sources, label=season)
            entry[architecture] = {
                "emulator": learned.summary(),
                "attention": weights.summary(),
                "emulator_versus_physical": vars(compare(truth, learned)),
                "attention_versus_physical": vars(compare(truth, weights))}

        output["connectivity"][season] = entry
        median, low, high = truth.summary()
        print(f"  {season}: the model itself spreads influence "
              f"{median:.0f} km (interquartile {low:.0f} to {high:.0f})")
        for architecture in ("attention", "edge"):
            e = entry[architecture]
            print(f"    {architecture:11s} emulator {e['emulator'][0]:6.0f} km, "
                  f"attention {e['attention'][0]:6.0f} km, "
                  f"correlation with the model "
                  f"{e['emulator_versus_physical']['correlation']:.2f} and "
                  f"{e['attention_versus_physical']['correlation']:.2f}")

    result = RUNS_DIR / "emulators.json"
    result.write_text(json.dumps(output, indent=1))
    print(f"written {result}")


if __name__ == "__main__":
    main()
