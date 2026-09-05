# Arctic sea ice: tipping thresholds and graph neural network emulators

Code for a study of abrupt transitions in Arctic sea ice. The work combines a
low order dynamical model, a spatially resolved model on real Arctic geography,
and graph neural network emulators trained on the latter.

## What the models are

**A seasonal column model** (`arcticgnn/physics`) carrying two prognostic
variables: the surface enthalpy, which describes either an ice cover or a warm
mixed layer, and the temperature of the water below the halocline. Two positive
feedbacks act on it. Thinning ice exposes darker ocean, which absorbs more
sunlight. Separately, a thinning ice cover lets wind stress reach the ocean
surface, mixing strengthens, and heat that had accumulated below the halocline
is released upward. The second feedback depends on a slow variable, which is
what allows the model to hold two different annual cycles under the same
forcing.

Shortwave forcing is computed from orbital geometry rather than prescribed as a
fitted cycle. This matters: the real polar summer is far brighter and shorter
than a low order fit suggests, and because the ice surface sits at the melting
point through that period, every additional watt goes into melting.

**A spatially resolved model** (`arcticgnn/spatial`) applies the same column
physics on an equal area polar grid with land taken from the Natural Earth
coastline, and couples the cells through atmospheric heat transport, sea ice
drift representing the Beaufort Gyre and the Transpolar Drift, and a
sub-halocline circulation supplied with Atlantic Water through the Nordic Seas.
Ice leaves the domain across its open southern boundary and is reflected at
coasts.

**Emulators** (`arcticgnn/models`) are trained on that model. Four are provided:
a baseline that sees each cell alone, message passing with weights fixed by the
graph, message passing with learned weights, and learned weights additionally
conditioned on the geometry of each connection. The layers are written directly
so that the attention coefficients used in the forward pass can be inspected
exactly.

## Why the spatial model exists

Its transport parameters are prescribed, so the distance over which one location
influences another is known in advance. That distance can be measured directly,
by perturbing one cell and watching where the response appears, and the value
recovered from a trained network's attention weights can be checked against it.
An attention length scale without such a reference is a number with nothing to
compare it to.

## Calibration

Both models have one free quantity in their surface energy budget. The seasonal
amplitude of the net upward flux is fixed beforehand from observed non-solar
surface fluxes over Arctic sea ice and is never fitted; leaving it free lets a
solver match observed thickness with a budget in which the winter surface gains
heat. The column model is then set to the observed annual mean thickness of the
central Arctic and the spatial model to the observed September extent, which
leaves the seasonal range and the March extent as predictions to be tested.

## Layout

```
arcticgnn/
  config.py       paths, constants, unit conventions
  physics/        column model, insolation, calibration
  spatial/        grid and land mask, transport, spatial model, ensemble
  models/         emulator architectures and training
  dynsys/         continuation of periodic states, rate induced tipping,
                  early warning indicators
  interpret/      influence length scales measured three ways
  viz/            figure styling and polar maps
scripts/          numbered pipeline stages, runnable from the command line
tests/            unit tests
```

## Data

All input data are public and small. The satellite record of Northern Hemisphere
sea ice extent comes from the National Snow and Ice Data Center Sea Ice Index
and is downloaded on first use. Coastlines come from Natural Earth through
Cartopy.

## Running the pipeline

```bash
export ARCTICGNN_ROOT=/path/to/project     # holds data, runs and figures
pip install -e .

python scripts/01_calibrate.py             # calibrate both models
python scripts/02_bifurcation.py           # branches, folds, regime map
python scripts/03_tipping.py               # ramps at different rates
python scripts/04_emulators.py             # ensemble, emulators, connectivity
python scripts/05_warning.py               # early warning indicators
python scripts/09_figures.py               # figures
```

Each stage writes a single file under `runs/` and reads only what earlier stages
wrote, so any stage can be repeated on its own.

## Requirements

Python 3.8 or later, NumPy, SciPy, Matplotlib, PyTorch, Cartopy and Shapely.
No graph learning library is needed. Everything runs on a laptop; the emulator
training is the slowest step and uses four processor threads by default.

## Licence

MIT, see `LICENSE`.
