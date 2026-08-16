# TSFM-Adapt

Lightweight domain adaptation for Time Series Foundation Models on energy data.

The idea: instead of fine-tuning the full model on every new site, insert a small adapter module (inspired by LoRA) that learns to correct domain-specific patterns with minimal target data. The backbone stays frozen.

## What this tests

Can a small adapter (5K-70K params) recover most of the performance gap between zero-shot TSFM inference and full fine-tuning, using only 1 week of unlabeled target data?

Current scope: Chronos-T5-Small on BDG2 (building energy consumption, 1578 buildings, hourly).

## Results so far

### Phase 1: baselines + zero-shot

20 buildings sampled from BDG2 (seed=42, completeness >= 80%, variance > 0.1).
Rolling forecast: 8 windows, step 7 days, horizon 24h, context 168h.
160 total evaluations.

```
       Method  MASE_median  MASE_mean  sMAPE_mean
SeasonalNaive        0.271      0.553         8.6
      AutoETS        0.317      0.655        12.3
    AutoTheta        1.297      1.464        37.5
      Chronos        0.281      0.550         8.6
```

Chronos zero-shot matches SeasonalNaive. It does not beat it.
This is already informative: on BDG2 hourly data with 24h horizon, the domain gap between Chronos and a tuned-per-site baseline is small. There is not much room for an adapter to improve things.

### Phase 2: adapter experiments

Tested two adapter approaches against zero-shot Chronos on the same 20 buildings and 8 evaluation windows.

**Input adapter** (bottleneck module before Chronos, trained with a proxy forecast head):

| Config | Params | Data | MASE adapted | MASE zero-shot | Delta |
|--------|--------|------|-------------|----------------|-------|
| per-building, no regularization | 71K | 7 days | 3.566 | 0.237 | -1405% |
| per-building, identity reg lambda=1.0 | 71K | 7 days | 0.239 | 0.259 | +7.7% |
| cross-building, identity reg, Chronos-validated checkpoint | 71K | 7 days | 0.260 | 0.271 | +4.1% |

Best result: +4.1% improvement (median MASE). Below the noise floor.

**Output correction** (per-step affine correction on Chronos predictions, fit by OLS):

Calibrated on 7 days immediately before test period, per-building, 73 calibration pairs each. Correction factors were reasonable (scale ~0.95-1.0) but did not transfer well across the 8-week test window due to nonstationarity. Net result negative.

### Interpretation

The adapter does not produce meaningful gains on this setup. The main reason is not the adapter architecture. It is that the domain gap is too small to begin with. Chronos already matches the seasonal naive baseline, so there is almost nothing to recover.

This is a valid negative result. It tells us where lightweight adaptation does NOT help: when the TSFM is already performing near the domain-specific baselines. The interesting test cases are where the gap is large (longer horizons, weather-dependent renewables, grid-level demand with regulatory structure).

## Repo structure

```
src/
  data/          loading, sampling, preprocessing, windowing
  eval/          MASE, sMAPE, evaluation utilities
  models/
    baselines/   SeasonalNaive, AutoETS, AutoTheta (statsforecast)
    tsfm_wrappers/  Chronos wrapper
    adapter/     input adapter, output correction
  training/      adapter training loop

experiments/     phase scripts (run_phase1.py, run_phase2.py, run_phase2b.py)
tests/           unit tests (metrics, adapter shapes, output correction)
results/         saved results per phase
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Data: BDG2 `electricity_cleaned.csv` goes in `data/raw/`. Download from Zenodo (Building Data Genome 2).

## Running

```
python -m pytest tests/ -v                  # all tests
python experiments/run_phase1.py            # baselines + zero-shot (~45 min)
python experiments/run_phase2.py            # input adapter (~5 min)
python experiments/run_phase2b.py           # output correction (~15 min)
```

## Next steps

- Test on longer horizons (168h) where Chronos should struggle more
- Test on ENTSO-E (grid demand) or NREL (solar/wind) where domain shift is stronger
- Produce fine-tuning upper bound for proper gap measurement
- If a real gap exists, the adapter machinery is ready to exploit it

## Stack

PyTorch, HuggingFace Transformers, chronos-forecasting, statsforecast, pandas, numpy, pytest.

## License

MIT