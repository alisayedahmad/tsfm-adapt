# TSFM-Adapt

Testing whether Time Series Foundation Models need domain adaptation on energy data. Short answer: they don't, but your evaluation protocol might make you think they do.

## The question

Chronos, TimesFM and Moirai are pretrained on massive time series corpora. The assumption in recent literature is that they degrade on domain-specific energy data (buildings, solar, grid) and need adaptation. We built a lightweight adapter to fix that.

The adapter didn't help. But the reason it didn't help is more interesting than if it had worked.

## What actually happened

### Phase 1: Chronos matches the naive baseline

We evaluated Chronos-T5-Small against SeasonalNaive, AutoETS and AutoTheta on 20 BDG2 buildings (hourly electricity, 24h-ahead, 8 rolling windows).

```
       Method  MASE_median
SeasonalNaive        0.271
      Chronos        0.271
      AutoETS        0.317
    AutoTheta        1.297
```

Chronos ties with the naive. No gap to close. We spent two weeks building and tuning an adapter anyway.

### Phase 2: the adapter does nothing useful

We tested six configurations of an input adapter (bottleneck module before the frozen backbone, trained with a proxy forecast head). Best result: +4.1% improvement, within noise. An output correction approach (OLS on Chronos residuals) also failed.

The adapter was not broken. There was simply nothing to correct.

### Phase 3-4: testing other setups

We tried 168h-ahead on BDG2 and 24h-ahead on NREL solar (137 PV plants). Chronos held up or won in both cases. No domain gap anywhere.

### Phase 5: three models, same conclusion (or so we thought)

We added TimesFM (200M) and Moirai (14M) to the comparison. On the same BDG2 evaluation:

```
Chronos   0.271  (tied with naive)
TimesFM   0.385  (+41% worse than naive)
Moirai    0.432  (+58% worse than naive)
```

This looked like a real finding: Chronos holds, the other two don't. We ran diagnostics, verified the wrappers had no bugs, and were about to write this up.

### The protocol was wrong

Then we checked what day of the week we were evaluating. Every single cutoff fell on a Saturday night, so every forecast started on a Sunday. With `step_size=168` (exactly one week), the evaluation never saw a Monday, a Wednesday, or any other day.

We reran everything with `step_size=48` (362 cutoffs instead of 8, all seven days covered, 7240 evaluations per model):

```
            Naive  Chronos  TimesFM  Moirai
Monday      1.773    0.963    0.706   0.672
Tuesday     0.544    0.463    0.518   0.504
Wednesday   0.434    0.395    0.458   0.444
Thursday    0.458    0.408    0.463   0.475
Friday      0.501    0.436    0.498   0.483
Saturday    1.598    0.505    0.635   0.968
Sunday      0.359    0.387    0.503   0.544

OVERALL     0.586    0.466    0.530   0.543
```

All three TSFMs beat the naive overall. The "gap" for TimesFM and Moirai was an artifact of evaluating only on Sundays, the one day where the naive baseline (repeat yesterday = repeat Saturday, which resembles Sunday) is naturally strong.

The real pattern: TSFMs dominate on transition days (Monday: naive repeats Sunday for a workday, Saturday: naive repeats Friday for a weekend day). These are exactly the cases where understanding weekly structure matters, and where a model pretrained on diverse time series has a genuine advantage over a method that copies yesterday.

## What this means

**For practitioners.** TSFMs work on energy data out of the box. On BDG2 hourly buildings, Chronos-T5-Small beats a daily naive by 20% overall. No adaptation needed for this use case.

**For researchers.** Your evaluation step size can flip your conclusion. A step that is a multiple of the dominant period (168h = 1 week for buildings) locks your evaluation to one phase of the cycle. This is not a minor detail. It turned "all three models lose" into "all three models win" in this study.

**For the adapter idea.** The hypothesis was reasonable but the premise was wrong on this data. The adapter machinery is in the repo and works correctly. It just has nothing to correct here. The interesting cases would be data where TSFMs genuinely fail: series with no periodic structure (wind generation), or deployment settings where the TSFM encounters patterns absent from its pretraining data.

## Repo structure

```
src/
  data/           BDG2 loader, NREL solar loader, preprocessing
  eval/           MASE, sMAPE
  models/
    baselines/    SeasonalNaive, AutoETS, AutoTheta (statsforecast)
    tsfm_wrappers/  Chronos, TimesFM, Moirai (one venv each)
    adapter/      input adapter, output correction (Phase 2)
  training/       adapter training loop

experiments/      phase scripts, diagnostics, analysis
tests/            unit tests (metrics, adapter shapes, output correction, NREL loader)
results/          saved results per phase
```

## Setup

Each TSFM needs its own virtual environment because their dependencies conflict.

```bash
# Chronos (main venv)
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install chronos-forecasting statsforecast pandas pyarrow pytest

# TimesFM
python -m venv .venv-tsfm
.venv-tsfm\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install timesfm pandas pyarrow

# Moirai
python -m venv .venv-moirai
.venv-moirai\Scripts\activate
pip install uni2ts pandas pyarrow
```

BDG2 data: `electricity_cleaned.csv` in `data/raw/` (download from Zenodo, Building Data Genome 2).
NREL solar: downloaded automatically on first run.

## Running

```bash
# tests (main venv)
python -m pytest tests/ -v

# full evaluation, one model per venv
.venv\Scripts\activate
python experiments\run_gap_by_day_multi.py --model chronos

.venv-tsfm\Scripts\activate
python experiments\run_gap_by_day_multi.py --model timesfm

.venv-moirai\Scripts\activate
python experiments\run_gap_by_day_multi.py --model moirai

# aggregate
python experiments\run_gap_by_day_multi.py --aggregate
```

## Key numbers

| Metric | Value |
|--------|-------|
| Buildings evaluated | 20 (BDG2, seed=42) |
| Evaluation windows | 362 per model (step=48h) |
| Total evaluations | 7240 per model |
| Forecast horizon | 24h |
| Context length | 168h |
| Days of week covered | all 7 |

## Stack

PyTorch 2.x, HuggingFace Transformers, chronos-forecasting, timesfm, uni2ts, statsforecast, pandas, numpy, pytest.

## License

MIT