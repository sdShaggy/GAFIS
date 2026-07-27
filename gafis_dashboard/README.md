# GAFIS Case Dashboard

A Streamlit report dashboard for the GAFIS project (AI-assisted forensic
enhancement + minutiae localization + matching for latent fingerprints).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## What's inside

All numbers on every page are taken directly from your executed notebooks
(GAFIS_dataset_pipeline, GAFIS_pix2pix_training, Localization&Matching,
SourceAFIS_matching) — dataset stats, the full 75-epoch Pix2Pix training
log, PSNR/SSIM results, YOLOv8 training metrics, and SourceAFIS
genuine/impostor matching + FAR/FRR. The one exception is the FAR/FRR
*curve* on the SourceAFIS page, which is modeled from summary statistics
and clearly labeled as illustrative — the source notebook only printed a
single measured operating point (threshold=40).

## Pages

- **Case Overview** — abstract, headline results, pipeline diagram, scope
- **Stage 1** — dataset & synthetic degradation stats
- **Stage 4** — Pix2Pix architecture, full training curve, PSNR/SSIM
- **Stage 5** — YOLOv8 minutiae localization training + aggregate results
- **Stage 6** — SourceAFIS matching, pass rates, FAR/FRR
- **Live Sample Walkthrough** — one validation print carried through the full pipeline
- **Limitations & Scope** — honest accounting of what was cut vs. the original proposal
- **Math Appendix** — every formula behind every metric shown

## Editing the numbers

All data lives in clearly-labeled Python dicts/DataFrames near the top of
`app.py` (search for "REAL DATA"). Update those if you re-run any stage
and get new numbers — nothing else needs to change.
