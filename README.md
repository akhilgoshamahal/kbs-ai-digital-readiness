# Knowledge-Based AI for Digital Transformation Readiness

This repository contains the reproducible experiment for the Short Communication:

**Knowledge-Based AI Decision Support Using Enterprise Innovation Data for Digital Transformation and Entrepreneurship Readiness**

The study merges the European Commission's European Innovation Scoreboard panel with Eurostat's Digital Intensity Index. It evaluates whether explicit knowledge rules add decision value to probabilistic machine-learning forecasts of next-year enterprise digital-readiness tiers.

## Reproduce locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
```

The script downloads the official public sources, builds a leakage-controlled country-year panel, runs country-grouped cross-validation, evaluates machine-learning and knowledge-based baselines, performs grouped bootstrap inference, and writes exactly two main figures and two main tables to `results/`.

## Evidence boundaries

- The outcome is a next-year, country-relative readiness tier derived from Eurostat enterprise digital-intensity statistics; it is not an audited firm-level maturity label.
- Country groups are held out during evaluation to reduce geographic leakage.
- Knowledge thresholds are estimated only from each training fold.
- The asymmetric decision cost is a transparent scenario assumption, not a measured monetary utility.
- The study supports screening and resource-prioritization research; it does not certify an enterprise or national economy as transformation-ready.

## Sources

- European Commission, European Innovation Scoreboard 2025 database.
- Eurostat, Digital Intensity by size class of enterprise, dataset `isoc_e_dii`, DOI: 10.2908/isoc_e_dii.

## License

Code is released under the MIT License. Source data remain subject to their respective European Commission and Eurostat terms.
