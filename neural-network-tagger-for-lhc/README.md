# Neural-Network Tagger for Hadronic V→qq̄ Jets

Welcome! This repo contains your applied ML project for the course.  
You’ll train a tiny neural network to tag boosted **W/Z→qq̄** jets and compare it to a cut-based baseline, then apply the model to real ATLAS Open Data.  
Learn more about the data source here: https://opendata.atlas.cern/

---

## Start here

1) **Read `assignment.pdf` first** — it explains the task, metrics (ROC, purity), and what to submit.  
2) **Set up a Python environment** (e.g. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`).  
3) **Run the baseline** in `data-exercise-template.py` to understand the cut-based selection and purity calculation.  
4) **Implement/train your MLP**, make ROC & purity curves, pick a working point, and **apply to `jets.csv` (data)**.

---

## What’s in this repo

- `assignment.pdf` — the project brief and deliverables (read this!).
- `data-exercise-template.py` — starter code:
  - `Particle`/`Event` classes
  - CSV loading
  - Cut-based selection + purity scaffold
- `jets.csv` — **ATLAS Open Data** (flattened; leading lepton + large-R jets).
- `pythia.csv` — **MC** in the same format plus `v_true` (truth label for large-R jets).
- `requirements.txt` — minimal Python deps (numpy, matplotlib, torch, etc.).
- `aux/` *(optional, not required for the assignment)*  
  - `main213.cc` — PYTHIA generator used to produce `pythia.csv`.  
  - `skimevents.py` — ROOT→CSV skimmer for the open data format.

---

## Quick commands

```bash
# create & activate a virtual env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the baseline script
python data-exercise-template.py
