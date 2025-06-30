# SumoBot SLM Hybrid Dataset Pipeline

This repository provides tools and scripts for building a **SumoBot Strategic Language Model (SLM)** dataset using both simulated and real battle log data. The pipeline allows for generating synthetic situations, parsing real logs, and merging both for robust SLM model training.

---

## File Structure

* **battle\_log\_parser.py**
  Parses `.json` battle logs into a simple CSV dataset (`sumobot_slm_dataset.csv`) with `situation` and `strategy` columns.
* **generate\_sumobot\_dataset.py**
  Generates a hybrid synthetic dataset (`sumobot_auto_dataset.csv`) that mimics both numeric and log-like data fields.
* **combine\_datasets.py**
  Merges simulation data and log data into a single dataset (`sumobot_slm_merged.csv`) suitable for SLM training.
* **sumobot\_auto\_dataset.csv**
  Synthetic data generated for model training (output).
* **sumobot\_slm\_dataset.csv**
  Real situation-action pairs parsed from gameplay logs (output).
* **sumobot\_slm\_merged.csv**
  Final merged dataset (output).

---

## Dataset Details

### Synthetic Dataset (`sumobot_auto_dataset.csv`)

* **Columns:**

  * *enemy\_distance, enemy\_angle, edge\_distance, center\_distance, enemy\_stuck, enemy\_behind, skill\_ready, dash\_ready*
    Numeric and boolean features simulating game state.
  * *situation*
    English sentence describing the above features.
  * *category, round*
    Simulated log metadata.
  * *strategy*
    Action label (e.g., `turn_left_45+accelerate`).

### Log Dataset (`sumobot_slm_dataset.csv`)

* **Columns:**

  * *situation*
    Textual representation of the pre-action state.
  * *strategy*
    Action taken in that situation.

### Merged Dataset (`sumobot_slm_merged.csv`)

* Combines and deduplicates both sources, forming a rich SLM training set.

---

## How to Use

1. **Parse Game Logs:**

   ```bash
   python battle_log_parser.py
   ```

   This will generate `sumobot_slm_dataset.csv` from all `battle_*.json` files in the folder.

2. **Generate Synthetic Dataset:**

   ```bash
   python generate_sumobot_dataset.py
   ```

   This produces `sumobot_auto_dataset.csv` containing randomized, yet realistic, state-action pairs.

3. **Merge Datasets:**

   ```bash
   python combine_datasets.py
   ```

   This script merges the synthetic and log datasets into `sumobot_slm_merged.csv`.

---