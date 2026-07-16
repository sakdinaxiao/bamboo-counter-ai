# Global Tracker Ablation Study Design

## 1. Context and Purpose
The current `GlobalTracker` implementation hardcodes several mathematical hyperparameters, which makes it read like an ad-hoc workaround rather than a novel spatial deduplication algorithm. To prepare for academic publication, we need to formalize these parameters and conduct an ablation study to prove their optimality. 

## 2. Architecture Updates
*   **Remove Global State:** Remove `cv2.setRNGSeed(42)` from the module level in `src/global_tracker.py` to prevent polluting the global OpenCV state. We will set seeds explicitly in our evaluation scripts instead.
*   **Expose Hyperparameters:** Update the `GlobalTracker.__init__` signature to accept the following parameters with their current values as defaults:
    *   `merge_distance` (default: 8.0)
    *   `update_rate` (default: 0.25)
    *   `ransac_threshold` (default: 5.0)
    *   `orb_features` (default: 1000)
*   **Update Consumers:** Ensure `src/main.py` explicitly instantiates `GlobalTracker` using the correct parameters (it currently uses `merge_distance=12.5`), removing "magic numbers" from being hidden deep in the codebase.

## 3. Ablation Study Framework
*   **`ablation_study.py`:** We will replace `test_merge_distance.py` with a new, comprehensive script: `ablation_study.py`.
*   **Grid Search:** The script will perform a grid search over key parameter ranges (e.g., `merge_distance` and `update_rate`).
*   **Metrics & Output:** For each combination, the script will calculate the final count, compute the absolute error against a known ground truth, and export all results to `ablation_results.csv` for use in the paper.
