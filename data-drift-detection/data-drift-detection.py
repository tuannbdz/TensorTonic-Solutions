import numpy as np
def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    reference_counts = np.asarray(reference_counts)
    production_counts = np.asarray(production_counts)
    reference_counts = reference_counts / np.sum(reference_counts)
    production_counts = production_counts / np.sum(production_counts)
    TVD = float(0.5 * np.sum(np.abs(reference_counts - production_counts)))
    return {
            "score": TVD,
            "drift_detected": TVD > threshold
            }