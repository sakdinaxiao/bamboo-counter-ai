import pytest
from src.global_tracker import GlobalTracker

def test_global_tracker_init_parameters():
    tracker = GlobalTracker(
        merge_distance=10.0,
        update_rate=0.5,
        ransac_threshold=3.0,
        orb_features=500
    )
    assert tracker.merge_distance == 10.0
    assert tracker.update_rate == 0.5
    assert tracker.ransac_threshold == 3.0
    assert tracker.orb_features == 500
