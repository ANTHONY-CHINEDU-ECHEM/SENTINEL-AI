import numpy as np
import pandas as pd
import pytest

from shelfsentinel.schema import KEYPOINT_NAMES, N_SIGNALS, SCENARIO_SCENE, THEFT_CLASSES, ClipBatch, Scenario


def test_constants_are_consistent():
    assert len(KEYPOINT_NAMES) == 17
    assert len(THEFT_CLASSES) == 7
    assert set(SCENARIO_SCENE) == set(THEFT_CLASSES)
    assert Scenario.NORMAL == 0


def test_batch_validates_shapes():
    meta = pd.DataFrame({"clip_id": ["a", "b"]})
    with pytest.raises(ValueError):
        ClipBatch(np.zeros((2, 8, 16, 3)), np.zeros((2, 8, N_SIGNALS)), meta)
    with pytest.raises(ValueError):
        ClipBatch(np.zeros((2, 8, 17, 3)), np.zeros((2, 9, N_SIGNALS)), meta)
    with pytest.raises(ValueError):
        ClipBatch(np.zeros((3, 8, 17, 3)), np.zeros((3, 8, N_SIGNALS)), meta)


def test_select_and_prefix(batch):
    sub = batch.select([0, 5, 9])
    assert len(sub) == 3
    assert sub.meta["clip_id"].tolist() == batch.meta["clip_id"].iloc[[0, 5, 9]].tolist()
    assert batch.prefix(10).n_frames == 10
    assert batch.prefix(10_000).n_frames == batch.n_frames
    mask = batch.meta["scenario"].to_numpy() != 0
    assert len(batch.select(mask)) == mask.sum()


def test_save_and_load_roundtrip(batch, tmp_path):
    small = batch.select(slice(0, 20))
    path = small.save(tmp_path / "clips")
    loaded = ClipBatch.load(path)
    assert len(loaded) == 20
    assert loaded.fps == small.fps
    np.testing.assert_allclose(loaded.keypoints, small.keypoints, atol=2e-3)  # stored as float16
    np.testing.assert_allclose(loaded.signals, small.signals, atol=1e-6)
    assert loaded.meta["subtype"].tolist() == small.meta["subtype"].tolist()
