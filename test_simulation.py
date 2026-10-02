import numpy as np

from shelfsentinel.schema import KP, SIG, Scenario
from shelfsentinel.simulation import REGISTRY, generate_batch, make_stores, subtype_weights
from shelfsentinel.simulation.kinematics import BODY, build_skeleton


def test_registry_covers_every_scenario():
    assert len(REGISTRY) == 32
    theft = {s.scenario for s in REGISTRY.values() if s.scenario != Scenario.NORMAL}
    assert theft == set(Scenario) - {Scenario.NORMAL}
    assert abs(subtype_weights().sum() - 1.0) < 1e-9
    share = sum(w for w, s in zip(subtype_weights(), REGISTRY.values(), strict=True) if s.scenario != Scenario.NORMAL)
    assert 0.08 < share < 0.20  # theft is enriched but still the minority


def test_generation_is_deterministic():
    a = generate_batch(40, seed=9)
    b = generate_batch(40, seed=9)
    c = generate_batch(40, seed=10)
    np.testing.assert_array_equal(a.keypoints, b.keypoints)
    np.testing.assert_array_equal(a.signals, b.signals)
    assert not np.array_equal(a.keypoints, c.keypoints)


def test_every_behaviour_runs_and_is_labelled():
    names = list(REGISTRY)
    batch = generate_batch(len(names) * 3, seed=4, subtypes=names)
    assert np.isfinite(batch.keypoints).all()
    assert np.isfinite(batch.signals).all()
    for row in batch.meta.itertuples():
        spec = REGISTRY[row.subtype]
        assert row.scene == int(spec.scene)
        assert row.scenario == int(spec.scenario)
        if spec.scenario == Scenario.NORMAL:
            assert row.value_at_risk_gbp == 0.0
        else:
            assert row.value_at_risk_gbp > 0.0


def test_signal_ranges(batch):
    s = batch.signals
    for chan in ("hand_item_l", "hand_item_r", "bag_conf", "cont_conf", "scan_match"):
        assert s[:, :, SIG[chan]].min() >= 0.0
        assert s[:, :, SIG[chan]].max() <= 1.0
    assert s[:, :, SIG["container_count"]].min() >= 0.0
    conf = batch.keypoints[..., 2]
    assert conf.min() >= 0.0 and conf.max() <= 1.0


def test_arms_respect_reach():
    rng = np.random.default_rng(0)
    n, t = 16, 10
    hip = np.tile(np.array([0.5, 0.7], dtype=np.float32), (n, t, 1))
    scale = np.full(n, 0.5, dtype=np.float32)
    far = rng.uniform(-1.0, 2.0, size=(n, t, 2)).astype(np.float32)  # mostly unreachable targets
    ones, zeros = np.ones((n, t), np.float32), np.zeros((n, t), np.float32)
    xy, vis = build_skeleton(hip, scale, ones, far, far, zeros, zeros, zeros, np.ones(n, np.float32))
    arm = BODY.arm_length * 0.5
    for side in ("left", "right"):
        reach = np.linalg.norm(xy[:, :, KP[f"{side}_wrist"]] - xy[:, :, KP[f"{side}_shoulder"]], axis=-1)
        upper = np.linalg.norm(xy[:, :, KP[f"{side}_elbow"]] - xy[:, :, KP[f"{side}_shoulder"]], axis=-1)
        assert reach.max() <= arm + 1e-4  # a wrist never leaves the shoulder by more than an arm
        assert upper.max() <= arm * 0.5 * 1.15  # projected upper arm stays close to its true length
    assert vis.shape == (n, t, 17)


def test_reachable_target_is_reached():
    hip = np.array([[[0.5, 0.7]]], dtype=np.float32)
    target = np.array([[[0.42, 0.62]]], dtype=np.float32)
    ones, zeros = np.ones((1, 1), np.float32), np.zeros((1, 1), np.float32)
    xy, _ = build_skeleton(hip, np.array([0.5], np.float32), ones, target, target, zeros, zeros, zeros, np.ones(1, np.float32))
    np.testing.assert_allclose(xy[0, 0, KP["left_wrist"]], target[0, 0], atol=1e-4)


def test_store_profiles_are_valid():
    stores = make_stores(12, seed=1)
    assert len({s.store_id for s in stores}) == 12
    for s in stores:
        assert abs(sum(s.quality_p) - 1) < 1e-6
        assert abs(sum(s.subtype_p) - 1) < 1e-6


def test_prefix_labels_follow_what_has_happened_so_far():
    from shelfsentinel.simulation import prefix_labels

    scen = np.array([0, 1, 1, 1, 3, 3, 5, 5, 7, 4])
    event = np.array([np.nan, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 9.0, np.nan])
    seen = 5.0  # seconds of the clip seen so far
    labels = prefix_labels(scen, event, seen)
    assert labels[0] == 0  # honest clips stay honest
    assert labels[1] == 0  # the concealment has not happened yet
    assert prefix_labels(scen, event, 6.3)[1] == -1  # too close to the act: dropped
    assert prefix_labels(scen, event, 7.0)[1] == 1  # the act is complete
    assert labels[4] == -1 and prefix_labels(scen, event, 7.0)[4] == 3  # a sweep in progress is never called normal
    assert labels[6] == -1 and prefix_labels(scen, event, 3.5)[6] == 0  # ticket switch: scan precedes bagging
    assert labels[8] == 7  # at the doors the evidence exists from the first frame
    assert labels[9] == -1  # theft clip whose act fell outside the window


def test_confirmation_rule_debounces_alerts():
    from shelfsentinel.evaluation.replay import confirmed

    tier = np.array([[0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0]])
    assert confirmed(tier, 2).tolist() == [[False, False, False, False, True, False], [False, True, True, False, False, False]]
    assert confirmed(tier, 1).tolist() == (tier >= 1).tolist()
