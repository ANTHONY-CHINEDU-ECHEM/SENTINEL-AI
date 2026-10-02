import numpy as np

from shelfsentinel.adapters import batch_to_frames, clips_from_track, resample_track, tracks_from_frames
from shelfsentinel.cli import main
from shelfsentinel.evaluation.replay import risk_curves
from shelfsentinel.explain import build_timeline
from shelfsentinel.features import extract_features
from shelfsentinel.schema import Scene
from shelfsentinel.viz.render import RenderStyle, contact_sheet, render_frame
from shelfsentinel.viz.storyboard import pick_keyframes, render_storyboard


def test_render_frame_for_every_scene(batch):
    style = RenderStyle(width=320, height=180, supersample=1)
    seen = set()
    for i, scene in enumerate(batch.meta["scene"]):
        if scene in seen:
            continue
        seen.add(scene)
        img = render_frame(batch, i, 30, risk=0.91, tier="priority", label="Test", style=style)
        assert img.size == (320, 180)
        assert np.asarray(img).std() > 5  # not a blank image
    assert seen == {0, 1, 2, 3}
    sheet = contact_sheet([render_frame(batch, 0, f, style=style) for f in (0, 10, 20)], cols=2)
    assert sheet.size[0] > 640


def test_keyframes_are_spread_out():
    timeline = [{"frame": f, "t": f / 5, "kind": "x", "text": "x", "weight": w} for f, w in ((10, 3), (11, 3), (12, 2), (30, 2), (50, 1))]
    keys = pick_keyframes(timeline, 64)
    frames = [k["frame"] for k in keys]
    assert len(frames) == 4 and frames == sorted(frames)
    assert min(np.diff(frames)) >= 4
    assert 10 in frames and 30 in frames
    assert len(pick_keyframes([], 64)) == 4  # falls back to context frames


def test_storyboard_renders(tiny_project, batch):
    model = tiny_project["model"]
    one = batch.select([0])
    counts, risk, tiers = risk_curves(model, one, min_frames=16, step=16)
    a = model.assess(extract_features(one)).iloc[0]
    assessment = {"tier": a["tier"], "scenario": a["flag_title"], "risk": float(a["risk"])}
    img = render_storyboard(one, 0, assessment, build_timeline(one, 0), counts, risk[0], tiers[0], (0.5, 0.9), "Title", "subtitle", "footer text")
    assert img.size[0] > 1200 and img.size[1] > 900


def test_pose_adapter_roundtrip(batch):
    size = (1920, 1080)
    frames = batch_to_frames(batch, 3, size, track_id=7)
    tracks = tracks_from_frames(frames, size)
    assert list(tracks) == [7]
    numbers, kp = tracks[7]
    np.testing.assert_allclose(kp, batch.keypoints[3], atol=1e-5)
    clips = clips_from_track(resample_track(numbers, kp, fps_in=5.0), Scene.AISLE, track_id=7)
    assert len(clips) == 1 and clips.n_frames == 64
    feats = extract_features(clips)
    assert np.isfinite(feats.to_numpy()).all()


def test_resample_marks_gaps_as_unseen():
    numbers = np.array([0, 5, 10, 40, 45])  # 25 fps source with a one second hole
    kp = np.ones((5, 17, 3), dtype=np.float32)
    out = resample_track(numbers, kp, fps_in=25.0, fps_out=5.0)
    assert out.shape[0] == 10
    assert out[0, 0, 2] == 1.0 and out[4, 0, 2] == 0.0 and out[-1, 0, 2] == 1.0


def test_long_track_is_windowed():
    kp = np.random.default_rng(0).random((200, 17, 3)).astype(np.float32)
    clips = clips_from_track(kp, Scene.EXIT, n_frames=64, stride=32)
    assert len(clips) == 5
    np.testing.assert_array_equal(clips.keypoints[1], kp[32:96])


def test_cli_help_exits_cleanly(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "gallery" in capsys.readouterr().out
