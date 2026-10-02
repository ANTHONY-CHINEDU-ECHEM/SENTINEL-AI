"""CCTV style renderer.

Draws what the system *sees*: an anonymous skeleton, item and bag detections,
zone outlines and point of sale events, on top of a stylised scene. Nothing in
these images is a photograph. They exist so a reader can look at a clip and
understand why the model scored it the way it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..scenes import layout_for
from ..schema import KP, SIG, SKELETON_EDGES, ClipBatch, Scene

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
INK = (10, 14, 22)
PANEL = (17, 24, 38)
GRID = (38, 50, 72)
TEXT = (226, 232, 240)
MUTED = (140, 152, 172)
MINT = (94, 234, 212)
AMBER = (251, 191, 36)
RED = (248, 113, 113)
GREEN = (74, 222, 128)
VIOLET = (167, 139, 250)
SKY = (125, 211, 252)

TIER_COLOURS = {"clear": GREEN, "review": AMBER, "priority": RED}

_FONT_FILES = {
    "sans": ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"),
    "bold": ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"),
    "mono": ("DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf", "LiberationMono-Bold.ttf"),
}


@lru_cache(maxsize=64)
def font(kind: str, size: int) -> ImageFont.ImageFont:
    """Load a TrueType font with graceful fallbacks."""
    candidates = list(_FONT_FILES[kind])
    try:  # matplotlib ships DejaVu, which makes this work on any platform
        from matplotlib import font_manager

        weight = "bold" if kind in ("bold", "mono") else "normal"
        family = "DejaVu Sans Mono" if kind == "mono" else "DejaVu Sans"
        candidates.append(font_manager.findfont(font_manager.FontProperties(family=family, weight=weight)))
    except Exception:  # pragma: no cover - matplotlib is a hard dependency
        pass
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class RenderStyle:
    width: int = 960
    height: int = 540
    supersample: int = 2
    min_conf: float = 0.30
    show_hud: bool = True
    font_scale: float = 1.0


# --------------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------------- #
_PRODUCT_HUES = (
    (176, 96, 84),
    (92, 132, 170),
    (196, 164, 92),
    (96, 150, 118),
    (150, 110, 160),
    (184, 130, 96),
    (110, 150, 168),
    (160, 160, 150),
)


def _vgrad(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    ramp = np.linspace(0.0, 1.0, h)[:, None, None]
    arr = np.array(top)[None, None, :] * (1 - ramp) + np.array(bottom)[None, None, :] * ramp
    return Image.fromarray(np.repeat(arr, w, axis=1).astype(np.uint8), "RGB")


def _floor(img: Image.Image, y0: float, tint: tuple[int, int, int] = (40, 48, 62)) -> None:
    w, h = img.size
    top = int(y0 * h)
    img.paste(_vgrad((w, h - top), tint, tuple(int(c * 0.62) for c in tint)), (0, top))
    d = ImageDraw.Draw(img)
    line = tuple(int(c * 1.22) for c in tint)
    for k in range(-6, 19):  # converging tile lines
        x_top = w * (0.5 + (k - 6) * 0.07)
        x_bot = w * (0.5 + (k - 6) * 0.17)
        d.line([(x_top, top), (x_bot, h)], fill=line, width=max(1, w // 960))
    for frac in (0.12, 0.30, 0.55, 0.85):
        y = top + (h - top) * frac
        d.line([(0, y), (w, y)], fill=line, width=max(1, w // 960))


def _aisle_bg(w: int, h: int) -> Image.Image:
    rng = np.random.default_rng(42)
    img = _vgrad((w, h), (22, 28, 40), (30, 36, 50))
    _floor(img, 0.60)
    d = ImageDraw.Draw(img)
    shelf = layout_for(Scene.AISLE).zones["shelf"]
    x0, y0, x1, y1 = shelf.x0 * w, shelf.y0 * h, shelf.x1 * w, shelf.y1 * h
    d.rectangle([x0, y0, x1, y1], fill=(26, 32, 46), outline=(58, 68, 88), width=max(2, w // 480))
    rows = 4
    row_h = (y1 - y0) / rows
    for r in range(rows):
        base = y0 + (r + 1) * row_h
        x = x0 + 0.006 * w
        while x < x1 - 0.03 * w:
            pw = rng.uniform(0.018, 0.040) * w
            ph = rng.uniform(0.55, 0.86) * (row_h - 0.016 * h)
            hue = _PRODUCT_HUES[int(rng.integers(len(_PRODUCT_HUES)))]
            shade = rng.uniform(0.50, 0.80)
            col = tuple(int(c * shade) for c in hue)
            run = int(rng.integers(1, 4))  # facings of the same product
            for _ in range(run):
                if x + pw > x1 - 0.006 * w:
                    break
                d.rectangle([x, base - 0.012 * h - ph, x + pw - 2, base - 0.012 * h], fill=col)
                d.rectangle(
                    [x + pw * 0.18, base - 0.012 * h - ph * 0.62, x + pw * 0.82 - 2, base - 0.012 * h - ph * 0.38],
                    fill=tuple(min(255, int(c * 1.35)) for c in col),
                )
                x += pw
            x += rng.uniform(0.0, 0.006) * w
        d.rectangle([x0, base - 0.012 * h, x1, base], fill=(70, 80, 100))  # price rail
    for fx in (0.2, 0.5, 0.8):  # ceiling lights
        d.rectangle([(fx - 0.06) * w, 0.03 * h, (fx + 0.06) * w, 0.045 * h], fill=(120, 134, 160))
    return img


def _counter_bg(w: int, h: int, scene: Scene) -> Image.Image:
    lay = layout_for(scene)
    img = _vgrad((w, h), (24, 30, 44), (34, 42, 58))
    d = ImageDraw.Draw(img)
    for fx in (0.08, 0.30, 0.70, 0.92):  # wall panels behind the shopper
        d.rectangle([(fx - 0.07) * w, 0.08 * h, (fx + 0.07) * w, 0.50 * h], outline=(46, 56, 76), width=max(1, w // 640))
    top_y, front_y = 0.58 * h, 0.80 * h
    d.rectangle([0.10 * w, top_y, 0.90 * w, front_y], fill=(52, 60, 78))  # counter top
    d.rectangle([0.10 * w, front_y, 0.90 * w, h], fill=(30, 36, 50))  # counter front
    d.line([(0.10 * w, front_y), (0.90 * w, front_y)], fill=(86, 98, 122), width=max(2, w // 480))
    pad = 0.012
    for name, rect in lay.zones.items():
        box = [(rect.x0 + pad) * w, (rect.y0 + 0.025) * h, (rect.x1 - pad) * w, (rect.y1 - 0.02) * h]
        if name == "scanner":
            d.rounded_rectangle(box, radius=w // 120, fill=(28, 44, 62), outline=(90, 150, 190), width=max(2, w // 480))
            cx, cy = rect.center
            d.line([(cx - 0.05) * w, cy * h + 0.01 * h, (cx + 0.05) * w, cy * h + 0.01 * h], fill=(200, 70, 70), width=max(1, w // 640))
        elif name == "input" and scene == Scene.STAFFED_TILL:
            d.rectangle(box, fill=(22, 26, 34))
            for k in range(1, 9):  # conveyor ribs
                xx = box[0] + (box[2] - box[0]) * k / 9
                d.line([(xx, box[1]), (xx, box[3])], fill=(40, 46, 58), width=max(1, w // 640))
        else:
            d.rounded_rectangle(box, radius=w // 120, fill=(44, 52, 70), outline=(78, 90, 114), width=max(1, w // 640))
    if scene == Scene.SELF_CHECKOUT:  # payment screen on a pole at the far right
        d.rectangle([0.835 * w, 0.30 * h, 0.845 * w, 0.58 * h], fill=(60, 70, 90))
        d.rounded_rectangle([0.79 * w, 0.24 * h, 0.89 * w, 0.40 * h], radius=w // 160, fill=(30, 60, 84), outline=(110, 170, 210), width=max(1, w // 640))
    return img


def _exit_bg(w: int, h: int) -> Image.Image:
    img = _vgrad((w, h), (22, 28, 40), (30, 36, 50))
    _floor(img, 0.46, tint=(44, 52, 66))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0.10 * h, w, 0.46 * h], fill=(26, 32, 46))
    for k in range(5):  # distant checkout lanes on the back wall
        x = (0.10 + k * 0.13) * w
        d.rectangle([x, 0.36 * h, x + 0.085 * w, 0.46 * h], fill=(50, 58, 76))
        d.rectangle([x + 0.03 * w, 0.27 * h, x + 0.055 * w, 0.36 * h], fill=(40, 48, 64))
        d.ellipse([x + 0.034 * w, 0.215 * h, x + 0.051 * w, 0.245 * h], fill=(72, 120, 150) if k % 2 else (64, 74, 96))
    # Glass doors on the right hand side.
    d.rectangle([0.80 * w, 0.12 * h, w, 0.46 * h], fill=(34, 52, 70), outline=(98, 130, 160), width=max(2, w // 480))
    d.line([(0.90 * w, 0.12 * h), (0.90 * w, 0.46 * h)], fill=(98, 130, 160), width=max(2, w // 480))
    d.rounded_rectangle([0.855 * w, 0.045 * h, 0.945 * w, 0.10 * h], radius=w // 200, fill=(22, 101, 52))
    f = font("bold", max(10, int(0.030 * h)))
    d.text((0.90 * w, 0.0725 * h), "EXIT", font=f, fill=(220, 252, 231), anchor="mm")
    return img


@lru_cache(maxsize=16)
def background(scene: int, w: int, h: int) -> Image.Image:
    scene = Scene(scene)
    if scene == Scene.AISLE:
        img = _aisle_bg(w, h)
    elif scene == Scene.EXIT:
        img = _exit_bg(w, h)
    else:
        img = _counter_bg(w, h, scene)
    # Soft vignette for a camera feel.
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    shade = np.clip(1.08 - 0.22 * r**2, 0.6, 1.0)[..., None]
    return Image.fromarray((np.asarray(img).astype(np.float32) * shade).clip(0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _tag(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, colour, f, anchor: str = "ls", pad: int = 4) -> None:
    """Text on a dark rounded plate so it stays readable on any background."""
    box = d.textbbox(xy, text, font=f, anchor=anchor)
    d.rounded_rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], radius=pad, fill=(8, 12, 20))
    d.text(xy, text, font=f, fill=colour, anchor=anchor)


def _glow(img: Image.Image, box: list[float], colour: tuple[int, int, int], strength: int = 90) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(box, radius=12, fill=(*colour, strength))
    img.paste(layer.filter(ImageFilter.GaussianBlur(10)), (0, 0), layer.filter(ImageFilter.GaussianBlur(10)))


def render_frame(
    batch: ClipBatch,
    index: int,
    frame: int,
    risk: float | None = None,
    tier: str | None = None,
    label: str | None = None,
    note: str | None = None,
    style: RenderStyle | None = None,
) -> Image.Image:
    """Render one frame of one clip.

    Parameters
    ----------
    risk, tier, label : optional model output shown in the banner. ``tier`` is
        one of ``clear``, ``review`` or ``priority``.
    note : optional short caption drawn at the bottom of the frame.
    """
    style = style or RenderStyle()
    ss = style.supersample
    w, h = style.width * ss, style.height * ss
    frame = int(np.clip(frame, 0, batch.n_frames - 1))
    meta = batch.meta.iloc[index]
    scene = Scene(int(meta["scene"]))
    lay = layout_for(scene)
    kp = batch.keypoints[index, frame]
    sig = batch.signals[index, frame]
    recent = batch.signals[index, max(frame - 2, 0) : frame + 1]

    img = background(int(scene), w, h).copy()
    d = ImageDraw.Draw(img, "RGBA")
    f_small = font("sans", int(13 * ss * style.font_scale))
    f_bold = font("bold", int(14 * ss * style.font_scale))
    f_mono = font("mono", int(13 * ss * style.font_scale))
    lw = max(2, int(2.2 * ss))
    accent = TIER_COLOURS.get(tier or "clear", MINT) if risk is not None else MINT

    def px(x: float, y: float) -> tuple[float, float]:
        return (float(x) * w, float(y) * h)

    valid = kp[:, 2] >= style.min_conf
    pts = kp[:, :2]
    if valid[[KP["left_shoulder"], KP["right_shoulder"]]].all() and valid[[KP["left_hip"], KP["right_hip"]]].any():
        hips = pts[[KP["left_hip"], KP["right_hip"]]][valid[[KP["left_hip"], KP["right_hip"]]]].mean(axis=0)
        s_hat = float(np.linalg.norm(pts[[KP["left_shoulder"], KP["right_shoulder"]]].mean(axis=0) - hips) / 0.29)
    else:
        s_hat = float(meta["body_scale"])
    s_hat = float(np.clip(s_hat, 0.2, 1.0))

    # Zones -------------------------------------------------------------- #
    if scene in (Scene.SELF_CHECKOUT, Scene.STAFFED_TILL):
        names = {"input": "BASKET" if scene == Scene.SELF_CHECKOUT else "BELT", "scanner": "SCANNER", "bagging": "BAGGING" if scene == Scene.SELF_CHECKOUT else "PACKING"}
        scanned = recent[:, SIG["pos_qty"]].sum() > 0
        for key, rect in lay.zones.items():
            box = [rect.x0 * w, rect.y0 * h, rect.x1 * w, rect.y1 * h]
            col = GREEN if (key == "scanner" and scanned) else (92, 108, 136)
            if key == "scanner" and scanned:
                _glow(img, box, GREEN, 70)
                d = ImageDraw.Draw(img, "RGBA")
            d.rounded_rectangle(box, radius=int(6 * ss), outline=col, width=max(1, ss))
            count = {"input": sig[SIG["container_count"]], "bagging": sig[SIG["bagging_count"]]}.get(key)
            text = names[key] if count is None else f"{names[key]}  {int(count)}"
            d.text((rect.x0 * w + 6 * ss, rect.y1 * h - 6 * ss), text, font=f_small, fill=col if key == "scanner" and scanned else MUTED, anchor="ls")
        if scanned:
            row = recent[recent[:, SIG["pos_qty"]] > 0][-1]
            match = row[SIG["scan_match"]]
            ratio = row[SIG["price_ratio"]]
            suspect = match > 0 and match < 0.5 and ratio < 0.6
            rect = lay.zones["scanner"]
            msg = f"SCAN  match {match:.2f}  price x{ratio:.2f}" if match > 0 else "VOID"
            _tag(d, (rect.center[0] * w, rect.y0 * h - 10 * ss), msg, RED if suspect else GREEN, f_mono, anchor="ms")
    if scene == Scene.EXIT:
        alarm = sig[SIG["eas_alarm"]] > 0.5
        for y0, y1 in ((0.48, 0.66), (0.80, 0.995)):  # security tag pedestals, far and near
            x = 0.82
            box = [x * w - 5 * ss, y0 * h, x * w + 5 * ss, y1 * h]
            if alarm:
                _glow(img, [box[0] - 14 * ss, box[1], box[2] + 14 * ss, box[3]], RED, 120)
                d = ImageDraw.Draw(img, "RGBA")
            d.rounded_rectangle(box, radius=int(4 * ss), fill=RED if alarm else (86, 98, 122))
        if alarm:
            _tag(d, (0.90 * w, 0.44 * h), "TAG ALARM", RED, f_mono, anchor="ms")

    # Container and personal bag ---------------------------------------- #
    if sig[SIG["cont_conf"]] >= style.min_conf and scene in (Scene.AISLE, Scene.EXIT):
        cx, cy = sig[SIG["cont_x"]], sig[SIG["cont_y"]]
        hip_x = pts[[KP["left_hip"], KP["right_hip"]], 0].mean()
        trolley = abs(cx - hip_x) / s_hat > 0.21
        bw, bh = (0.15 * s_hat, 0.20 * s_hat) if trolley else (0.085 * s_hat, 0.055 * s_hat)
        box = [(cx - bw) * w, (cy - bh) * h, (cx + bw) * w, (cy + bh) * h]
        d.rounded_rectangle(box, radius=int(5 * ss), outline=SKY, width=max(1, ss), fill=(125, 211, 252, 26))
        if trolley:
            d.line([box[0], box[3], box[0] - 0.02 * w, box[3] + 0.05 * s_hat * h], fill=SKY, width=max(1, ss))
            d.line([box[2], box[3], box[2] + 0.02 * w, box[3] + 0.05 * s_hat * h], fill=SKY, width=max(1, ss))
        name = "TROLLEY" if trolley else "BASKET"
        _tag(d, (box[0], box[3] + 16 * ss), f"{name}  {int(sig[SIG['container_count']])}", SKY, f_small)
    if sig[SIG["bag_conf"]] >= style.min_conf:
        bx, by = sig[SIG["bag_x"]], sig[SIG["bag_y"]]
        bw, bh = 0.05 * s_hat, 0.085 * s_hat
        box = [(bx - bw) * w, (by - bh) * h, (bx + bw) * w, (by + bh) * h]
        d.rounded_rectangle(box, radius=int(6 * ss), outline=VIOLET, width=max(1, ss), fill=(167, 139, 250, 40))
        _tag(d, (box[0], box[1] - 6 * ss), "BAG", VIOLET, f_small)

    # Skeleton ----------------------------------------------------------- #
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            d.line([px(*pts[a]), px(*pts[b])], fill=(*accent, 235), width=lw)
    head_pts = [i for i in (KP["left_ear"], KP["right_ear"], KP["nose"]) if valid[i]]
    if head_pts:
        hx, hy = pts[head_pts].mean(axis=0)
        r = 0.052 * s_hat
        d.ellipse([(hx - r * 0.62) * w, (hy - r * 1.25) * h, (hx + r * 0.62) * w, (hy + r * 1.05) * h], outline=(*accent, 235), width=lw)
        if valid[KP["nose"]] and kp[KP["nose"], 2] > 0.5:
            nx, ny = px(*pts[KP["nose"]])
            d.ellipse([nx - 2 * ss, ny - 2 * ss, nx + 2 * ss, ny + 2 * ss], fill=TEXT)
    for i in range(5, 17):
        if valid[i]:
            x, y = px(*pts[i])
            r = 3.2 * ss
            d.ellipse([x - r, y - r, x + r, y + r], fill=TEXT, outline=INK)

    # Items in hand ------------------------------------------------------- #
    for chan, wrist in (("hand_item_l", "left_wrist"), ("hand_item_r", "right_wrist")):
        if sig[SIG[chan]] > 0.5 and valid[KP[wrist]]:
            x, y = px(*pts[KP[wrist]])
            r = 8 * ss
            d.rounded_rectangle([x - r, y - r * 1.25, x + r, y + r * 0.75], radius=int(3 * ss), fill=AMBER, outline=INK, width=max(1, ss))
    if scene == Scene.AISLE:
        delta = recent[:, SIG["shelf_delta"]].sum()
        if delta != 0:
            wr = [pts[KP[k]] for k in ("left_wrist", "right_wrist") if valid[KP[k]]]
            if wr:
                top = min(wr, key=lambda p: p[1])
                _tag(d, (top[0] * w + 14 * ss, top[1] * h - 12 * ss), "SHELF  item taken" if delta > 0 else "SHELF  item returned", AMBER if delta > 0 else GREEN, f_small)

    # Person box ---------------------------------------------------------- #
    if valid.sum() >= 4:
        vx, vy = pts[valid, 0], pts[valid, 1]
        pad = 0.035 * s_hat
        box = [(vx.min() - pad) * w, (vy.min() - pad * 2.4) * h, (vx.max() + pad) * w, min((vy.max() + pad) * h, h - 2 * ss)]
        d.rounded_rectangle(box, radius=int(8 * ss), outline=(*accent, 200), width=max(1, ss))
        tid = int(str(meta["clip_id"])[-4:]) if str(meta["clip_id"])[-4:].isdigit() else index
        _tag(d, (box[0] + 4 * ss, box[1] - 8 * ss), f"TRACK {tid:04d}", accent, f_small)

    if scene == Scene.STAFFED_TILL:  # customer silhouette in the foreground
        d.ellipse([0.84 * w, 0.78 * h, 0.92 * w, 0.93 * h], fill=(12, 16, 24, 255))
        d.rounded_rectangle([0.77 * w, 0.90 * h, 0.99 * w, 1.10 * h], radius=int(40 * ss), fill=(12, 16, 24, 255))

    # Heads up display: camera label and clock along the bottom, verdict at the top.
    if style.show_hud:
        base = h - 20 * ss
        cam = f"CAM {int(meta['store_id']) % 9 + 1:02d}  {lay.camera_label}"
        _tag(d, (16 * ss, base), cam, TEXT, f_mono)
        t_s = frame / batch.fps
        clock = f"{int(meta['hour']):02d}:{14 + int(t_s) // 60:02d}:{int(t_s) % 60:02d}.{int((t_s % 1) * 10)}"
        box = d.textbbox((w - 16 * ss, base), clock, font=f_mono, anchor="rs")
        _tag(d, (w - 16 * ss, base), clock, TEXT, f_mono, anchor="rs")
        d.ellipse([box[0] - 26 * ss, base - 12 * ss, box[0] - 14 * ss, base], fill=RED)
        if risk is not None:
            name = (tier or "clear").upper()
            text = f"{name}   {label}   {risk:.2f}" if label else f"{name}   risk {risk:.2f}"
            y_banner = 30 * ss
            box = d.textbbox((w / 2, y_banner), text, font=f_bold, anchor="ms")
            d.rounded_rectangle([box[0] - 12 * ss, box[1] - 7 * ss, box[2] + 12 * ss, box[3] + 7 * ss], radius=int(12 * ss), fill=(8, 12, 20), outline=accent, width=max(1, ss))
            d.text((w / 2, y_banner), text, font=f_bold, fill=accent, anchor="ms")
        if note:
            _tag(d, (16 * ss, base - 26 * ss), note, TEXT, f_small)

    return img.resize((style.width, style.height), Image.LANCZOS)


def contact_sheet(images: list[Image.Image], cols: int, gap: int = 8, bg=INK) -> Image.Image:
    """Tile equally sized images into a grid."""
    if not images:
        raise ValueError("no images to tile")
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), bg)
    for i, im in enumerate(images):
        sheet.paste(im, (gap + (i % cols) * (w + gap), gap + (i // cols) * (h + gap)))
    return sheet
