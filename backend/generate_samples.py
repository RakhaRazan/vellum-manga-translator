"""Synthetic comic page generator for Japanese (ja), Korean (ko), and Chinese (zh).

Generates three 1000x1400 px comic pages saved to `samples/`:
- `samples/sample_ja.jpg`: Japanese Manga layout (Right-to-Left, Top-to-Bottom)
- `samples/sample_ko.jpg`: Korean Manhwa layout (Left-to-Right, Top-to-Bottom)
- `samples/sample_zh.jpg`: Chinese Manhua layout (Left-to-Right, Top-to-Bottom)

Each page includes comic panels, background screentones/action lines, character silhouettes,
white speech bubbles with tails, and multi-line CJK dialogue rendered with standard
Windows CJK system fonts (with graceful fallback).
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFont

PAGE_WIDTH = 1000
PAGE_HEIGHT = 1400


class SampleGenerationResult(TypedDict):
    """Metadata returned after generating a synthetic comic page."""

    lang: str
    path: str
    size: tuple[int, int]
    panels: int
    bubbles: int
    font: str


@dataclass(frozen=True, slots=True)
class BubbleSpec:
    """Specification for a speech bubble and its dialogue on a synthetic comic page."""

    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    tail_target: tuple[int, int]  # (tx, ty) character mouth direction
    text: str
    shape: str = "ellipse"  # "ellipse" or "rounded_rect"


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """Specification for a comic panel containing artwork and speech bubbles."""

    rect: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    bg_color: tuple[int, int, int]
    accent_color: tuple[int, int, int]
    character_pos: tuple[int, int]  # (cx, cy) head center
    character_facing: str  # "left" or "right"
    style: str  # "dots", "speedlines", "radial"
    bubbles: list[BubbleSpec]


# Standard Windows CJK font candidates ordered by visual clarity for comic dialogue
FONT_CANDIDATES: dict[str, list[str]] = {
    "ja": [
        "YuGothB.ttc",
        "YuGothM.ttc",
        "meiryob.ttc",
        "meiryo.ttc",
        "msgothic.ttc",
        "YuGothR.ttc",
        "msmincho.ttc",
        "NotoSansCJK-Bold.ttc",
        "NotoSansJP-Bold.otf",
    ],
    "ko": [
        "malgunbd.ttf",
        "malgun.ttf",
        "gulim.ttc",
        "batang.ttc",
        "NotoSansCJK-Bold.ttc",
        "NotoSansKR-Bold.otf",
    ],
    "zh": [
        "msyhbd.ttc",
        "msyh.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "kaiti.ttf",
        "NotoSansCJK-Bold.ttc",
        "NotoSansSC-Bold.otf",
    ],
}


def get_font_search_dirs() -> list[Path]:
    """Return candidate system font directories across Windows, macOS, and Linux."""
    dirs: list[Path] = []
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    dirs.append(Path(windir) / "Fonts")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        dirs.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
    dirs.extend(
        [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
        ]
    )
    return [d for d in dirs if d.is_dir()]


def load_cjk_font(
    lang: str,
    size: int = 30,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]:
    """Load a CJK system font for `lang` with graceful fallback."""
    candidates = FONT_CANDIDATES.get(lang, []) + [
        "msyh.ttc",
        "msgothic.ttc",
        "malgun.ttf",
        "arialuni.ttf",
        "seguiemj.ttf",
        "arial.ttf",
    ]
    font_dirs = get_font_search_dirs()

    for font_name in candidates:
        for font_dir in font_dirs:
            font_path = font_dir / font_name
            if font_path.is_file():
                try:
                    return ImageFont.truetype(str(font_path), size=size), str(font_path)
                except OSError:
                    continue
        try:
            return ImageFont.truetype(font_name, size=size), font_name
        except OSError:
            continue

    try:
        return ImageFont.load_default(size=size), "Pillow-default"
    except TypeError:
        return ImageFont.load_default(), "Pillow-default"


def _draw_panel_background(
    img: Image.Image,
    panel: PanelSpec,
) -> None:
    """Draw panel background fill, screentone pattern/speedlines, and character artwork."""
    x1, y1, x2, y2 = panel.rect
    w = x2 - x1
    h = y2 - y1

    panel_img = Image.new("RGB", (w, h), panel.bg_color)
    pdraw = ImageDraw.Draw(panel_img)

    # Screentone / comic background effects
    if panel.style == "dots":
        step = 18
        for py in range(10, h, step):
            offset = (step // 2) if ((py // step) % 2 == 1) else 0
            for px in range(10 + offset, w, step):
                r = 2 if py < h // 2 else 3
                pdraw.ellipse(
                    (px - r, py - r, px + r, py + r),
                    fill=panel.accent_color,
                )
    elif panel.style == "speedlines":
        for py in range(12, h, 16):
            line_len = int(w * (0.35 + 0.25 * ((py * 7) % 5) / 5.0))
            if (py // 16) % 2 == 0:
                pdraw.line((0, py, line_len, py), fill=panel.accent_color, width=2)
            else:
                pdraw.line((w - line_len, py, w, py), fill=panel.accent_color, width=2)
    elif panel.style == "radial":
        cx_rad = w // 2
        cy_rad = h // 2
        max_r = int(math.hypot(w, h))
        for angle_deg in range(0, 360, 8):
            rad = math.radians(angle_deg)
            inner_r = int(min(w, h) * 0.28)
            sx = cx_rad + int(math.cos(rad) * inner_r)
            sy = cy_rad + int(math.sin(rad) * inner_r)
            ex = cx_rad + int(math.cos(rad) * max_r)
            ey = cy_rad + int(math.sin(rad) * max_r)
            pdraw.line((sx, sy, ex, ey), fill=panel.accent_color, width=2)

    # Simple stylized comic character bust so the page has realistic visual context
    cx, cy = panel.character_pos[0] - x1, panel.character_pos[1] - y1
    head_r = 56
    shoulder_w = 135
    shoulder_h = 175

    # Torso / shoulders
    pdraw.chord(
        (cx - shoulder_w, cy + head_r - 12, cx + shoulder_w, cy + head_r + shoulder_h * 2),
        start=180,
        end=360,
        fill=(55, 62, 74),
        outline=(20, 20, 25),
        width=4,
    )
    # Collar / shirt V-neck detail
    pdraw.polygon(
        [
            (cx - 26, cy + head_r - 5),
            (cx + 26, cy + head_r - 5),
            (cx, cy + head_r + 48),
        ],
        fill=(235, 235, 240),
        outline=(20, 20, 25),
    )
    # Neck
    pdraw.rectangle(
        (cx - 18, cy + head_r - 22, cx + 18, cy + head_r + 8),
        fill=(242, 228, 216),
        outline=(20, 20, 25),
        width=3,
    )
    # Head
    pdraw.ellipse(
        (cx - head_r, cy - head_r, cx + head_r, cy + head_r),
        fill=(248, 236, 224),
        outline=(20, 20, 25),
        width=4,
    )
    # Hair cap
    pdraw.chord(
        (cx - head_r - 4, cy - head_r - 8, cx + head_r + 4, cy + head_r - 10),
        start=175,
        end=5,
        fill=(32, 34, 42),
        outline=(15, 15, 20),
        width=3,
    )
    # Eyes and mouth oriented toward speech bubble
    eye_offset = -14 if panel.character_facing == "left" else 14
    for ex_delta in (-18, 18):
        ex = cx + ex_delta + (eye_offset // 2)
        ey = cy - 4
        pdraw.ellipse((ex - 6, ey - 8, ex + 6, ey + 8), fill=(25, 25, 30))
        pdraw.ellipse((ex - 3, ey - 5, ex + 1, ey - 1), fill=(255, 255, 255))
        pdraw.line((ex - 10, ey - 14, ex + 10, ey - 12), fill=(25, 25, 30), width=3)

    mouth_x = cx + (eye_offset // 2)
    mouth_y = cy + 24
    pdraw.arc(
        (mouth_x - 12, mouth_y - 6, mouth_x + 12, mouth_y + 8),
        start=10,
        end=170,
        fill=(30, 30, 35),
        width=3,
    )

    img.paste(panel_img, (x1, y1))


def _draw_speech_bubble(
    draw: ImageDraw.ImageDraw,
    bubble: BubbleSpec,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw a white speech bubble with black border, pointer tail, and centered CJK text."""
    bx1, by1, bx2, by2 = bubble.box
    bcx = (bx1 + bx2) // 2
    bcy = (by1 + by2) // 2
    tx, ty = bubble.tail_target

    # Compute tail base points anchored inside the bubble center
    dx = tx - bcx
    dy = ty - bcy
    dist = max(1.0, math.hypot(dx, dy))
    nx = -dy / dist
    ny = dx / dist
    base_half = 16

    tail_poly = [
        (int(bcx + nx * base_half), int(bcy + ny * base_half)),
        (int(bcx - nx * base_half), int(bcy - ny * base_half)),
        (tx, ty),
    ]

    # Draw tail border first, then bubble outline, then white fill so tail merges cleanly
    draw.polygon(tail_poly, fill=(255, 255, 255), outline=(15, 15, 15))
    draw.line([tail_poly[0], tail_poly[2]], fill=(15, 15, 15), width=3)
    draw.line([tail_poly[1], tail_poly[2]], fill=(15, 15, 15), width=3)

    if bubble.shape == "rounded_rect":
        draw.rounded_rectangle(
            bubble.box,
            radius=28,
            fill=(255, 255, 255),
            outline=(15, 15, 15),
            width=4,
        )
    else:
        draw.ellipse(
            bubble.box,
            fill=(255, 255, 255),
            outline=(15, 15, 15),
            width=4,
        )

    # Re-fill interior of tail where it joins the bubble perimeter to erase inner border seam
    inner_box = (bx1 + 4, by1 + 4, bx2 - 4, by2 - 4)
    if bubble.shape == "rounded_rect":
        draw.rounded_rectangle(inner_box, radius=24, fill=(255, 255, 255))
    else:
        draw.ellipse(inner_box, fill=(255, 255, 255))

    # Measure multi-line text and draw centered inside the speech bubble
    spacing = 10
    text_bbox = draw.multiline_textbbox(
        (0, 0),
        bubble.text,
        font=font,
        spacing=spacing,
        align="center",
    )
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    text_x = bcx - tw // 2 - text_bbox[0]
    text_y = bcy - th // 2 - text_bbox[1]

    draw.multiline_text(
        (text_x, text_y),
        bubble.text,
        fill=(12, 12, 12),
        font=font,
        spacing=spacing,
        align="center",
    )


def build_page_specs() -> dict[str, tuple[str, list[PanelSpec]]]:
    """Define the 3 synthetic comic pages for Japanese (ja), Korean (ko), and Chinese (zh)."""
    # 1. Japanese Manga (Right-to-Left, Top-to-Bottom reading order)
    ja_panels = [
        # Top-Right panel (Bubble 1)
        PanelSpec(
            rect=(515, 50, 955, 470),
            bg_color=(214, 220, 230),
            accent_color=(182, 190, 204),
            character_pos=(825, 320),
            character_facing="left",
            style="dots",
            bubbles=[
                BubbleSpec(
                    box=(535, 75, 855, 235),
                    tail_target=(810, 275),
                    text="何をしているんだ？\nここは危険だぞ！",
                    shape="ellipse",
                ),
            ],
        ),
        # Top-Left panel (Bubble 2)
        PanelSpec(
            rect=(45, 50, 490, 470),
            bg_color=(226, 222, 214),
            accent_color=(196, 190, 178),
            character_pos=(170, 325),
            character_facing="right",
            style="speedlines",
            bubbles=[
                BubbleSpec(
                    box=(135, 80, 470, 240),
                    tail_target=(195, 275),
                    text="まさか…\nここで会うとはな！",
                    shape="ellipse",
                ),
            ],
        ),
        # Middle full-width action panel (Bubble 3)
        PanelSpec(
            rect=(45, 495, 955, 920),
            bg_color=(204, 212, 224),
            accent_color=(170, 180, 196),
            character_pos=(280, 745),
            character_facing="right",
            style="radial",
            bubbles=[
                BubbleSpec(
                    box=(440, 540, 905, 735),
                    tail_target=(355, 725),
                    text="気をつけろ！\n敵がすぐ近くまで来ている！",
                    shape="rounded_rect",
                ),
            ],
        ),
        # Bottom full-width panel (Bubble 4)
        PanelSpec(
            rect=(45, 945, 955, 1350),
            bg_color=(222, 228, 220),
            accent_color=(188, 198, 186),
            character_pos=(730, 1185),
            character_facing="left",
            style="dots",
            bubbles=[
                BubbleSpec(
                    box=(110, 995, 560, 1185),
                    tail_target=(655, 1165),
                    text="よし、準備は完了だ。\n一気に突破するぞ！",
                    shape="ellipse",
                ),
            ],
        ),
    ]

    # 2. Korean Manhwa (Left-to-Right, Top-to-Bottom reading order)
    ko_panels = [
        # Top-Left panel (Bubble 1)
        PanelSpec(
            rect=(45, 50, 490, 470),
            bg_color=(216, 226, 236),
            accent_color=(184, 198, 212),
            character_pos=(165, 325),
            character_facing="right",
            style="dots",
            bubbles=[
                BubbleSpec(
                    box=(125, 75, 468, 235),
                    tail_target=(190, 275),
                    text="여기서 뭐 하고 있어?\n여긴 너무 위험해!",
                    shape="ellipse",
                ),
            ],
        ),
        # Top-Right panel (Bubble 2)
        PanelSpec(
            rect=(515, 50, 955, 470),
            bg_color=(232, 224, 218),
            accent_color=(204, 192, 182),
            character_pos=(825, 325),
            character_facing="left",
            style="speedlines",
            bubbles=[
                BubbleSpec(
                    box=(535, 80, 865, 240),
                    tail_target=(805, 275),
                    text="설마 진짜로\n혼자 찾아온 거야?",
                    shape="ellipse",
                ),
            ],
        ),
        # Middle full-width panel (Bubble 3)
        PanelSpec(
            rect=(45, 495, 955, 920),
            bg_color=(212, 218, 230),
            accent_color=(178, 186, 204),
            character_pos=(720, 745),
            character_facing="left",
            style="radial",
            bubbles=[
                BubbleSpec(
                    box=(100, 540, 565, 735),
                    tail_target=(645, 725),
                    text="조심해!\n적이 바로 앞까지 다가왔어!",
                    shape="rounded_rect",
                ),
            ],
        ),
        # Bottom full-width panel (Bubble 4)
        PanelSpec(
            rect=(45, 945, 955, 1350),
            bg_color=(220, 230, 224),
            accent_color=(186, 200, 192),
            character_pos=(260, 1185),
            character_facing="right",
            style="dots",
            bubbles=[
                BubbleSpec(
                    box=(430, 995, 885, 1185),
                    tail_target=(335, 1165),
                    text="좋아, 준비는 끝났어.\n지금 바로 출발하자!",
                    shape="ellipse",
                ),
            ],
        ),
    ]

    # 3. Chinese Manhua (Left-to-Right, Top-to-Bottom reading order)
    zh_panels = [
        # Top-Left panel (Bubble 1)
        PanelSpec(
            rect=(45, 50, 490, 470),
            bg_color=(228, 222, 214),
            accent_color=(198, 188, 176),
            character_pos=(165, 325),
            character_facing="right",
            style="dots",
            bubbles=[
                BubbleSpec(
                    box=(125, 75, 468, 235),
                    tail_target=(190, 275),
                    text="你在这里做什么？\n这里非常危险！",
                    shape="ellipse",
                ),
            ],
        ),
        # Top-Right panel (Bubble 2)
        PanelSpec(
            rect=(515, 50, 955, 470),
            bg_color=(216, 226, 234),
            accent_color=(184, 198, 210),
            character_pos=(825, 325),
            character_facing="left",
            style="speedlines",
            bubbles=[
                BubbleSpec(
                    box=(535, 80, 865, 240),
                    tail_target=(805, 275),
                    text="难道你已经\n发现这里的秘密了？",
                    shape="ellipse",
                ),
            ],
        ),
        # Middle full-width panel (Bubble 3)
        PanelSpec(
            rect=(45, 495, 955, 920),
            bg_color=(214, 218, 228),
            accent_color=(178, 184, 200),
            character_pos=(720, 745),
            character_facing="left",
            style="radial",
            bubbles=[
                BubbleSpec(
                    box=(100, 540, 565, 735),
                    tail_target=(645, 725),
                    text="大家小心！\n敌人正在迅速靠近！",
                    shape="rounded_rect",
                ),
            ],
        ),
        # Bottom full-width panel (Bubble 4)
        PanelSpec(
            rect=(45, 945, 955, 1350),
            bg_color=(222, 230, 220),
            accent_color=(188, 200, 186),
            character_pos=(260, 1185),
            character_facing="right",
            style="dots",
            bubbles=[
                BubbleSpec(
                    box=(430, 995, 885, 1185),
                    tail_target=(335, 1165),
                    text="太好了，准备完毕。\n我们立刻出发吧！",
                    shape="ellipse",
                ),
            ],
        ),
    ]

    return {
        "ja": ("sample_ja.jpg", ja_panels),
        "ko": ("sample_ko.jpg", ko_panels),
        "zh": ("sample_zh.jpg", zh_panels),
    }


def generate_comic_page(
    lang: str,
    panels: list[PanelSpec],
    output_path: Path,
    font_size: int = 30,
) -> SampleGenerationResult:
    """Render a 1000x1400 synthetic comic page and save as JPEG."""
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (246, 245, 240))
    draw = ImageDraw.Draw(page)
    font, font_used = load_cjk_font(lang, size=font_size)

    bubble_count = 0
    for panel in panels:
        _draw_panel_background(page, panel)
        # Draw crisp comic panel border
        draw.rectangle(panel.rect, outline=(18, 18, 20), width=5)
        for bubble in panel.bubbles:
            _draw_speech_bubble(draw, bubble, font)
            bubble_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path, format="JPEG", quality=95, subsampling=0)
    return {
        "lang": lang,
        "path": str(output_path),
        "size": page.size,
        "panels": len(panels),
        "bubbles": bubble_count,
        "font": font_used,
    }


def generate_all_samples(
    samples_dir: Path | str | None = None,
) -> list[SampleGenerationResult]:
    """Generate `sample_ja.jpg`, `sample_ko.jpg`, and `sample_zh.jpg` inside `samples/`."""
    if samples_dir is None:
        workspace_root = Path(__file__).resolve().parents[1]
        target_dir = workspace_root / "samples"
    else:
        target_dir = Path(samples_dir).expanduser().resolve()

    results: list[SampleGenerationResult] = []
    for lang, (filename, panels) in build_page_specs().items():
        out_path = target_dir / filename
        info = generate_comic_page(lang=lang, panels=panels, output_path=out_path)
        results.append(info)
    return results


def main() -> int:
    """CLI entry point to generate all 3 synthetic comic pages."""
    print("=== Generating Synthetic Comic Pages (1000x1400 px) ===")
    results = generate_all_samples()
    for res in results:
        print(
            f"  [OK] {res['lang'].upper()} -> {res['path']} "
            + f"(size={res['size']}, panels={res['panels']}, "
            + f"bubbles={res['bubbles']}, font={res['font']})"
        )
    print("SUCCESS: Generated sample_ja.jpg, sample_ko.jpg, and sample_zh.jpg.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
