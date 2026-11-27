# Vellum Detector Engine v2 - 9-Class Comic Annotation Standard (CVAT / Label Studio)

This document defines the canonical annotation standard for curating instance segmentation datasets for **Vellum Detector Engine v2**. Every class ID and name maps 1:1 to `vellum::detector::ClassId` in `backend-cpp/src/detector/detector_types.hpp`.

---

## 1. Exact 9-Class Taxonomy & Hex Colors

| Class ID | Snake Name | C++ `ClassId` | Hex Color | Description |
|---:|---|---|---|
| `0` | `speech_bubble` | `SpeechBubble` | `#22C55E` | Standard rounded/oval/rectangular spoken dialogue balloons with solid or semi-solid outlines. |
| `1` | `thought_bubble` | `ThoughtBubble` | `#38BDF8` | Cloud-shaped or scalloped internal monologue bubbles. |
| `2` | `narration_box` | `NarrationBox` | `#F59E0B` | Rectangular/chamfered narrator boxes, captions, scene/time cards. |
| `3` | `shout_bubble` | `ShoutBubble` | `#EF4444` | Jagged, spiky, or burst-shaped high-energy dialogue balloons. |
| `4` | `whisper_bubble` | `WhisperBubble` | `#A855F7` | Dashed-outline, dotted-outline, or faded/small-font whispered dialogue bubbles. |
| `5` | `borderless_dialogue` | `BorderlessDialogue` | `#EC4899` | Spoken/internal dialogue typeset directly over artwork without a enclosing border. |
| `6` | `text_region` | `TextRegion` | `#EAB308` | Tight text-only region inside a bubble or standalone typeset UI/sign text (when not covered by `0..5`). |
| `7` | `panel` | `Panel` | `#64748B` | Individual comic/manga/manhwa panel frames used for reading-order association. |
| `8` | `sfx_ignore` | `SfxIgnore` | `#F97316` | Hand-drawn onomatopoeia / sound effects that must be ignored or suppressed from dialogue translation. |

---

## 2. Polygon Boundary Rules

1. **Speech Tails (`speech_bubble`, `whisper_bubble`):**
   - **Cut off speech tails at the base curve** where the tail meets the main balloon body. Do **not** trace the long pointed tail tip, as tails distort `safe_bbox` and panel association.
2. **Thought Connector Dots (`thought_bubble`):**
   - Trace only the main cloud/scalloped bubble body. **Ignore trailing connector circles/dots** leading to the speaker.
3. **Shout Spikes (`shout_bubble`):**
   - **Trace the inner hull of spikes** (connecting the inner valleys/notches around the text-bearing interior) rather than tracing every outer spike tip. This keeps `safe_bbox` centered on the printable interior.
4. **Borderless Dialogue (`borderless_dialogue`):**
   - Draw a tight polygon around the text block with a **4–8 px margin** so OCR receives clean context without pulling in unrelated background art.
5. **Narration Boxes (`narration_box`) & Panels (`panel`):**
   - Trace the outer frame border cleanly (4 vertices for axis-aligned boxes, or exact vertices for slanted/trapezoidal panels).

---

## 3. Edge-Case Decision Matrix

| Edge Case | Rule |
|---|---|
| **Connected / Double Bubbles** | **Split at the pinch neck** into two separate `speech_bubble` instances so each speaker turn receives its own reading order and `safe_bbox`. |
| **Truncated Bubbles at Image/Panel Borders** | **Clamp strictly to the image or panel boundary** (`[0, W] x [0, H]`). Never extrapolate coordinates outside the canvas. |
| **Occluded Bubbles (split by character/object)** | Assign a **multi-polygon group / shared instance ID** in CVAT/Label Studio when two visible halves belong to the same continuous bubble. |
| **Borderless Dialogue vs. SFX (`5` vs. `8`)** | Apply the **2-step typography vs. hand-drawn brush rule**: (1) Is it typeset in a uniform comic/dialogue font? If yes -> `borderless_dialogue` (`5`). (2) Is it hand-lettered, warped, brush-stroked, or stylized sound/impact art? If yes -> `sfx_ignore` (`8`). |
| **Phone Screens / Monitors** | **Include main readable chat/message bubbles** on phone screens as `speech_bubble` (`0`) or `narration_box` (`2`); **ignore tiny background props**, decorative UI icons, and unreadable micro-text. |
| **Empty Bubbles** | **Skip** completely empty bubbles containing no text or punctuation. |
| **Single-Punctuation Bubbles (`!`, `?`, `...`, `!?`)** | **Keep** and label with the appropriate bubble class (`0..4`); translators/typesetters still localize or preserve punctuation bubbles. |

---

## 4. Target Dataset Composition & Minority Class Quotas

### Target Page Composition (2,500–4,000 Pages Total)
- **50% B&W Manga** (Japanese right-to-left, vertical text, screentones)
- **25% Color Manhwa / Manhua** (Korean/Chinese left-to-right, tall Webtoon strips, gradients)
- **15% Hard Examples** (overlapping double bubbles, borderless text over busy art, heavy SFX, slanted panels)
- **10% Background-Only / Negative Pages** (clean illustrations, title pages, splash art with zero dialogue)

### Minimum Minority Class Instance Quotas
Before training, `dataset_builder.py qa-check` verifies the following minimum instance counts:
- `whisper_bubble` (`4`): **>= 600** instances
- `borderless_dialogue` (`5`): **>= 1,000** instances
- `shout_bubble` (`3`): **>= 800** instances
- `thought_bubble` (`1`): **>= 800** instances
- `narration_box` (`2`): **>= 1,200** instances
- `sfx_ignore` (`8`): **>= 3,000** instances

---

## 5. Automated QA Checklist

Run `python tools/detector_trainer/dataset_builder.py qa-check --coco <annotations.json>` before splitting:
1. **Polygon Validity:** Every polygon has `>= 3` vertices (`>= 6` flat coordinates), is non-degenerate, and has shoelace area `>= 144 px^2` (`12x12` minimum).
2. **Duplicate Suppression Check:** Flag any same-class pair on the same image with bounding-box IoU `> 0.85` (likely accidental double-click annotations).
3. **Class Quota Audit:** Report per-class instance counts and highlight any minority class below its target quota.
4. **Title-Level Split Integrity:** Always use `split-dataset` (75% train / 15% val / 10% test grouped by series folder) so zero pages from the same mangaka/series leak across splits.
