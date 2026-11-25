# AI Turntable Video Prompts for Video → VRM Studio

This studio uses a **Zero-ML 3D visual-hull + anatomical polar-loft pipeline**.
Because there is no neural segmentation or generative back-fill, the cleanest VRM results come from turnaround videos where the character stays rigid, centered, and shadow-free against pure chroma-key green (`#00FF00`).

---

## Prompt A — Horizontal Standard (Recommended)

Replace `[CHARACTER]` with any outfit/hair description. Avoid green colors on skin, hair, eyes, or clothing so chroma-keying remains crisp.

```text
A clean 3D anime character game model turnaround video for 3D asset scanning.
[CHARACTER: an anime-style character with short-to-medium dark navy hair (no long hair covering shoulders or collar), wearing a fitted white short-sleeve sailor blouse with a small red ribbon, a crisp dark navy pleated skirt with a clean horizontal hemline, white socks, and dark brown loafers — strictly NO green colors anywhere on the character or eyes].

POSE & MOTION (STRICT):
The character is completely frozen like a rigid plastic 3D figurine in a strict symmetrical T-pose: arms extended perfectly straight horizontally at shoulder height, palms facing down with five straight fingers slightly separated, legs straight and slightly apart at shoulder width, with the background visible between the legs, standing upright facing straight forward at frame 1.
The rigid frozen figure rotates smoothly 360 degrees at a perfectly constant speed around its vertical center axis over the entire video, returning to the exact front-facing view at the end.
ZERO animation: no breathing, no blinking, no head turning, no arm movement, no hair swaying, no skirt fluttering.

CAMERA & LIGHTING (STRICT):
Locked static camera with a 135mm telephoto orthographic-like lens centered at waist height, zero zoom, zero pan, zero tilt, zero camera shake. Full body from top of head to bottom of shoes stays centered with comfortable green margin on all four sides (never cropped).
Completely flat, even, shadowless diffuse studio lighting (clean cel-shaded albedo look, no harsh shadows under chin, arms, or skirt, no rim lighting, no glossy reflections).
Background is a 100% solid uniform pure chroma-key green (#00FF00) void — NO floor plane, NO turntable stand, NO cast shadows, NO gradients, NO text, NO UI.
```

---

## Prompt B — Slight +10° High-Angle (Better Shoulder & Crown Coverage)

Replace the `CAMERA & LIGHTING` paragraph in Prompt A with:

```text
Locked static camera with a 135mm telephoto orthographic-like lens positioned at a slight 10-degree downward angle (gentle high-angle view showing the top of shoulders and head cleanly), zero zoom, zero pan, zero camera shake. Full body from top of head to bottom of shoes stays centered with comfortable green margin on all four sides (never cropped).
```

---

## 4-Point Quality Checklist (生成後の確認ポイント)

1. **Rigid turnaround (静止回転)**: The character does not breathe, blink, or sway arms/hair mid-spin.
2. **Clean green void (足元の影・床線なし)**: No dark shadow disk or studio floor line beneath the shoes.
3. **Full-body framing (見切れなし)**: Head crown to shoe soles remain inside the frame for all 360°.
4. **Visible leg/neck gaps (首・脚の分離)**: Background green is visible between the legs and around the neck, so the hemline and legs can be separated automatically.
