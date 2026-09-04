"""Full-scene and shared-context rendering for pairwise verification."""
from PIL import Image, ImageDraw, ImageFont

PALETTE = ("#ff3b30", "#00a651")

def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _xywh_to_xyxy(box):
    x, y, w, h = [float(value) for value in box]
    return x, y, x + w, y + h


def _letterbox(image, size, color=(238, 238, 238)):
    target_w, target_h = size
    scale = min(target_w / max(image.width, 1), target_h / max(image.height, 1))
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.resize((width, height), resampling)
    canvas = Image.new("RGB", size, color)
    offset = ((target_w - width) // 2, (target_h - height) // 2)
    canvas.paste(resized, offset)
    return canvas, scale, offset


def _draw_tag(draw, x, y, label, color, font, image_size):
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    width = right - left + 12
    height = bottom - top + 8
    x = min(max(int(x), 0), max(image_size[0] - width, 0))
    y = min(max(int(y), 0), max(image_size[1] - height, 0))
    draw.rectangle((x, y, x + width, y + height), fill=color, outline="white", width=2)
    draw.text((x + 6, y + 3), label, fill="white", font=font)


def _context_crop(image, box, context_fraction):
    x1, y1, x2, y2 = _xywh_to_xyxy(box)
    width, height = x2 - x1, y2 - y1
    pad_x = max(width * context_fraction, 8.0)
    pad_y = max(height * context_fraction, 8.0)
    crop_box = (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image.width), x2 + pad_x),
        min(float(image.height), y2 + pad_y),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        # Keep an off-image prediction off-image; do not invent a visible box.
        crop_box = (0., 0., float(image.width), float(image.height))
    crop = image.crop(tuple(int(round(value)) for value in crop_box))
    local_box = (
        x1 - crop_box[0],
        y1 - crop_box[1],
        x2 - crop_box[0],
        y2 - crop_box[1],
    )
    return crop, local_box


def _joint_context_crop(image, boxes, context_fraction):
    corners = [_xywh_to_xyxy(box) for box in boxes]
    x1 = min(box[0] for box in corners)
    y1 = min(box[1] for box in corners)
    x2 = max(box[2] for box in corners)
    y2 = max(box[3] for box in corners)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = max(width * context_fraction, 16.0)
    pad_y = max(height * context_fraction, 16.0)
    crop_box = (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image.width), x2 + pad_x),
        min(float(image.height), y2 + pad_y),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        crop_box = (0., 0., float(image.width), float(image.height))
    crop = image.crop(tuple(int(round(value)) for value in crop_box))
    local_boxes = [
        (
            box[0] - crop_box[0],
            box[1] - crop_box[1],
            box[2] - crop_box[0],
            box[3] - crop_box[1],
        )
        for box in corners
    ]
    return crop, local_boxes


def _draw_candidate_box(draw, box, scale, offset, color, label, font, image_size):
    transformed = (
        offset[0] + box[0] * scale,
        offset[1] + box[1] * scale,
        offset[0] + box[2] * scale,
        offset[1] + box[3] * scale,
    )
    draw.rectangle(transformed, outline=color, width=5)
    _draw_tag(draw, transformed[0], transformed[1], label, color, font, image_size)


def render_pairwise_relation_montage(
    image,
    candidates,
    overview_size=(960, 500),
    panel_size=(320, 250),
    context_fraction=0.55,
    joint_context_fraction=0.25,
):
    """Render full scene, shared relation context, and two candidate details.

    The shared middle view is the important difference from independent crop
    scoring: both candidates and their nearby anchors remain visible at once.
    Candidate order is deliberately the only semantic label so the same image
    can be rendered again in reversed order for permutation calibration.
    """

    if len(candidates) != 2:
        raise ValueError("pairwise relation montage requires exactly two candidates")
    image = image.convert("RGB")
    overview, scale, offset = _letterbox(image, overview_size)
    overview_draw = ImageDraw.Draw(overview)
    overview_font = _font(26)
    for index, candidate in enumerate(candidates):
        _draw_candidate_box(
            overview_draw,
            _xywh_to_xyxy(candidate["bbox"]),
            scale,
            offset,
            PALETTE[index],
            str(index + 1),
            overview_font,
            overview.size,
        )

    panel_w, panel_h = panel_size
    montage = Image.new(
        "RGB",
        (max(overview_size[0], panel_w * 3), overview_size[1] + panel_h),
        (245, 245, 245),
    )
    montage.paste(overview, ((montage.width - overview.width) // 2, 0))
    panel_font = _font(24)

    joint_crop, joint_boxes = _joint_context_crop(
        image,
        [candidate["bbox"] for candidate in candidates],
        joint_context_fraction,
    )
    joint_panel, joint_scale, joint_offset = _letterbox(joint_crop, panel_size)
    joint_draw = ImageDraw.Draw(joint_panel)
    for index, box in enumerate(joint_boxes):
        _draw_candidate_box(
            joint_draw,
            box,
            joint_scale,
            joint_offset,
            PALETTE[index],
            str(index + 1),
            panel_font,
            joint_panel.size,
        )
    _draw_tag(
        joint_draw,
        4,
        joint_panel.height - 42,
        "SHARED CONTEXT",
        "#333333",
        _font(18),
        joint_panel.size,
    )
    montage.paste(joint_panel, (0, overview_size[1]))

    for index, candidate in enumerate(candidates):
        crop, local_box = _context_crop(image, candidate["bbox"], context_fraction)
        panel, panel_scale, panel_offset = _letterbox(crop, panel_size)
        draw = ImageDraw.Draw(panel)
        _draw_candidate_box(
            draw,
            local_box,
            panel_scale,
            panel_offset,
            PALETTE[index],
            str(index + 1),
            panel_font,
            panel.size,
        )
        _draw_tag(
            draw,
            4,
            panel.height - 42,
            f"CANDIDATE {index + 1}",
            PALETTE[index],
            _font(18),
            panel.size,
        )
        montage.paste(panel, ((index + 1) * panel_w, overview_size[1]))
    return montage


def build_pairwise_relation_prompt(expression):
    return (
        "Compare two candidate regions for referring-expression grounding. "
        "The top view is the complete scene. The lower-left view keeps both "
        "candidates and their shared context visible, followed by one detail "
        "view for each candidate.\n"
        f'Referring expression: "{expression}"\n'
        "Resolve the comparison explicitly: identify the target category, check "
        "visual attributes, then check left/right, above/below, front/back, "
        "ordinal, and anchor-object relations. Choose the candidate that satisfies "
        "the whole expression. Return exactly 1 or 2, with no explanation."
    )
