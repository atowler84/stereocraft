"""Draw the app icon: one scene, twice, with the parallax between the two.

The sun barely moves between the panels, the ridge moves more and the
foreground hill most -- which is the thing the depth model spends its time
working out.  Everything is laid out on one big canvas and downsampled, so
there is no antialiasing to think about and the sizes cannot drift apart.

Re-running this overwrites the icon the app ships with:

    python packaging/windows/make_icon.py --preview /tmp
"""

import argparse
import os

from PIL import Image, ImageDraw

S = 2048  # master canvas; every measurement below is a fraction of it
GROUND = (27, 27, 27, 255)  # the same near-black the window uses behind a preview
SKY_TOP = (38, 92, 158)
SKY_LOW = (122, 176, 214)
SUN = (255, 206, 92)
RIDGE = (24, 42, 58)
FORE = (16, 26, 36)
LEFT_EDGE = (232, 74, 74)  # the red/cyan pairing everyone reads as "3D"
RIGHT_EDGE = (74, 205, 217)

SIZES = [256, 128, 64, 48, 32, 24, 16]
# Below this the ridge and the foreground hill collapse into one dark mass and
# the panels are too small to read, so the small tiles are drawn differently.
DETAILED_ABOVE = 48


def panel(width, height, shift, detail=True):
    """One eye's view.  `shift` moves the scene, which is the whole point: the
    near things move further than the far ones."""
    img = Image.new("RGBA", (width, height), SKY_TOP)
    draw = ImageDraw.Draw(img)

    for y in range(height):  # sky, top to bottom
        t = y / height
        draw.line([(0, y), (width, y)],
                  fill=tuple(round(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_LOW)))

    if not detail:  # sky, sun and one horizon is all that survives being tiny
        sun = round(shift * 0.25)
        r = width * 0.19
        cx, cy = width * 0.66 - sun, height * 0.32
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SUN)
        draw.polygon([(-width, height), (width * 0.34 - shift, height * 0.46),
                      (width * 2, height)], fill=RIDGE)
        return img

    # The sun sits far away, so it barely moves between the two eyes.
    sun = round(shift * 0.25)
    r = width * 0.15
    cx, cy = width * 0.68 - sun, height * 0.30
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SUN)

    # Ridge line: middle distance, moves a little.
    ridge = round(shift * 0.6)
    draw.polygon([(-width, height), (width * 0.30 - ridge, height * 0.42),
                  (width * 0.62 - ridge, height * 0.72), (width * 0.84 - ridge, height * 0.52),
                  (width * 2, height)], fill=RIDGE)

    # Foreground hill: nearest, so it moves most.
    draw.polygon([(-width, height * 1.2), (width * 0.14 - shift, height * 0.70),
                  (width * 0.52 - shift, height), (width * 2, height * 1.2)], fill=FORE)
    return img


def rounded(size, radius, fill):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius, fill=fill)
    return img


def build(margin_f=0.10, gutter_f=0.035, edge_f=0.012, detail=True):
    """The whole tile.  The fractions are what the small sizes vary: at 16 px
    the panels have to fill nearly all of it or they turn into a smudge."""
    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    icon.alpha_composite(rounded((S, S), round(S * 0.18), GROUND))

    margin, gutter = round(S * margin_f), round(S * gutter_f)
    pw = (S - 2 * margin - gutter) // 2
    ph = round(pw * 0.80)
    top = (S - ph) // 2
    radius = round(pw * 0.10)
    edge = max(2, round(S * edge_f))

    for x, shift, colour in ((margin, 0, LEFT_EDGE),
                             (margin + pw + gutter, round(pw * 0.07), RIGHT_EDGE)):
        icon.paste(panel(pw, ph, shift, detail), (x, top),
                   rounded((pw, ph), radius, (255, 255, 255, 255)))
        # A thin red / cyan edge: at a glance the 3D cue, and up close it says
        # which panel is which eye.
        ImageDraw.Draw(icon).rounded_rectangle(
            [x, top, x + pw - 1, top + ph - 1], radius, outline=colour + (255,), width=edge)
    return icon


def tiles():
    big = build()
    # Roomier panels and fatter edges: what survives being tiny.
    small = build(margin_f=0.045, gutter_f=0.028, edge_f=0.022, detail=False)
    return [(big if n >= DETAILED_ABOVE else small).resize((n, n), Image.LANCZOS) for n in SIZES]


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, os.pardir, os.pardir, "stereocraft", "stereocraft.ico")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=os.path.normpath(default))
    parser.add_argument("--preview", help="folder to also write a large view and a size strip to")
    args = parser.parse_args(argv)

    frames = tiles()
    frames[0].save(args.output, format="ICO", sizes=[(n, n) for n in SIZES],
                   append_images=frames[1:])
    print(f"wrote {args.output}  ({', '.join(str(n) for n in SIZES)})")

    if args.preview:
        build().resize((512, 512), Image.LANCZOS).save(os.path.join(args.preview, "icon.png"))
        # Every real size on a mid grey, to check it survives being small.
        strip = Image.new("RGBA", (sum(SIZES) + 20 * len(SIZES), 300), (128, 128, 128, 255))
        x = 10
        for n, tile in zip(SIZES, frames):
            strip.alpha_composite(tile, (x, (300 - n) // 2))
            x += n + 20
        strip.save(os.path.join(args.preview, "icon-sizes.png"))
        print(f"wrote previews to {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
