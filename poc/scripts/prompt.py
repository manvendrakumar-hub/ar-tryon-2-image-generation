"""One prompt template for every face. PID attributes come from the catalog data."""

TEMPLATE = """Three reference images:
- Image 1 is the SCENE: a Lenskart catalog photo of a model wearing eyewear.
- Image 2 is the CUSTOMER: a photo of the real person who must appear in the final image.
- Image 3 is the PRODUCT: the exact eyewear ({pid_desc}).

Recreate Image 1 exactly: same crop, framing, camera angle, head pose and tilt, gaze direction, lighting, background, clothing and colour grade. The only change: the person is now the CUSTOMER from Image 2.

IDENTITY LOCK (from Image 2 only): same face shape and bone structure, eyes, eyebrows, nose, lips, ears, skin tone, skin texture, facial hair exactly as in Image 2, hairline and hairstyle, apparent age. The customer must be instantly recognisable to their own family. Do not beautify, slim, lighten, smooth or make the face more symmetrical. Ignore Image 2's background, clothing, lighting, camera angle and any glasses the customer may be wearing there.

PRODUCT LOCK: the customer wears exactly the eyewear from Image 3, as worn in Image 1: same frame shape, proportions, {frame_color} colour and finish, rim thickness, bridge, temples, hinges{lens_clause}. Do not restyle, recolour, add patterns, tortoiseshell mottling, logos or text. The frame sits naturally on the customer's nose bridge and ears at a realistic size for their face, with natural contact shadows and soft lens reflections.

NATURAL RENDER LOCK: photorealistic photograph, neutral documentary colour, visible pores and fine skin texture, natural expression. No HDR, no beauty filter, no plastic or waxy skin, no extra people, no text or watermark."""


def build(pid_desc: str, frame_color: str, sunglasses: bool, lens_color: str = "") -> str:
    lens_clause = (
        f", and {lens_color} tinted sunglass lenses with the same opacity" if sunglasses
        else ", and clear prescription lenses"
    )
    return TEMPLATE.format(pid_desc=pid_desc, frame_color=frame_color, lens_clause=lens_clause)
