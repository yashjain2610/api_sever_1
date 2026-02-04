white_bgd_prompt = """
You are provided with a reference photograph of real earrings.

ABSOLUTE REQUIREMENT (AMAZON COMPLIANCE):
The earrings in the output must be the exact same physical earrings as shown in the reference image.
No redesign, enhancement, beautification, stylization, or modification is allowed.

Preserve exactly:
- Overall geometry, silhouette, scale, and proportions
- Metal thickness, curves, edges, and construction
- Exact stone count, size, shape, color, and placement
- Stud / hook / clasp structure
- Perfect left–right symmetry and alignment

TASK:
Create a high-resolution, photorealistic studio product image suitable for Amazon marketplace listings.
Both earrings must be fully visible, upright, centered, and clearly separated.

BACKGROUND (AMAZON MAIN IMAGE STANDARD):
Pure white (#FFFFFF), seamless, uniform background.
No gradients, textures, or environmental elements.

LIGHTING:
Professional e-commerce catalog lighting:
- Soft, even illumination
- Neutral white balance
- Sharp focus on fine jewelry details
- Minimal, natural contact shadow directly beneath the earrings only

IMAGE FIDELITY:
- True-to-life metal reflections (no exaggerated shine)
- Accurate gemstone clarity and color (no enhancement)
- Clean edges, no warping, no noise, no artifacts
- Real photographic appearance (not CGI or illustration)

STRICTLY PROHIBITED:
- Redesign, enhancement, beautification, or stylization
- Altered proportions or geometry
- Changed, missing, or extra stones
- Extra sparkle or dramatic lighting
- CGI, 3D render, illustration, or artificial look
- Props, hands, lifestyle elements
- Text, logos, borders, or watermarks
- Blur, distortion, color cast, or overexposure

OUTPUT:
Generate only the final polished image.
No text, captions, watermarks, or explanations.
"""

multicolor_1_prompt = """
You are given a reference image of real earrings.

ABSOLUTE CONSTRAINT:
The earrings must remain IDENTICAL to the reference image.
No changes to design, shape, materials, or proportions are allowed.

Preserve exactly:
- Geometry and silhouette
- Metal form and thickness
- Stone number, shape, and placement
- Hook / stud structure
- Left-right symmetry

Create a high-resolution, hyper-realistic studio image of the earrings,
with both left and right earrings clearly visible.

Background:
A soft, elegant multi-color complementary background inspired by the earrings’ materials.
Use 2–3 harmonious hues with smooth blended gradients or subtle iridescent transitions.
Colors should enhance the jewelry without overpowering it.

Lighting & Style:
Soft professional studio lighting,
sharp focus on jewelry details,
natural gentle shadows.

Details:
True-to-life metal texture and gemstone clarity,
no distortion, no color bleeding onto the jewelry.

Atmosphere:
Luxurious, refined, gentle, modern, Instagram-worthy.

Avoid:
Busy or distracting backgrounds, harsh lighting,
unnatural colors, or any jewelry alteration.

Output:
Final image only. No text or explanations.

Negative prompt:
redesign, reinterpretation, altered geometry, missing stones,
extra stones, fantasy jewelry, CGI, illustration, blur, artifacts
"""

multicolor_2_prompt = """
You are given a reference image of real earrings.

ABSOLUTE CONSTRAINT:
The earrings must be an exact match to the reference image.
Do not change the design, materials, or proportions in any way.

Preserve exactly:
- Shape and silhouette
- Metal curves and thickness
- Stone count, size, and placement
- Hook / stud structure
- Correct scale and symmetry

Create a high-resolution, hyper-realistic editorial-style image
with both earrings fully visible and centered.

Background:
A rich, moody multi-color backdrop using 2–3 deep complementary tones
inspired by the earrings (e.g., sapphire blue, emerald green, champagne gold).
Use soft gradients, subtle bokeh, or refined studio color transitions.

Lighting & Style:
Directional studio lighting with controlled highlights and natural shadows.
High contrast between background and earrings without altering their appearance.

Details:
Accurate metal reflections, crisp gemstone edges,
luxurious and realistic finish.

Atmosphere:
Premium, editorial, sophisticated, moody luxury.

Avoid:
Cluttered backgrounds, low resolution,
over-stylization, or jewelry distortion.

Output:
Only the final image. No text or descriptions.

Negative prompt:
redesign, altered proportions, missing stones, extra stones,
changed hook, artistic jewelry, CGI, illustration, blur, artifacts
"""

props_img_prompt = """
You are given a reference image of real earrings.

ABSOLUTE CONSTRAINT:
The earrings in the output MUST be the exact same physical earrings as in the reference image.
Do NOT redesign, reinterpret, stylize, or modify the earrings in any way.

Preserve exactly:
- Geometry and silhouette
- Metal thickness, curves, and edges
- Stone count, size, shape, and placement
- Hook / stud / clasp structure
- Correct proportions and left-right symmetry

Create a high-resolution, hyper-realistic flat lay photograph
of the earrings with both left and right earrings fully visible and well-spaced.

Background:
A luxury surface such as satin fabric, soft velvet, marble, or handmade paper.

Props:
Minimal and elegant props only (e.g., dried flowers, soft petals, small stones, gentle fabric folds).
Use very few props. Props must NOT overlap, obscure, or touch the earrings.

Lighting & Style:
Luxurious flat lay photography with soft, diffused studio lighting.
Shallow depth of field with the earrings in sharp focus.

Details:
Crisp surface textures, true-to-life metal reflections,
accurate gemstone brilliance, clean edges.

Atmosphere:
Refined, aesthetic, boutique-style luxury presentation.

Strictly avoid:
Overpowering or cluttered props, busy composition,
dramatic shadows, color cast, or any jewelry alteration.

Output:
Final image only. No text or explanations.

Negative prompt:
redesign, altered geometry, missing stones, extra stones,
changed hook, changed stud, fantasy jewelry, CGI,
illustration, blur, artifacts, asymmetry
"""


lifestyle_nature_prompt = """
You are given a reference image of real earrings.

ABSOLUTE CONSTRAINT:
The earrings must be IDENTICAL to the reference image.
Do NOT redesign, reinterpret, stylize, or alter the earrings in any way.

Preserve exactly:
- Shape and silhouette
- Metal thickness and curvature
- Stone count, size, and placement
- Hook / stud / clasp structure
- Accurate scale and symmetry

TASK:
Create a high-resolution, hyper-realistic lifestyle flat lay photograph
with both earrings clearly visible and displayed elegantly.

STYLING:
- Earrings placed flat on a beautiful natural surface
- Surface can be: soft beige linen, light marble, wooden texture, or neutral fabric
- Soft green leaves and small botanical elements arranged around the earrings
- Small natural props like dried flowers, eucalyptus leaves, or subtle greenery
- Natural, organic aesthetic with minimal and tasteful props

COMPOSITION:
- Both earrings displayed flat, slightly separated, facing upward
- Leaves and botanical elements placed softly around (not on top of) the earrings
- Clean, balanced flat lay composition
- Earrings must be the clear focal point
- Props in corners or edges, not covering the jewelry

LIGHTING:
- Soft, natural daylight feel
- Even illumination with gentle shadows
- No harsh highlights or dark areas

IMAGE FIDELITY:
- True-to-life metal reflections
- Accurate gemstone clarity and color
- Realistic textures on leaves and surface
- Sharp focus on earrings, slight depth blur on props

ATMOSPHERE:
Organic, natural, elegant, Instagram-worthy, boutique jewelry aesthetic.

STRICTLY PROHIBITED:
- Redesign or alteration of earrings
- Too many props or cluttered composition
- Artificial or fake-looking plants
- Props covering or touching the earrings
- Hands or human elements
- Earrings hanging on sticks or branches

OUTPUT:
Generate only the final polished image.
No text, logos, watermarks, or explanations.

Negative prompt:
redesign, altered geometry, missing stones, extra stones,
changed hook, fantasy jewelry, CGI, illustration, blur,
cluttered, busy background, artificial plants, hanging earrings
"""


dimension_prompt_template = """
You are given a reference image of real earrings.

ABSOLUTE CONSTRAINT:
The earrings must be IDENTICAL to the reference image.
Do NOT redesign, reinterpret, stylize, or alter the earrings in any way.

Preserve exactly:
- Shape and silhouette
- Metal thickness and curvature
- Stone count, size, and placement
- Hook / stud / clasp structure
- Accurate scale and symmetry

TASK:
Create a high-resolution, photorealistic product image showing the earrings with clear dimension markings.

DIMENSION DISPLAY:
- Show HEIGHT dimension: {height}
- Show WIDTH dimension: {width}
- Use clean, thin black lines with arrows at both ends for measurement indicators
- Place dimension lines slightly outside the earring edges (not overlapping the jewelry)
- Add small text labels showing the measurements in a clean, professional font
- Height line should be vertical on the left or right side
- Width line should be horizontal at the top or bottom

BACKGROUND:
Pure white (#FFFFFF), seamless, uniform background.
No gradients, textures, or environmental elements.

LIGHTING:
Professional e-commerce catalog lighting:
- Soft, even illumination
- Neutral white balance
- Sharp focus on fine jewelry details
- Minimal, natural contact shadow directly beneath the earrings only

IMAGE FIDELITY:
- True-to-life metal reflections
- Accurate gemstone clarity and color
- Clean edges, no warping, no noise, no artifacts
- Dimension lines must be crisp and clearly readable

STRICTLY PROHIBITED:
- Redesign or alteration of earrings
- Overlapping dimension lines on the jewelry
- Cluttered or unclear measurements
- Blurry or illegible text
- Props, hands, lifestyle elements

OUTPUT:
Generate only the final image with dimension markings.
No additional text, captions, or explanations outside the dimension labels.

Negative prompt:
redesign, altered geometry, missing stones, extra stones,
changed hook, fantasy jewelry, CGI, illustration, blur, artifacts
"""


model_wearing_prompt = """
You are provided with a reference image of real earrings.

ABSOLUTE CONSTRAINT:
The earrings in the output must be IDENTICAL to the earrings in the reference image.
Do NOT redesign, reinterpret, stylize, enhance, beautify, or alter the earrings in any way.

PRESERVE EXACTLY:
- Overall shape, geometry, and silhouette
- Metal thickness, curvature, edges, and construction
- Exact stone count, size, shape, color, and placement
- Stud / hook / clasp structure
- Accurate real-world scale and proportions
- Left–right symmetry and alignment

CRITICAL SCALE & PERCEPTION REQUIREMENT:
The earrings must appear the SAME SIZE and visual weight as in the reference image.
Do NOT upscale, downscale, slim, enlarge, or visually exaggerate the earrings.
Their perceived size relative to the ear must remain realistic and unchanged.

TASK:
Create a high-resolution, photorealistic lifestyle image showing ONLY a woman’s ear wearing these exact earrings.

SUBJECT REQUIREMENTS:
- Photorealistic female ear (Indian skin tone preferred)
- Age appearance: 25–35 years
- Natural, realistic skin texture
- Well-groomed ear with no heavy retouching
- No visible face, eyes, nose, lips, or full head

COMPOSITION & FRAMING:
- Tight close-up focused exclusively on the ear and earring
- Ear should be upright and naturally positioned
- Earring must be fully visible and unobstructed
- Crop must exclude the full face entirely
- Camera angle must not distort earring size or proportions

HAIR STYLING:
- Hair fully pulled back or cropped out
- Ear and earring completely unobstructed

BACKGROUND:
Soft, neutral studio background
(light gray, beige, or subtle gradient only).

LIGHTING:
- Soft, professional studio lighting
- Even illumination on ear and earring
- Neutral white balance
- Natural metal reflections and gemstone highlights without enhancement

IMAGE FIDELITY:
- Realistic skin pores and texture (no plastic or AI-smoothed look)
- True-to-life metal reflections
- Accurate gemstone clarity and color
- Professional jewelry photography quality

ATMOSPHERE:
Clean, elegant, refined, premium jewelry presentation
with a natural and believable lifestyle feel.

STRICTLY PROHIBITED:
Any redesign, modification, enhancement, beautification, or stylization of the earrings;
any change in geometry, proportions, scale, or perceived size;
missing, added, or altered stones;
changed stud, hook, or clasp structure;
exaggerated sparkle or dramatic lighting;
distorting camera angles or perspective;
CGI, 3D render, illustration, or artificial appearance;
visible face or facial features;
over-smoothed or plastic skin;
hair covering the ear or earring;
props, hands, or lifestyle objects;
text, logos, borders, or watermarks;
blur, noise, artifacts, color shift, or overexposure;
multiple people or multiple ears in frame.

OUTPUT:
Generate only the final polished image.
No text, captions, logos, watermarks, or explanations.
"""


hand_prompt = """
You are given a reference image of real earrings.

ABSOLUTE CONSTRAINT:
The earrings must be IDENTICAL to the reference image.
Do NOT redesign, reinterpret, stylize, or alter the earrings in any way.

Preserve exactly:
- Shape and silhouette
- Metal thickness and curvature
- Stone count, size, and placement
- Hook / stud / clasp structure
- Accurate scale and symmetry

Create a high-resolution, hyper-realistic lifestyle photograph
of the earrings resting naturally on an open human hand,
with both left and right earrings clearly visible.

Background:
Soft neutral beige or ivory tone.

Hand Details:
A natural, well-manicured human hand with palm facing upward.
Realistic skin texture with subtle pores and natural color variation.

Lighting & Style:
Soft natural lighting with gentle highlights and shadows.
Sharp focus on earrings, realistic depth of field.

Details:
Correct scale between earrings and hand,
crisp gemstone clarity, realistic metal reflections,
natural skin detail without over-retouching.

Atmosphere:
Warm, organic, refined, premium lifestyle feel.

Strictly avoid:
Stiff or unnatural hand poses, over-smoothed skin,
harsh or artificial lighting, excessive retouching,
or any jewelry distortion.

Output:
Only the final polished image. No text or annotations.

Negative prompt:
redesign, altered proportions, missing stones, extra stones,
changed hook, changed stud, fantasy jewelry,
CGI, illustration, blur, artifacts, asymmetry
"""


white_bgd_bracelet_prompt = """
    Create a high-resolution, hyper-realistic image of an elegant bracelet, fully visible and centered, precisely based on the provided reference image.

Preserve exactly: design details, colors, textures, and proportions.

Background: pure white, seamless studio backdrop.

Lighting & Style: professional-grade studio lighting, sharp focus emphasizing jewelry details, soft natural shadows, polished and premium aesthetic.

Surface Details: realistic metal reflections, accurate gemstone clarity, no distortions or artifacts.

Overall Atmosphere: clean, crisp, minimalistic, luxury e-commerce presentation.

Strictly avoid: blurry textures, cartoon-like rendering, unrealistic reflections, overexposure, or added graphic elements.

Output: generate only the final polished image, without any additional text, captions, or descriptions.
negative prompt - dont change the design of the bracelet each detail should remain same
"""

multicolor_1_bracelet_prompt = """
you are provided with
Bracelet image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

A high-resolution, hyper-realistic image of an elegant bracelet (fully visible and centered) based on the provided reference image.

The bracelet should retain its exact design, color, texture, and proportions.

Background: soft pastel gradient (pastel pink, lavender, ivory, or marble texture) enhancing the jewelry colors.

Style: professional soft lighting, sharp focus on jewelry details, subtle natural shadows.

Details: accurate material texture, gemstone clarity, no distortions, clean and elegant background contrast.

Atmosphere: luxurious, gentle, Instagram-worthy, sophisticated.

Avoid: distracting backgrounds, harsh lighting, unrealistic colors.

Generate only the final polished image, with no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""


multicolor_2_bracelet_prompt = """
you are provided with
Bracelet image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

A high-resolution, hyper-realistic image of an elegant bracelet (fully visible and centered) based on the provided reference image.

The bracelet should retain its exact design, color, texture, and proportions.

Background: rich textured background (muted teal, matte beige, velvet black, or soft gold) providing elegant contrast.

Style: studio-quality deep lighting, sharp focus on jewelry details, realistic reflections and shadows.

Details: true-to-life gemstone and metal textures, centered composition, luxurious feel.

Atmosphere: premium, editorial, artistic, moody luxury.

Avoid: cluttered backgrounds, low resolution, distortion.

Generate only the final polished image, with no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""


props_img_bracelet_prompt = """
you are provided with
Bracelet image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

A high-resolution, hyper-realistic flat lay image of an elegant bracelet (fully visible) based on the provided reference image.

The bracelet should retain its exact design, color, texture, and proportions.

Background: satin fabric, soft velvet, marble, or handmade paper surface with minimal elegant props (dried flowers, soft petals, small stones, fabric folds) dont add too many props.

Style: luxurious flat lay photography, soft diffused lighting, shallow depth of field focusing on the jewelry.

Details: crisp textures, true-to-life metal shine and stone brilliance, balanced spacing.

Atmosphere: luxurious, Instagram-worthy, sophisticated, aesthetic.

Avoid: overpowering props, busy composition, loss of jewelry focus.

Generate only the final polished image, with no additional text or explanation
negative prompt - dont change the design of the bracelet each detail should remain same

"""


hand_bracelet_prompt = """
you are provided with
Bracelet image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

Create a high-resolution, hyper-realistic image of an elegant bracelet, precisely based on the provided reference image.

Preserve exactly: design, colors, textures, and proportions of the bracelet.

Background: neutral beige or ivory tone. The bracelet should be draped naturally around a well-manicured wrist on an open hand.

Lighting & Style: natural soft lighting, sharp focus highlighting both jewelry and realistic skin textures.

Surface Details: correct scale, crisp metal and gemstone clarity, natural skin pores subtly visible, gentle natural shadows enhancing depth.

Atmosphere: warm, organic, luxurious, and natural feel.

Strictly avoid: unrealistic or stiff poses, over-edited skin, harsh or artificial lighting, excessive retouching.

Output: generate only the final polished image without any additional text, annotations, or explanations.
negative prompt - dont change the design of the bracelet each detail should remain same
"""



white_bgd_necklace_prompt = """
you are provided with
Necklace image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

Create a high-resolution, hyper-realistic image of an elegant necklace (fully visible, including clasp and chain), precisely based on the provided reference image.

Preserve exactly: design details, colors, textures, and proportions.

Background: pure white (#ffffff), seamless studio backdrop.

Lighting & Style: professional-grade studio lighting, sharp focus emphasizing jewelry details, soft natural shadows, polished and premium aesthetic.

Surface Details: realistic metal reflections, accurate gemstone clarity, no distortions or artifacts.

Overall Atmosphere: clean, crisp, minimalistic, luxury e-commerce presentation.

Strictly avoid: blurry textures, cartoon-like rendering, unrealistic reflections, overexposure, or added graphic elements.

Output: generate only the final polished image, without any additional text, captions, or descriptions.
negative prompt - dont change the design of the bracelet each detail should remain same
"""


multicolor_1_necklace_prompt = """
you are provided with
Necklace image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

A high-resolution, hyper-realistic image of an elegant necklace (fully visible, including clasp and chain) based on the provided reference image.

The necklace should retain its exact design, color, texture, and proportions.

Background: soft pastel gradient (pastel pink, lavender, ivory, or marble texture) enhancing jewelry colors.

Style: professional soft lighting, sharp focus on jewelry details, subtle natural shadows.

Details: accurate material texture, gemstone clarity, no distortions, clean and elegant background contrast.

Atmosphere: luxurious, gentle, Instagram-worthy, sophisticated.

Avoid: distracting backgrounds, harsh lighting, unrealistic colors.

Generate only the final polished image, with no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""


multicolor_2_necklace_prompt = """
you are provided with
Necklace image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

A high-resolution, hyper-realistic image of an elegant necklace (fully visible, including clasp and chain) based on the provided reference image.

The necklace should retain its exact design, color, texture, and proportions.

Background: rich textured background (muted teal, matte beige, velvet black, or soft gold) providing elegant contrast.

Style: studio-quality deep lighting, sharp focus on jewelry details, realistic reflections and shadows.

Details: true-to-life gemstone and metal textures, centered composition, luxurious feel.

Atmosphere: premium, editorial, artistic, moody luxury.

Avoid: cluttered backgrounds, low resolution, distortion.

Generate only the final polished image, with no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""


props_img_necklace_prompt = """
you are provided with
Necklace image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

A high-resolution, hyper-realistic flat lay image of an elegant necklace (fully visible, including clasp and chain) based on the provided reference image.

The necklace should retain its exact design, color, texture, and proportions.

Background: satin fabric, soft velvet, marble, or handmade paper surface with minimal elegant props (dried flowers, soft petals, small stones, fabric folds) dont add too many props.

Style: luxurious flat lay photography, soft diffused lighting, shallow depth of field focusing on the jewelry.

Details: crisp textures, true-to-life metal shine and gemstone brilliance, balanced spacing.

Atmosphere: luxurious, Instagram-worthy, sophisticated, aesthetic.

Avoid: overpowering props, busy composition, loss of jewelry focus.

Generate only the final polished image, with no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""

hand_necklace_prompt = """
you are provided with
Necklace image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

Create a high-resolution, hyper-realistic image of an elegant necklace, precisely based on the provided reference image.

Preserve exactly: design, colors, textures, and proportions of the necklace.

Background: neutral beige or ivory tone. The necklace should rest naturally draped over an open hand.

Lighting & Style: natural soft lighting, sharp focus highlighting both jewelry and realistic skin or fabric textures.

Surface Details: correct scale, crisp metal and gemstone clarity, gentle natural shadows enhancing depth.

Atmosphere: warm, organic, luxurious, and natural feel.

Strictly avoid: unrealistic or stiff poses, over-edited skin or fabric, harsh or artificial lighting, excessive retouching.

Output: generate only the final polished image without any additional text, annotations, or explanations.
negative prompt - dont change the design of the bracelet each detail should remain same
"""

neck_necklace_prompt = """
you are provided with
Necklace image (use this to extract the jewelry design—the design should be preserved 100% with no changes).

Create a high-resolution, hyper-realistic image of an elegant necklace, precisely based on the provided reference image.

Preserve exactly: design, colors, textures, and proportions of the necklace.

Background: neutral beige or ivory tone. The necklace should rest naturally around a well-manicured neck.

Lighting & Style: natural soft lighting, sharp focus highlighting both jewelry and realistic skin or fabric textures.

Surface Details: correct scale, crisp metal and gemstone clarity, gentle natural shadows enhancing depth.

Atmosphere: warm, organic, luxurious, and natural feel.

Strictly avoid: unrealistic or stiff poses, over-edited skin or fabric, harsh or artificial lighting, excessive retouching.

Output: generate only the final polished image without any additional text, annotations, or explanations.
negative prompt - dont change the design of the bracelet each detail should remain same
"""


# =============================================================================
# AMAZON EARRINGS CATALOG IMAGE TYPES
# =============================================================================
# Image 1: White background (main product image)
# Image 2: Hand holding earrings
# Image 3: Dimension image (with height/width markings)
# Image 4: Lifestyle - earrings hanging on stick with leaves
# Image 5: Model wearing earrings

EARRING_IMAGE_TYPES = {
    1: {
        "name": "White Background",
        "description": "Pure white background - Amazon main product image",
        "prompt": white_bgd_prompt,
        "requires_dimensions": False
    },
    2: {
        "name": "Hand Holding",
        "description": "Earrings resting on open human hand",
        "prompt": hand_prompt,
        "requires_dimensions": False
    },
    3: {
        "name": "Dimension Image",
        "description": "White background with height/width dimension markings",
        "prompt": dimension_prompt_template,
        "requires_dimensions": True
    },
    4: {
        "name": "Lifestyle Nature",
        "description": "Earrings hanging on stick with leaves background",
        "prompt": lifestyle_nature_prompt,
        "requires_dimensions": False
    },
    5: {
        "name": "Model Wearing",
        "description": "AI-generated model wearing the earrings",
        "prompt": model_wearing_prompt,
        "requires_dimensions": False
    }
}


def get_earring_prompt(image_type: int, height: str = None, width: str = None) -> str:
    """
    Get the appropriate prompt for the given image type.

    Args:
        image_type: Integer 1-5 representing the catalog image type
        height: Height dimension string (e.g., "2.5 cm") - required for type 3
        width: Width dimension string (e.g., "1.8 cm") - required for type 3

    Returns:
        The prompt string ready for use

    Raises:
        ValueError: If invalid image type or missing dimensions for type 3
    """
    if image_type not in EARRING_IMAGE_TYPES:
        raise ValueError(f"Invalid image type: {image_type}. Must be 1-5.")

    config = EARRING_IMAGE_TYPES[image_type]
    prompt = config["prompt"]

    if config["requires_dimensions"]:
        if not height or not width:
            raise ValueError("Dimension image (type 3) requires both height and width parameters.")
        prompt = prompt.format(height=height, width=width)

    return prompt


def list_earring_image_types():
    """Print available image types for reference."""
    print("\n=== Amazon Earrings Catalog Image Types ===")
    for type_id, config in EARRING_IMAGE_TYPES.items():
        print(f"\nType {type_id}: {config['name']}")
        print(f"  {config['description']}")
        if config['requires_dimensions']:
            print("  * Requires: height, width parameters")
    print()