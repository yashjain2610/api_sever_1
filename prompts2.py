white_bgd_prompt = """
you are provide with
Earrings image (use this to extract the jewelry design the design shoulb be preserved 100% no changes should be there).

"Create a high-resolution, hyper-realistic image of an elegant earrings (both left and right earrings fully visible), precisely based on the provided reference image.

Preserve exactly: design details, colors, textures, and proportions.

Background: pure white, seamless studio backdrop.

Lighting & Style: professional-grade studio lighting, sharp focus emphasizing jewelry details, soft natural shadows, polished and premium aesthetic.

Surface Details: realistic metal reflections, accurate gemstone clarity, no distortions or artifacts.

Overall Atmosphere: clean, crisp, minimalistic, luxury e-commerce presentation.

Strictly avoid: blurry textures, cartoon-like rendering, unrealistic reflections, overexposure, or added graphic elements.

Output: generate only the final polished image, without any additional text, captions, or descriptions.
negative prompt - dont change the design of the bracelet each detail should remain same
"""

multicolor_1_prompt = """
you are provide with
Earrings image (use this to extract the jewelry design the design shoulb be preserved 100% no changes should be there).
A high-resolution, hyper-realistic image of an elegant earrings (both left and right visible) based on the provided reference image.

The earrings should retain their exact design, color, texture, and proportions.

Background: Generate a dynamic, multi-color complementary background uniquely tailored to the natural hues and materials of the earrings. Use soft blended gradients, iridescent overlays, or gentle color transitions that incorporate 2–3 harmonious colors inspired by the metal, gemstones, or enamel in the earrings. For example, pastel blends with metallic hints, blurred floral tones, or shimmering marble with gold flecks. Avoid repetition; ensure each background is distinct and elevates the elegance of the design without overpowering it.
Style: Professional soft lighting, sharp focus on jewelry details, subtle natural shadows.

Details: Accurate material texture, gemstone clarity, no distortions, clean and elegant background contrast.

Atmosphere: luxurious, gentle, Instagram-worthy, sophisticated.

Avoid: distracting backgrounds, harsh lighting, unrealistic colors.

Generate only the final image, no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""

multicolor_2_prompt = """
you are provide with
Earrings image (use this to extract the jewelry design the design shoulb be preserved 100% no changes should be there).

A high-resolution, hyper-realistic image of an elegant earrings (both left and right visible) based on the provided reference image.

The earrings should retain their exact design, color, texture, and proportions.

Background: choose a *dynamic, multi-color complementary* backdrop that is visually distinct from the first. Use refined blends of 2–3 hues inspired by the earrings’ materials—such as warm golds with muted rose, sapphire blues with soft greys, or emerald greens merging into champagne beige. Incorporate high-end effects like artistic bokeh with tinted highlights, luxurious marble gradients with metallic flecks, or soft studio-style color transitions. The background must enhance the metal and gemstone elements while maintaining a clean, editorial aesthetic. Every image should feature a unique, thoughtfully designed background.

Style: Studio-quality deep lighting, sharp focus on jewelry details, realistic reflections and shadows.

Details: True-to-life gemstone and metal textures, centered composition, luxurious feel.

Atmosphere: premium, editorial, artistic, moody luxury.

Avoid: cluttered backgrounds, low resolution, distortion.

Generate only the final image, no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""

props_img_prompt = """
you are provide with
Earrings image (use this to extract the jewelry design the design shoulb be preserved 100% no changes should be there).

A high-resolution, hyper-realistic flat lay image of a pair of elegant earrings (left and right visible) based on the provided reference image.

The earrings should retain their exact design, color, texture, and proportions.

Background: satin fabric, soft velvet, marble, or handmade paper surface with minimal elegant props (dried flowers, soft petals, small stones, fabric folds) dont add too many props.

Style: Luxurious flat lay photography, soft diffused lighting, shallow depth of field focusing on jewelry.

Details: Crisp textures, true-to-life metal shine and stone brilliance, balanced spacing.

Atmosphere: luxurious, Instagram-worthy, sophisticated, aesthetic.

Avoid: overpowering props, busy composition, loss of jewelry focus.

Generate only the final image, no additional text or explanation.
negative prompt - dont change the design of the bracelet each detail should remain same
"""


hand_prompt = """
you are provide with
Earrings image (use this to extract the jewelry design the design shoulb be preserved 100% no changes should be there).
Create a high-resolution, hyper-realistic image of a pair of elegant earrings (both left and right earrings clearly visible), precisely based on the provided reference image.

Preserve exactly: design, colors, textures, and proportions of the earrings.

Background: neutral beige or ivory tone. The earrings should rest naturally on an open, well-manicured human hand with the palm facing upward.

Lighting & Style: natural soft lighting, sharp focus highlighting both jewelry and realistic skin textures.

Surface Details: correct scale, crisp gemstone clarity, natural skin pores subtly visible, gentle natural shadows enhancing depth.

Atmosphere: warm, organic, luxurious, and natural feel.

Strictly avoid: unrealistic or stiff hand poses, over-edited skin, harsh or artificial lighting, excessive retouching.

Output: generate only the final polished image without any additional text, annotations, or explanations.
negative prompt - dont change the design of the bracelet each detail should remain same
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