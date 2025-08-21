prompt_questions_earrings_flipkart = """
you are jewellery expert and an e-commerce catalog manager for flipkart,you will be given a image of a jewellery item.
answer the given questions. If options are provided for a question, then you have to choose the right answer from that option only.
if no options are provided generate response on your expertise in one or two words max. dont write not applicable analyse the image and generate a response

Questions {
    {
        field_name: Type
        question: What is the type of the earring?
        options: Chandbali Earring, Clip-on Earring, Cuff Earring, Drops & Danglers, Ear Thread, Earring Set, Hoop Earring, Huggie Earring, Jhumki Earring, Magnetic Earring, Plug Earring, Rhinestone Studs, Stud Earring, Tassel Earring, Tunnel Earring
    }
    {
        field_name: Color
        question: What is the color of the jewellery?
        options: Aqua, Beige, Black, Blue, Bronze, Brown, Copper, Gold, Green, Grey, Maroon, Multicolor, Orange, Pink, Platinum, Purple, Red, Rose Gold, Sea Green, Silver, Turquoise, White, Yellow
    }
    {
        field_name: Base Material
        question: what is the base material of the item, Base Material refers to the substance used to make the bezel/case/body of a particular product.
        options: Acrylic, Alloy, Aluminum, Bone, Brass, Bronze, Ceramic, Cobalt, Copper, Cotton Dori, Crystal, Enamel, Fabric, German Silver, Glass, Gold, Ivory, Jute, Lac, Leather, Metal, Mother of Pearl, Nickel, Paper, Plastic, Porcelain, Resin, Ribbon, Rubber, Shell, Silicone, Silk Dori, Silver, Stainless Steel, Steel, Sterling Silver, Stone, Terracotta, Tungsten, White Metal, Wood, Zinc
    }
    {
        field_name: Gemstone
        question: What is the gemstone used in the jewellery?
        options: Agate, Alexandrite, Amber, Amethyst, Andalusite, Aquamarine, Beads, Beryl, Black Diamond, Blue Sapphire, Carnelian, Cat's Eye, Chalcedony, Citrine, Coral, Crystal, Cubic Zirconia, Danburite, Diamond, Diopside, Emerald, Garnet, Iolite, Jade, Kyanite, Labradorite, Lapis Lazuli, Malachite, Moissanite, Moonstone, Morganite, Mother of Pearl, NA, Onyx, Opal, Orange Sapphire, Pearl, Peridot, Quartz, Ruby, Sapphire, Spinel, Swarovski Crystal, Swarovski Zirconia, Tanzanite, Tiger Eye, Titanium Drusy, Topaz, Tourmaline, Tsavorite, Turquoise, White Zircon, Zircon
    }
    {
        field_name: Pearl Type
        question: What is the pearl type of the jewellery? Pearl Type refers to the type of Pearls that the product comes with.
        options: Cultured, Freshwater, Plastic, South Sea, Tahitian, NA
    }
    {
        field_name: Collection
        question: What is the collection?
        options: Ethnic, Contemporary
    }
    {
        field_name: Occasion
        question: What is the occasion?
        options: Everyday, Love, Party, Religious, Wedding & Engagement, Workwear
    }
    {
        field_name: Piercing Required
        question: Is piercing required for this earring? Piercing Required refers to whether there is any piercing required to wear the product or not. Possible values are Yes or No.
        options: Yes, No
    }
    {
        field_name: Number of Gemstones
        question: How many gemstones are there in the jewellery?
    }
    {
        field_name: Earring Back Type
        question: What is the earring back type?, Earring Back Type refers to the type of locking mechanism that the earrings have at the back. Possible values are Butterfly Back, Push Back, Screw Back, Locking Back, French Back, Lever Back, Friction Back, Fish Hook Back, etc.
    }
    {
        field_name: Finish
        question: What is the finish of the jewellery?, Finish refers to the final appearance given to the base material of a product. Possible values are Matte, Glossy, Shimmer, etc.
    }
    {
        field_name: Design
        question: What is the design?
        options: Heavy, Minimal, Statement
    }
    {
        field_name: Metal Color
        question: What is the metal colour of the jewellery? Metal Color refers to the color of the metals present in the product. Possible values are Black, White, etc.
    }
    {
        field_name: Natural/Synthetic Diamond
        question: Is the diamond used in the jewellery natural or synthetic? Natural/Synthetic Diamond refers to whether the stone is a natural produced stone or synthetically produced/cultivated in a laboratory/industry.
        options: Natural, Synthetic
    }
    {
        field_name: Pearl Shape
        question: What is the shape of the pearl used in the jewellery? Pearl Shape refers to the physical form of the pearl.
    }
    {
        field_name: Pearl Grade
        question: what is the grade of the pearl , Pearl Grade refers to the quality with respect to the existence and visual appearance of internal characteristics of a pearl. Possible values are BB Grade, A Grade, AA Grade, AA+ Grade, AAA Grade, etc.
    }
    {
        field_name: Pearl Color
        question: What is the color of the pearl used in the jewellery?
    }
    {
        field_name: Other Features
        question: give some other feautures of the jewellery item, Other Features refers to any additional information on the features of the product that would be useful to a customer. Possible values are Handmade, Bendy & Wearable, Gentle Fit, etc.
    }
    {
        field_name: Search Keywords
        question: share the top ranking keywords of the item for flipkart marketplace. Make sure the item gets maximum visibility. The keywords should be '::' separated.
    }
    {
        field_name: Key Features
        question: share key features of the item(image) as per flipkart standards. dont give in bullet points ,only in one line
    }
    {
        field_name: Trend
        question: What is the trend of the jewellery? select one from the options.
        options: Afghani, American Diamond Jewellery, Antique Picks, Asymmetrical, Big Size, Botanical Designs, Chandbalis, Chandelier, Chuncky Chain Jewellery, Crystals, Cutwork, Dainty-Light, Detailed Tassels, Drop Earrings, Dual Tone, Earcuff, Enamel, Geometrical Shapes, Gold Jhumkas, Handcrafted, Hearts, Hoops, Jhumki, Kundan, Layered, Long Chain, Minimal Pearl Jewellery, Organic Forms, Oversized Danglers, Oxidised Silver, Party Blings, Peacock, Pearl, Rose Gold, Silver, Stud Earrings, Swarovski, Tassel, Tribal, White Gold
    }
    {
        field_name: Closure Type
        question: What is the closure type of the jewellery?
        options: Clip-on, Hooks, Hoopwire, Magnetic, Push Plugs, Screw
    }
    {
        field_name: Sub Type
        question: What is the sub type of the jewellery?
        options: Bar Danglers, Bead Tassels, Chain Cuffs, Chandbali Jhumkis, Chandelier Earring, Closed Hoop, Dangle Earring, Dangler Tunnels, Dreamcatchers, Drop Tassels, Ear Spike, Fan Tassels, Hoop Chandbali, Hoop Jhumkis, Layered Hoop, Layered Jhumki, Mesh Danglers, Needle Thread, Open Hoop, Pom Pom Tassels, Stud Tunnels, Tiered Tassels
    }
    {
        field_name: Earring Shape
        question: what is the earring shape of the product, Earring Shape refers to the physical form of a product.
    }
    {
        field_name: With Ear Chain
        question: Does the jewellery have an ear chain?
        options: Yes, No
    }
    {
        field_name: Earring Set Type
        question: What is the earring set type?
        options: Chandbali Earring, Cuff Earring, Drops & Danglers, Hoop Earring, Jhumki Earring, Stud Earring, Tunnel Earring
    }
    {
        field_name: Ornamentation Type
        question: What is the ornamentation type of the jewellery? Ornamentation Type refers to the decorative elements added to the product to enhance its appearance and make it more attractive. Select only one option.
        options: Beads, Coins, Cutwork/Filigree, Dried Flowers, Enamel Decorations, Feather, Gemstones, Ghungroo, Glitter, Hand-painted, Kundan, Mirror Work, Pearl, Pom Poms, Stones, Tassel
    }
    {
        field_name: Trend AW 16
        question: What is the Trend 2 of the jewellery? Trend AW 16 refers to the name of the trend for Autumn Winter 2016. Select from the options.
        options: Abstract Geometric Danglers, Asymmetric Earring, Chain Links, Chandbali, Chandelier Earrings, Charm Studs, Double Side Earrings, Ear Jacket Double Side Earrings, Ear cuffs, Fine Hoop Earrings, Gold Metal Jhumkas, Jhumka with Long Chain, Long Chain Drop Earrings, Minimal Gold Tone Jewelry, Offbeat Danglers, Silver Metal Jhumkas, Spike Earrings, Statement Jewelry, Tassel, Tribal, Unpaired Earring, Upside Down Asymmetric Earrings
    }
    {
        field: is_scale,
        question: Is there a scale in the image besides the jwellery item? 
        options: Yes, No
    }
}
For output format = give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""

prompt_description_earrings_flipkart = """
        you are jewellery expert,you will be given a image of a jewellery item.
        for the jewellery item Create a flipkart friendly description with top ranking keywords as per marketplace trends with 10k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footers
    """


prompt_dimensions_earrings_flipkart = """
    You are given an image containing a single earring placed next to a metric ruler. The full visible vertical length of the ruler is 21 cm, and the full visible horizontal length of the ruler is 15 cm. Using the ruler as a scale reference, perform the following measurements on the earring that appears most clearly in the front view:
    if no scale is present guess the dimensions to the best of your thinking.
Width (mm): Determine the maximum horizontal distance from the leftmost edge of the earring to its rightmost edge. Measure that span against the ruler in centimeters, then convert to millimeters (multiply by 10).

Height (mm): Determine the maximum vertical distance from the topmost point of the earring to its bottommost point. Measure that span against the ruler in centimeters, then convert to millimeters (multiply by 10).

Diameter (mm): Identify any circular component of the earring (for example the central stone or circular frame). Measure its diameter in centimeters using the ruler, then convert to millimeters (multiply by 10).

Output Requirements:

Provide only a JSON object with the following keys:

"width (mm)"

"height (mm)"

"diameter (mm)"

All values must be numbers (integers or decimals) in millimeters.

Do not include any additional text, explanations, or units.

For example, a valid response should look like:
{"width (mm)": 35.0, "height (mm)": 50.5, "diameter (mm)": 14.5}

Begin your analysis of the image and return only the raw JSON with measured values.
"""


prompt_description_necklace_flipkart = """
        you are jewellery expert,you will be given a image of a jewellery item.
        for the jewellery item Create a flipkart friendly description with top ranking keywords as per marketplace trends with 10k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footers
"""


prompt_questions_necklace_flipkart = """
you are a jewellery expert and an e-commerce catalog manager for flipkart.
you will be given an image of a jewellery item.
answer the given questions. if options are provided, choose the most accurate one based on the image.
if no options are given, respond in one or two words using your expertise. try not write "not applicable/NA/NULL".
dont skip any question give answer for each.

Questions {
    {
        field_name: Base Material,
        question: What is the base material of the jewellery?,
        options: Alloy, Brass, Ceramic, Copper, Crystal, Dori, Fabric, Glass, Jute, Lac, Leather, Metal, Mother of Pearl, Oxidised Silver, Paper, Plastic, Resin, Rubber, Rudraksha, Shell, Silver, Stainless Steel, Steel, Sterling Silver, Stone, Terracotta, Wood
    },
    {
        field_name: Type,
        question: What is the type of this jewellery?,
        options: Chain, Chain Set, Choker, Layered, Necklace, Necklace Set
    },
    {
        field_name: Color,
        question: What is the primary color of the jewellery?,
        options: Beige, Black, Blue, Bronze, Brown, Copper, Gold, Green, Grey, Lavender, Lime, Maroon, Multicolor, Orange, Pink, Platinum, Purple, Red, Rose, Rose Gold, Silver, Turquoise, White, Yellow
    },
    {
        field_name: Gemstone,
        question: What is the gemstone used in this jewellery?,
        options: Agate, Alexandrite, Amber, Amethyst, Aquamarine, Beads, Beryl, Carnelian, Cat's Eye, Chalcedony, Citrine, Coral, Crystal, Cubic Zirconia, Diamond, Emerald, Garnet, Jade, Lapis Lazuli, Malachite, Moissanite, Moonstone, Mother of Pearl, NA, Onyx, Opal, Pearl, Peridot, Quartz, Rose Quartz, Ruby, Sapphire, Spinel, Swarovski Crystal, Tanzanite, Tiger's Eye, Topaz, Tourmaline, Tsavorite, Turquoise, Zircon
    },
    {
        field_name: Ideal For,
        question: Who is the jewellery ideal for?,
        options: Baby Boys, Baby Girls, Boys, Girls, Men, Women
    },
    {
        field_name: Certification,
        question: Is the jewellery certified? (if yes, mention certification),
        options: BIS Hallmark, Brand Certification, EGL, GIA, GSL, HKD, IDI, IGI, IGL, NA, SGL, Swarovski Authenticity
    },
    {
        field_name: Plating,
        question: What is the plating used?,
        options: 800 Silver, 830 Silver, 900 Silver, 958 Silver, 999 Silver, Black Silver, Brass, Copper, Enamel, Gold-plated, NA, Palladium, Platinum, Rhodium, Silver, Sterling Silver, Titanium
    },
    {
        field_name: Setting,
        question: What is the stone setting type?
    },
    {
        field_name: Occasion,
        question: What is the primary occasion for this jewellery?,
        options: Everyday, Love, Party, Religious, Wedding & Engagement, Workwear
    },
    {
        field_name: Diamond Color Grade,
        question: What is the diamond color grade?,
        options: D, E, EF, F, FG, G, GH, H, HI, I, IJ, J, JK, K, L, M, N, NA, O, P, Q, R, S, T, U, V, W, X, Y, Z
    },
    {
        field_name: Pearl Type,
        question: What type of pearl is used?,
        options: Cultured, Freshwater, NA, Plastic, South Sea, Tahitian
    },
    {
        field_name: Collection,
        question: What collection does the item belong to?,
        options: Ethnic, Contemporary
    },
    {
        field_name: Diamond Cut,
        question: What is the diamond cut?,
        options: Brilliant, NA, Polki, Rose Cut, Uncut
    },
    {
        field_name: Finish,
        question: What is the finish of the jewellery?,
        options: Antique, Bronze, Copper, Embossed, Engraved, Glossy, Gold, Hammered, Matte, Oxidised, Silver, Wooden, Zinc
    },
    {
        field_name: Number of Gemstones,
        question: How many gemstones are there in the jewellery?
    },
    {
        field_name: Necklace Type,
        question: What is the necklace type?,
        note: Necklace Type refers to the type of necklace the product is. Possible values are Choker, Princess Necklace, Matinee Necklace, etc.
    },
    {
        field_name: Clasp Type,
        question: What is the clasp type?,
        options: Barrel Clasp, Fish Hook, Fold Over, Hook and Eye Barrel, Lobster Claw, S-hook, Toggle
    },
    {
        field_name: Pearl Color,
        question: What is the pearl color?
    },
    {
        field_name: Pearl Shape,
        question: What is the shape of the pearl?,
        note: Pearl Shape refers to the physical form of the pearl. Possible values are Round, Semi-round, Button, Drop, Pear, Oval, Baroque, or Circled.
    },
    {
        field_name: Other Features,
        question: What other features does the jewellery have?,
        note: Other Features refers to any additional information on the features of the product that would be useful to a customer. Possible values are Handmade, Bendy & Wearable, Gentle Fit, etc.
    },
    {
        field_name: Search Keywords,
        question: Share top search keywords for Flipkart marketplace (use '::' to separate)
    },
    {
        field_name: Key Features,
        question: Share the key features of the item in one sentence (no bullet points)
    },
    {
        field_name: Trend,
        question: What is the trend of the jewellery?,
        options: Afghani, Antique Picks, Boho, Chokers, Chuncky Chain Jewellery, Chunky, Coin Necklace, Crystal, Cutwork, Dainty-Light, Dual Tone, Enamel, Ethnic Chokers, Handcrafted, Kundan, Layered, Long Chain with pendants, Metal Chain neckpiece, Minimal Pearl Jewellery, Organic Forms, Ornaments, Oversized Statement, Oxidized Silver, Peacock, Pearl, Pearl & Uncut Stones, Rose Gold, Silver, Silver - Bold Collection, Statement, Studded, Swarovski, Tassel, Tribal, Tribal Trims, White Gold, White Jewellery
    },
    {
        field_name: Necklace Style,
        question: What is the necklace style?,
        options: Bib Necklace, Choker, Layered Necklace, Multi-String Necklace
    },
    {
        field_name: Necklace Length,
        question: What is the necklace length?,
        options: Choker, Long, Short/Chest Length
    },
    {
        field_name: Work Finish,
        question: What is the work finish?,
        note: Work Finish refers to the final appearance given to the base material of a necklace.,
        options: Antique, Bronze, Copper, Embossed, Engraved, Glossy, Gold, Hammered, Matte, Oxidised, Silver, Wooden, Zinc
    },
    {
        field_name: Chain Style,
        question: What is the chain style of the necklace?,
        options: Bodychain, Link Chain, Mesh Chain, Pendant Chain, Rope Chain, Simple (Single) Chain, Snake Chain
    },
    {
        field_name: Ornamentation Type,
        question: What is the ornamentation type of the jewellery?,
        options: Beads, Coins, Cutwork/Filigree, Dried Flowers, Enamel Decorations, Gemstones, Ghungroo, Glitter, Hand-painted, Kundan, Mirror Work, Stones
    },
    {
        field_name: Pendant Shape,
        question: What is the shape of the pendant?,
        options: Alphabet, Anchor, Animal, Arrow, Bar, Bees, Bird, Bows, Butterfly, Celestial, Coin, Cross, Elephant, Feather, Floral, Fruits, Ganesha, Heart, Hexagon, Infinity, Insects, Irregular, Medallion, Moon, Musical Notes, Om, Oval, Owl, Paisley/Mango, Peacock, Pearl, Religious Symbols, Rose, Round, Spike, Square, Stars, Sun, Tassels, Teardrop, Tree, Triangle
    },
    {
        field_name: Necklace Design,
        question: What is the design style of the necklace?,
        options: Bohemian, Botanical, Chunky, Cluster, Delicate, Floral, Temple
    },
    {
        field_name: Necklace Width,
        question: What is the necklace width?,
        options: Chunky, Delicate
    },
    {
        field_name: Necklace Color,
        question: What is the necklace color?,
        options: Bronze, Copper, Gold, Gold Oxidised, Multicolor, Silver, Silver Oxidised, Wooden, Zinc
    }
    {
        field: is_scale,
        question: Is there a scale in the image besides the jwellery item? 
        options: Yes, No
    }
}

For output format = give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""


prompt_dimensions_necklace_flipkart = """
    You are given an image containing a pair of earrings (front and back view) placed next to a metric scale (calibrated in centimeters). Using the scale as a visual reference, identify the markings of the scale, the full scale in length (vertically) is 21 cm and horizontally is 15 cm, carefully identify and measure the following dimensions of one of the earrings (specifically, the one that appears most clearly in the front view):
    if no scale is present guess the dimensions to the best of your thinking.
    Chain/Necklace Thickness (mm): Chain/Necklace Thickness refers to the thickness of the chain or necklace. Please ensure that the values entered are in Millimeters. Possible values are 1.5, 5, etc. 
    Chain/Necklace Length (inch): Chain/Necklace Length refers to the length of the chain or necklace. Please ensure that the values entered are in Inches. Possible values are 30.5, 32, etc.
    Width (mm): Measure the maximum horizontal extent (breadth, broadness, or wideness) of the entire earring in the front view, from its leftmost to its rightmost point, in centimeters (cm), then convert this measurement to millimeters (mm).
    Height (mm): Measure the maximum vertical extent of the entire earring in the front view, from its topmost to its bottommost point, in centimeters (cm), then convert this measurement to millimeters (mm).
    Depth (mm): Depth refers to the distance from the front to the back of the product. Please ensure that the values entered are in Millimeters. Possible values are 1.5, 5, etc.

    Output Format:  give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown. 
"""



prompt_description_bracelet_flipkart = """
        you are jewellery expert,you will be given an image of a jewellery item.
        for the jewellery item Create a flipkart friendly description with top ranking keywords as per marketplace trends with 10k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footers
"""

prompt_dimensions_bracelet_flipkart = """
    You are given an image containing a pair of earrings (front and back view) placed next to a metric scale (calibrated in centimeters). Using the scale as a visual reference, identify the markings of the scale, the full scale in length (vertically) is 21 cm and horizontally is 15 cm, carefully identify and measure the following dimensions of one of the earrings (specifically, the one that appears most clearly in the front view):
    if no scale is present guess the dimensions to the best of your thinking.
    Pearl Length (mm): Identify the pearl component of the earring and measure its longest dimension in centimeters (cm), then convert this measurement to millimeters (mm).
    Pearl Diameter (mm): Identify the pearl component of the earring and measure its approximate diameter (widest point across) in centimeters (cm), then convert this measurement to millimeters (mm).
    Width (mm): Measure the maximum horizontal extent (breadth, broadness, or wideness) of the entire earring in the front view, from its leftmost to its rightmost point, in centimeters (cm), then convert this measurement to millimeters (mm).
    Chain Length (mm): Chain Length refers to the measure of the length of the chain present in the product. Please ensure that the values entered are in Millimeters. Possible values are 2.5, 3, etc.
    Thickness (mm): Thickness refers to the thickness of the circumference or length of the product. Please ensure that the values entered are in Millimeters. Possible values are 2.5, 3, etc.

    Output Format:  give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""


prompt_questions_bracelet_flipkart = """
you are jewellery expert and an e-commerce catalog manager for flipkart. you will be given a image of a jewellery item.
answer the given questions. If options are provided for a question, then you have to choose the right answer from that option only.
if no options are provided, generate response on your expertise in one or two words max. don't write not applicable — analyze the image and generate a response.

Questions {
    {
        field_name: Type,
        questions: What is the type of the bracelet?,
        options: Armlet, Armlet Set, Bangle, Bangle Set, Bracelet, Bracelet Set, Charm Bracelet, Chudas, Cuff, Hand Thong, Kada, Kalire, Ring Bangle, Ring Bracelet, Shakha Pola, Tennis Bracelet
    },
    {
        field_name: Color,
        questions: What is the color of the bracelet?,
        options: Beige, Black, Blue, Bronze, Brown, Copper, Gold, Green, Grey, Lavender, Lime, Magenta, Maroon, Multicolor, Orange, Pink, Platinum, Purple, Red, Rose, Silver, Turquoise, White, Yellow
    },
    {
        field_name: Base Material,
        questions: What is the base material of the bracelet?,
        options: 7 Chakra, Alloy, Amethyst, Aquamarine, Black Obsidian, Black Picasso Jasper, Brass, Citrine, Citrine + Golden, Clear Quartz, Copper, Crystal, Dalmation, Dori, Evil Eye, Fabric, Glass, Golden Pyrite, Green Aventurine, Green Jade, Howlite, Jute, Lac, Lapiz Lazuli, Lava, Leather, Metal, Mother of Pearl, Oxidised Silver, Paper, Peridot, Plastic, Pyrite, Red, Resin, Rhodonite, Rose Quartz, Rubber, Ruby, Rudraksha, Russian Serpentine, Shell, Silver, Stainless Steel, Steel, Sterling Silver, Stone, Terracotta, Tiger's Eye, Turquoise, Wood
    },
    {
        field_name: Collection,
        questions: Which collection does the bracelet belong to?,
        options: Contemporary, Ethnic
    },
    {
        field_name: Occasion,
        questions: For what occasion is the bracelet suitable?,
        options: Everyday, Love, Party, Religious, Wedding & Engagement, Workwear
    },
    {
        field_name: Body Structure,
        questions: What is the body structure of the bracelet? Body Structure refers to how the body structure of the product is present. Possible values are Open, Closed, Crossover, etc.
    },
    {
        field_name: Number of Gemstones,
        questions: How many gemstones are on the bracelet?
    },
    {
        field_name: Design,
        questions: What is the design style of the bracelet?
    },
    {
        field_name: Finish,
        questions: What is the finish of the bracelet?
    },
    {
        field_name: Tags/Charms Attachable,
        questions: Are tags or charms attachable to the bracelet?,
        options: yes, no
    },
    {
        field_name: Tag/Charm Shape,
        questions: What is the shape of the tag or charm on the bracelet?
    },
    {
        field_name: Clasp,
        questions: What type of clasp does the bracelet have?,
        options: Barrel Clasp, Buckle, Button, Clip-On, Drawstring Clasp, Fish Hook, Fold Over, Hinged Clasp, Hook & Eye, Knots & Tie-ups, Lobster, Lock and Key, Magnetic Clasp, None, S-Hook, Screw Clasp, Tang, Toggle Clasp
    },
    {
        field_name: Stretchable,
        questions: Is the bracelet stretchable?,
        options: yes, no
    },
    {
        field_name: Adjustable Length,
        questions: Does the bracelet have adjustable length?,
        options: Yes, no
    },
    {
        field_name: Metal Color,
        questions: What is the metal color of the bracelet?
    },
    {
        field_name: Natural/Synthetic Diamond,
        questions: Are the diamonds used in the bracelet natural or synthetic?
    },
    {
        field_name: Pearl Color,
        questions: What is the color of the pearl on the bracelet?
    },
    {
        field_name: Pearl Shape,
        questions: What is the shape of the pearl on the bracelet?
    },
    {
        field_name: Pearl Grade,
        questions: What is the grade of the pearl on the bracelet?
    },
    {
        field_name: Product Title,
        questions: What is the product title of the bracelet?
    },
    {
        field_name: Other Features,
        questions: What other features does the bracelet have? Other Features refers to any additional information on the features of the product that would be useful to a customer. Possible values are Handmade, Bendy & Wearable, Gentle Fit, etc.
    },
    {
        field_name: Trend,
        questions: What trend does the bracelet follow?,
        options: American Diamond Jewellery, Charms, Crystals, Cuff, Cutwork, Dainty-Light, Enamel, Hearts, Kundan, Layered, Long Chain with pendants, Metal Chain, Minimal Pieces, Oxidized Silver, Partywear, Peacock, Pearl, Pearl & Uncut Stones, Rose Gold, Silver, Studded, Swarovski, Tassel, Traditional Bangle Set, Tribal inspired Silver jewellery, White Jewellery
    },
    {
        field_name: Search Keywords,
        questions: Share the top ranking keywords of the item for flipkart marketplace. Make sure the item gets maximum visibility. The keywords should be '::' separated.
    },
    {
        field_name: Key Features,
        questions: Share key features of the item(image) as per flipkart standards. Don't give in bullet points, only in one line
    },
    {
        field_name: Style Type,
        questions: What is the style type of the bracelet?,
        options: Armlet, Chain Link Bracelets, Charms Bracelet, Chunky Cuff Bracelets, Coin Bracelets, Friendship Bracelets, Funky/Quirky Bracelets, Heavy Ornamental Bangle, Heavy Ornamental Bracelets, None, Open Slim Cuff Bracelets, Regular, Religious Bracelets, Sleek Bangle, Sleek/ Dainty Bracelets, Snake Chains, Tribal Look Bracelets
    },
    {
        field_name: Ornamentation Type,
        questions: What type of ornamentation does the bracelet have?,
        options: Beads, Coins, Cutwork/Filigree, Dried Flowers, Enamel Decorations, Floral Design, Fringes, Fur, Gemstones, Ghungroo, Glitter, Gold Details, Hand-painted, Kundan, Metal Embellishment, Mirror Work, None, Pearl, Pom Pom, Sequins, Stones, Tassels, Thread Work
    },
    {
        field_name: Brand Color,
        questions: What is the brand color of the bracelet?
    }
    {
        field: is_scale,
        question: Is there a scale in the image besides the jwellery item? 
        options: Yes, No
    }
}

output format: give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""


prompt_description_earrings_amz = """
        you are jewellery expert,you will be given an image of a jewellery item.
        for the jewellery item Create an amazon friendly description with top ranking keywords as per marketplace trends with 2k max characters
        in the description include all the key features of the item. it should be in between 475 and 500 characters.
        give only description dont add any headers or footers
"""

prompt_questions_earrings_amz = """
    you are a jewellery expert and an e-commerce catalog manager for amazon. you will be given an image of a jewellery item.
    answer the given questions. If options are provided for a question, choose the correct answer based on the image.
    if no options are given, respond based on your expert analysis using one or two words. do not write "not applicable".
    give output in json format with field names and answer as key value pairs,Return only valid raw JSON.
Questions {
    {
        field: item_name,
        question: What is the item name (title) of the jewellery give a title like the following example? give in max 200 characters example title - "Alya Jewels Vintage-Style Flower Stud Earrings, Rose Gold and Silver Glitter Finish, 5-Petal Floral Design, Pack of 1" give title according to design of earrings and this example
    }
    {
        field: stone_color,
        question: What is the stone colour used in the jewellery?
        options: D, D-E, E, E-F, F, F-G, G, G-H, H, H-I, I, I-J, J, J-K, K, K-L, L, L-M, M, M-N, N-O, O-P, P-Q, Q-R, R-S, S-T, T-U, U-V, V-W, W-X, X-Y, Y-Z, Beige, Black, Blue, Blue Green, Blue Violet, Brown, Champagne, Clear, Color Changing, Cream, Gold, Gray, Green, Green Blue, Green Yellow, Metallic, Multiple Colors, Orange, Orange Red, Orange Yellow, Pink, Red, Red Orange, Red Violet, Silver, Violet, Violet Blue, Violet Red, White, Yellow, Yellow Green, Yellow Orange, Other Color
    }
    {
        field: style_name,
        question: What is the style of the jewellery?
        options: Western, Traditional, Bohemian, Contemporary, Minimal, Retro
    }
    {
        field: stones_color,
        question: what is the stone colour used in the item
        options: D, D-E, E, E-F, F, F-G, G, G-H, H, H-I, I, I-J, J, J-K, K, K-L, L, L-M, M, M-N, N-O, O-P, P-Q, Q-R, R-S, S-T, T-U, U-V, V-W, W-X, X-Y, Y-Z, Beige, Black, Blue, Blue Green, Blue Violet, Brown, Champagne, Clear, Color Changing, Cream, Gold, Gray, Green, Green Blue, Green Yellow, Metallic, Multiple Colors, Orange, Orange Red, Orange Yellow, Pink, Red, Red Orange, Red Violet, Silver, Violet, Violet Blue, Violet Red, White, Yellow, Yellow Green, Yellow Orange, Other Color
    }
    {
        field: item_type_name,
        question: What is the item type name?
        options: Earrings, Spike, Hoop Earrings, Ball Earrings, Clip On Earrings, Dangle Earrings, Jhumkis, Stud Earrings, Ear Cuffs, Drop Earrings, Earring Jackets
    }
    {
        field: occasion_type1,
        question: What is the primary occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: occasion_type2,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: occasion_type3,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: occasion_type4,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: occasion_type5,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: clasp_type,
        question: What is the clasp type of the item?
        options: Lobster Claw, Spring Ring, Box Clasp, Toggle, S-Hook, Magnetic
    }
    {
        field: color_map,
        question: What is the color map of the jewellery?
        options: Bronze, Brown, Gold, Blue, Multi, Black, Orange, Clear, Red, Silver, Off-white, Pink, White, Metallic, Beige, Ivory, Purple, Yellow, Turquoise, Grey, Green
    }
    {
        field: material_type_free1,
        question: What is the free-form material type?
        options: Cadmium Free, Aluminium Free, Nickel Free, Plastic Free, Copper Free, Lead Free
    }
    {
        field: stone_shape,
        question: What is the stone shape?
        options: Heart, Bullet, Pear, Hexagonal, Square, Rectangular, Star, Triangular, Oval, Irregular, Round, Octagonal
    }
    {
        field: stone_clarity,
        question: What is the stone clarity?
        options: Slightly Included, Very Slightly Included, Highly Included, Included, Internally Flawless, Moderately Included, Flawless, Very Very Slightly Included		
    }
    {
        field: collection_name,
        question: What collection does this jewellery belong to?
        options: Ethnic, Contemporary
    }
    {
        field: material_type,
        question: What is the material type?
        options: Amber, Rattan, Sterling Silver, Metal, Stone, Zircon, Stainless Steel, Resin, Titanium, White Gold, Copper, Polyvinyl Chloride, Bronze, Created Pearl, Faux Fur, Polyester, Silicone, Rose Gold, Aluminium, Yellow Gold, Pearl, Silver, Alloy Steel, Acrylic, Crystal, Gemstone, Brass, Glass, Leather, Opal, Iron, Plastic, Cotton, Shell, Silk, Cubic Zirconia, Platinum, Rayon, Rhinestone, Rubber, Turquoise, Faux Leather, Wood, Diamond, Created Opal, Zinc, Coral, Agarwood, Ceramic, Paper
    }
    {
        field: stones_number_of_stones,
        question: How many stones are there in the jewellery? if no stones are there then answer 0.
    }
    {
        field: setting_type,
        question: What is the setting type?
        options: Wrap, Pave, Post, Channel, Shared Prong, Cocktail, Cluster, Flush, Illusion, Micro Pave, Half Bezel, Bezel, Bar, Bead, Prong, Tension, Invisible, Halo
    }
    {
        field: bullet_point1,
        question: Write Bullet Point 1 (150 characters) highlighting the style, appeal, or use case of the jewellery. Begin with a bolded or striking benefit-focused phrase (e.g., 'Exquisite Design', 'Timeless Appeal'). Follow it with a brief, engaging sentence describing how it enhances the wearer's look or complements various outfits or occasions. dont use '*' or other special characters.
    }
    {
        field: bullet_point2,
        question: Write Bullet Point 2 (150 characters) focusing on the material, craftsmanship, or design details. Start with a phrase like 'Premium Quality' or 'Expert Craftsmanship'. Mention the type of metal, finish, or design technique used and the benefits it provides (e.g., durability, elegance, or polish). dont use '*' or other special characters.
    }
    {
        field: bullet_point3,
        question: Write Bullet Point 3 (150 characters) describing the stone or embellishments used, along with their shape, color, or cut. Begin with a phrase like 'Elegant Embellishments' or 'Stunning Stones', and explain how they add sparkle, charm, or sophistication to the jewellery.dont use '*' or other special characters.
    }
    {
        field: bullet_point4,
        question: Write Bullet Point 4 (150 characters) related to the comfort, closure type, or ease of wearing. Begin with something like 'All-Day Comfort' or 'Secure Fit'. Mention features such as lightweight design, comfortable wear, or the clasp type (e.g., Lobster Claw, Spring Ring). dont use '*' or other special characters.
    }
    {
        field: bullet_point5,
        question: Write Bullet Point 5 (150 characters) presenting the gifting potential or occasion versatility of the jewellery. Start with a phrase like 'Perfect Gift Choice' or 'Versatile Accessory'. Mention if it’s ideal for birthdays, anniversaries, weddings, or daily wear. dont use '*' or other special characters.
    }
    {
        field: color_name,
        question: What is the overall color of the item?
    }
    {
        field: stones_shape,
        question: What is the stone shape?
        options: Heart, Bullet, Pear, Hexagonal, Square, Rectangular, Star, Triangular, Oval, Irregular, Round, Octagonal
    }
    {
        field: theme,
        question: What is the theme of the jewellery?
        options: Vegetables, Moon, Stars, Fruits, Mountains, Feathers, Leaves, Beach, Anime, Candy, Fish, Skeleton, Evil Eye, Movies, Sports, Nautical, Religion, Birds, Plants, Floral, Sun, Animals, Insects, Television, Love, Music, Fantasy, Lightning Bolt
    }
    {
        field: back_finding,
        question: What is the earring back finding?
        options: French Wire, Latch Back, Lever Back, Screw Back, Clip On, Clutchless, La Pousette, Magnetic, Snap Back, French Clip, Hinged Hoop, Push Back
    }
    {
        field: gem_type,
        question: What type of gemstone is used?
        options: Created Alexandrite, Created Cubic Zirconia, Abalone, Alexandrite, Amber, Larimar, Kyanite, Marcasite, Ametrine, Amethyst, Created Sapphire, Zircon, Mystic Topaz, Prasiolite, Aquamarine, Cinnabar, Composite Opal, Andesine, No Gemstone, Ammolite, Created Pearl, Hematite, Tigers Eye, Danburite, Obsidian, Labradorite, Created Emerald, Moonstone, Pearl, Howlite, Apatite, Tanzanite, Jade, Morganite, Rhodonite, Tourmaline, Created Zircon, Moldavite, Tsavorite, Agate, Opal, Chalcedony, Created Diamond, Garnet, Cubic Zirconia, Sapphire, Emerald, Madeira Citrine, Citrine, Gabbro, Nacre, Rhinestone, Ruby, Turquoise, Sandstone, Aventurine, Kunzite, Diamond, Created Opal, Quartz, Carnelian, Created Quartz, Lapis Lazuli, Cats Eye, Jasper, Moissanite, Spinel, Coral, Pietersite, Chrysoberyl, Created Ruby, Onyx, Topaz, Peridot, Created Moissanite
    }
    {
        field: is_scale,
        question: Is there a scale in the image besides the jwellery item? 
        options: Yes, No
    }

    output format: give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
}
"""

prompt_dimensions_earrings_amz = """
    You are given an image containing a pair of earrings placed next to a metric scale (calibrated in centimeters). Using the scale as a visual reference, identify the markings of the scale, the full scale in length (vertically) is 21 cm and horizontally is 15 cm, carefully identify and measure the following dimensions of one of the earrings (specifically, the one that appears most clearly in the front view):
    if no scale is present guess the dimensions to the best of your thinking.
    item_width: Measure the maximum horizontal extent (breadth, broadness, or wideness) of the entire earring in the front view, from its leftmost to its rightmost point, in centimeters (cm).
    item_height: Measure the maximum vertical extent of the entire earring in the front view, from its topmost to its bottommost point, in centimeters (cm).
    item_length: Measure the depth or thickness of the earring — i.e., the front-to-back extent that would be visible if the earring were viewed from the side. This is typically the least prominent dimension in a front-facing image, so use visible shadows, curves, or hooks as reference to estimate it. Provide this value in centimeters (cm).

    Output Format:  give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""





prompt_description_necklace_amz = """
        you are jewellery expert,you will be given an image of a necklace.
        for the jewellery item Create an amazon friendly description with top ranking keywords as per marketplace trends with 2k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footerst
"""

prompt_questions_necklace_amz = """
you are a jewellery expert and an e-commerce catalog manager for amazon. you will be given an image of a necklace.
answer the given questions. If options are provided for a question, choose the correct answer based on the image.
if no options are given, respond based on your expert analysis using one or two words. do not write "not applicable".

Questions {
    {
        field name: item_name
        question: What is the item name (title) of the jewellery? in max 200 characters, example title 
    }
    {
        field name: stone_color
        question: What is the stone colour used in the jewellery?
        options: F-G, H-I, Cream, J-K, Black, Champagne, Green Blue, L-M, Orange, Other Color, N-O, Violet Blue, P-Q, Beige, Blue Violet, R-S, T-U, V-W, Gold, Gray, Violet, X-Y, Yellow Orange, Orange Yellow, Multiple Colors, Silver, Pink, Metallic, Yellow, Violet Red, Color Changing, E-F, Brown, G-H, D, E, F, G, I-J, H, I, J, K, K-L, L, M, Green Yellow, M-N, Clear, Red Violet, O-P, Q-R, Orange Red, S-T, Red Orange, U-V, W-X, Blue, Y-Z, Yellow Green, White, Blue Green, D-E, Green
    }
    {
        field name: style_name
        question: What is the style of the jewellery?
        options: Art Deco, Cape Cod, Bohemian, Contemporary, Casual, Tropical, Victorian, Classic, Asian, Moroccan, Retro, Modern, Mediterranean, Cottage, Scandinavian, Fine, Old World, Southwestern, Baroque
    }
    {
        field name: stones_color
        question: What is the stone colour used in the item?
        options:
    }
    {
        field name: item_type_name
        question: What is the item type name?
        options: Tennis Bracelets, Choker Necklaces, Strand Bracelets, Cuff Bracelets, Pearl Strands, Italian Style Charm Starter Bracelets, Wrap Bracelets, Bangle Bracelets, Strand Necklaces, Charm Bracelets, Link Bracelets, Necklaces, Stretch Bracelets, Bracelets, Chain Necklaces, Identification Bracelets, Anklets, Pendant Enhancers, Locket Necklaces, Link Charm Bracelets, Snake Charm Bracelets, Pendant Necklaces
    }
    {
        field name: occasion_type1
        question: What is the primary occasion the jewellery is suited for?
        options: Festive, Engagement, Promise, Wedding, Anniversary, Christening, Eternity, Birth
    }
    {
        field name: clasp_type
        question: What is the clasp type of the item?
        options: Ball, S-Hook, Slide Lock, Barrel, Box, Buckle, Snap, Hook and Eye, Fishhook, Lobster, Magnetic, Fold Over, Spring Ring, Push Button, Toggle, Swivel, Pinch Clip, Bolt Ring
    }
    {
        field name: color_map
        question: What is the color map of the jewellery?
        options: Brown, Gold, Blue, Multi-Colour, Black, Orange, Clear, Red, Silver, Pink, White, Metallic, Beige, Purple, Yellow, Off-White, Green, Grey
    }
    {
        field name: collection_name
        question: What collection does this jewellery belong to?
        options: Ethnic, Contemporary
    }
    {
        field name: material_type
        question: What is the material type?
        options: Gemstone, Brass, Sterling Silver, Glass, Leather, Plastic, Metal, Stone, Cubic Zirconia, Pewter, Platinum, Stainless Steel, Resin, White Gold, Rhinestone, Copper, Gold, Faux Leather, Wood, Rose Gold, Diamond, Cowrie, Yellow Gold, Zinc, Pearl, Acrylic, Crystal
    }
    {
        field name: stones_type
        question: What type of stones are used in the jewellery?
    }
    {
        field name: number_of_stones
        question: How many stones are there in the jewellery?
    }
    {
        field name: setting_type
        question: What is the setting type?
        options: Wrap, Gypsy, Pave, Channel, Cluster, Illusion, Band, Micro Pave, Bezel, Bar, Bead, Prong, Tension, Invisible, Halo
    }
    {
        field: bullet_point1,
        question: Write Bullet Point 1 (150 characters) highlighting the style, appeal, or use case of the jewellery. Begin with a bolded or striking benefit-focused phrase (e.g., 'Exquisite Design', 'Timeless Appeal'). Follow it with a brief, engaging sentence describing how it enhances the wearer's look or complements various outfits or occasions. dont use '*' or other special characters.
    }
    {
        field: bullet_point2,
        question: Write Bullet Point 2 (150 characters) focusing on the material, craftsmanship, or design details. Start with a phrase like 'Premium Quality' or 'Expert Craftsmanship'. Mention the type of metal, finish, or design technique used and the benefits it provides (e.g., durability, elegance, or polish). dont use '*' or other special characters.
    }
    {
        field: bullet_point3,
        question: Write Bullet Point 3 (150 characters) describing the stone or embellishments used, along with their shape, color, or cut. Begin with a phrase like 'Elegant Embellishments' or 'Stunning Stones', and explain how they add sparkle, charm, or sophistication to the jewellery.dont use '*' or other special characters.
    }
    {
        field: bullet_point4,
        question: Write Bullet Point 4 (150 characters) related to the comfort, closure type, or ease of wearing. Begin with something like 'All-Day Comfort' or 'Secure Fit'. Mention features such as lightweight design, comfortable wear, or the clasp type (e.g., Lobster Claw, Spring Ring). dont use '*' or other special characters.
    }
    {
        field: bullet_point5,
        question: Write Bullet Point 5 (150 characters) presenting the gifting potential or occasion versatility of the jewellery. Start with a phrase like 'Perfect Gift Choice' or 'Versatile Accessory'. Mention if it’s ideal for birthdays, anniversaries, weddings, or daily wear. dont use '*' or other special characters.
    }
    {
        field name: color_name
        question: What is the overall color of the item?
        options: Red, Brown, Pink, White, Blue, Purple, Yellow, Black, Orange, Green, Grey
    }
    {
        field name: stone_shape
        question: What is the shape of the stone used?
        options: Heart, Bullet, Pear, Hexagonal, Square, Rectangular, Star, Triangular, Oval, Irregular, Round, Octagonal
    }
    {
        field name: chain_length_unit
        question: What is the length unit of the chain?
        options: Inches, Centimeters, Millimeters
    }
    {
        field name: theme
        question: What is the theme of the jewellery?
        options: Alphabet, Princess, Bird, Botanical, Geometric Shape, Cartoon, Weapon, Prison, Reptile, Sport, King, Rainbow, Currency, Snowflake, Coastal, Fish, Insect, Comic, Space, Zodiac, Fruit, Nautical, Queen, Religion, City, Animal, Love, Music, Dance, Fantasy, Scary, Skull, Sign
    }
    {
        field name: stone_cut
        question: What is the cut of the stone?
    }
    {
        field name: chain_type
        question: What is the chain type?
        options: Ball, Cable, Cobra, Herringbone, Tennis, Curb, Franco, Railroad, Singapore, Snake, Rolo, Byzantine, Box, Popcorn, Mariner, Omega, Figaro, French Rope, Bar, Crisscross, Wheat, Cord, Boston
    }
    {
        field name: lifecycle_supply_type
        question: What is the lifecycle supply type?
        options: Perennial, Year Round Replenishable, Seasonal Basic, Fashion
    }
    {
        field name: gem_type
        question: What type of gemstone is used?
        options: Created Alexandrite, Lava Stone, Abalone, Alexandrite, Amber, Larimar, Chrysoprase, Amazonite, Amethyst, Created Sapphire, Zircon, Aquamarine, Beryl, Andesine, No Gemstone, Ammolite, Created Pearl, Hematite, Tigers Eye, Sodalite, Obsidian, Serpentine, Labradorite, Created Emerald, Moonstone, Pearl, Selenite, Howlite, Bloodstone, Tanzanite, Jade, Morganite, Rhodonite, Tourmaline, Moldavite, Sunstone, Agate, Opal, Shungite, Created Diamond, Garnet, Cubic Zirconia, Sapphire, Emerald, Citrine, Rhinestone, Ruby, Turquoise, Sandstone, Aventurine, Diamond, Created Opal, Quartz, Carnelian, Lapis Lazuli, Cats Eye, Jasper, Kundan, Created Turquoise, Moissanite, Coral, Jet, Created Ruby, Onyx, Created Topaz, Topaz, Peridot
    }
    {
        field name: pearl_type
        question: What type of pearl is used (if any)?
        options: Saltwater Pearl, South Sea Cultured Pearl, Akoya Cultured Pearl, Conch Pearl, Blister Pearl, Fireballs Pearl, Scallop Pearl, Coin Pearl, Freshwater Cultured Pearl, Blue Mussel Pearl, Biwa Pearl, Seed Pearl, Keshi Cultured Pearl, Basra Pearl, Tahitian Cultured Pearl, Simulated Pearl, Tahitian Pearl, Melo Pearl, Shell Pearl, Abalone Pearl
    }
    {
        field name: pearl_minimum_color
        question: What is the minimum color of the pearl?
    }
    {
        field name: number_of_pearls
        question: How many pearls are there?
    }
    { 
        field: is_scale,
        question: Is there a scale in the image besides the jwellery item? 
        options: Yes, No
    }

    output format: give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
}
"""

prompt_description_bracelet_amz = """
        you are jewellery expert,you will be given an image of a bracelet.
        for the jewellery item Create an amazon friendly description with top ranking keywords as per marketplace trends with 2k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footerst
"""

prompt_questions_bracelet_amz = """
you are a jewellery expert and an e-commerce catalog manager for amazon. you will be given an image of a bracelet.
answer the given questions. If options are provided for a question, choose the correct answer based on the image.
if no options are given, respond based on your expert analysis using one or two words. do not write "not applicable".

Questions {
    {
        field name: item_name
        question: What is the item name (title) of the jewellery? in max 200 characters, example title 
    }
    {
        field name: stone_color
        question: What is the stone colour used in the jewellery?
        options: F-G, H-I, Cream, J-K, Black, Champagne, Green Blue, L-M, Orange, Other Color, N-O, Violet Blue, P-Q, Beige, Blue Violet, R-S, T-U, V-W, Gold, Gray, Violet, X-Y, Yellow Orange, Orange Yellow, Multiple Colors, Silver, Pink, Metallic, Yellow, Violet Red, Color Changing, E-F, Brown, G-H, D, E, F, G, I-J, H, I, J, K, K-L, L, M, Green Yellow, M-N, Clear, Red Violet, O-P, Q-R, Orange Red, S-T, Red Orange, U-V, W-X, Blue, Y-Z, Yellow Green, White, Blue Green, D-E, Green
    }
    {
        field name: style_name
        question: What is the style of the jewellery?
        options: Art Deco, Cape Cod, Bohemian, Contemporary, Casual, Tropical, Victorian, Classic, Asian, Moroccan, Retro, Modern, Mediterranean, Cottage, Scandinavian, Fine, Old World, Southwestern, Baroque
    }
    {
        field name: stones_color
        question: What is the stone colour used in the item?
        options:
    }
    {
        field name: item_type_name
        question: What is the item type name?
        options: Tennis Bracelets, Choker Necklaces, Strand Bracelets, Cuff Bracelets, Pearl Strands, Italian Style Charm Starter Bracelets, Wrap Bracelets, Bangle Bracelets, Strand Necklaces, Charm Bracelets, Link Bracelets, Necklaces, Stretch Bracelets, Bracelets, Chain Necklaces, Identification Bracelets, Anklets, Pendant Enhancers, Locket Necklaces, Link Charm Bracelets, Snake Charm Bracelets, Pendant Necklaces
    }
    {
        field name: occasion_type1
        question: What is the primary occasion the jewellery is suited for?
        options: Festive, Engagement, Promise, Wedding, Anniversary, Christening, Eternity, Birth
    }
    {
        field: occasion_type2,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: occasion_type3,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: occasion_type4,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field: occasion_type5,
        question: give any other occasion the jewellery is suited for?
        options: Easter, Christmas, Graduation, Prom, Anniversary, Birthday, Cocktail Party, Thanksgiving, Womens Day, Bridal Shower, Engagement, New Year, Wedding, Communion, Halloween, Valentines Day	
    }
    {
        field name: clasp_type
        question: What is the clasp type of the item?
        options: Ball, S-Hook, Slide Lock, Barrel, Box, Buckle, Snap, Hook and Eye, Fishhook, Lobster, Magnetic, Fold Over, Spring Ring, Push Button, Toggle, Swivel, Pinch Clip, Bolt Ring
    }
    {
        field name: color_map
        question: What is the color map of the jewellery?
        options: Brown, Gold, Blue, Multi-Colour, Black, Orange, Clear, Red, Silver, Pink, White, Metallic, Beige, Purple, Yellow, Off-White, Green, Grey
    }
    {
        field name: collection_name
        question: What collection does this jewellery belong to?
        options: Ethnic, Contemporary
    }
    {
        field name: material_type
        question: What is the material type?
        options: Gemstone, Brass, Sterling Silver, Glass, Leather, Plastic, Metal, Stone, Cubic Zirconia, Pewter, Platinum, Stainless Steel, Resin, White Gold, Rhinestone, Copper, Gold, Faux Leather, Wood, Rose Gold, Diamond, Cowrie, Yellow Gold, Zinc, Pearl, Acrylic, Crystal
    }
    {
        field name: stones_type
        question: What type of stones are used in the jewellery?
    }
    {
        field name: number_of_stones
        question: How many stones are there in the jewellery?
    }
    {
        field name: setting_type
        question: What is the setting type?
        options: Wrap, Gypsy, Pave, Channel, Cluster, Illusion, Band, Micro Pave, Bezel, Bar, Bead, Prong, Tension, Invisible, Halo
    }
    {
        field: bullet_point1,
        question: Write Bullet Point 1 (150 characters) highlighting the style, appeal, or use case of the jewellery. Begin with a bolded or striking benefit-focused phrase (e.g., 'Exquisite Design', 'Timeless Appeal'). Follow it with a brief, engaging sentence describing how it enhances the wearer's look or complements various outfits or occasions. dont use '*' or other special characters.
    }
    {
        field: bullet_point2,
        question: Write Bullet Point 2 (150 characters) focusing on the material, craftsmanship, or design details. Start with a phrase like 'Premium Quality' or 'Expert Craftsmanship'. Mention the type of metal, finish, or design technique used and the benefits it provides (e.g., durability, elegance, or polish). dont use '*' or other special characters.
    }
    {
        field: bullet_point3,
        question: Write Bullet Point 3 (150 characters) describing the stone or embellishments used, along with their shape, color, or cut. Begin with a phrase like 'Elegant Embellishments' or 'Stunning Stones', and explain how they add sparkle, charm, or sophistication to the jewellery.dont use '*' or other special characters.
    }
    {
        field: bullet_point4,
        question: Write Bullet Point 4 (150 characters) related to the comfort, closure type, or ease of wearing. Begin with something like 'All-Day Comfort' or 'Secure Fit'. Mention features such as lightweight design, comfortable wear, or the clasp type (e.g., Lobster Claw, Spring Ring). dont use '*' or other special characters.
    }
    {
        field: bullet_point5,
        question: Write Bullet Point 5 (150 characters) presenting the gifting potential or occasion versatility of the jewellery. Start with a phrase like 'Perfect Gift Choice' or 'Versatile Accessory'. Mention if it’s ideal for birthdays, anniversaries, weddings, or daily wear. dont use '*' or other special characters.
    }
    {
        field name: color_name
        question: What is the overall color of the item?
        options: Red, Brown, Pink, White, Blue, Purple, Yellow, Black, Orange, Green, Grey
    }
    {
        field name: stone_shape
        question: What is the shape of the stone used?
        options: Heart, Bullet, Pear, Hexagonal, Square, Rectangular, Star, Triangular, Oval, Irregular, Round, Octagonal
    }
    {
        field name: chain_length_unit
        question: What is the length unit of the chain?
        options: Inches, Centimeters, Millimeters
    }
    {
        field name: theme
        question: What is the theme of the jewellery?
        options: Alphabet, Princess, Bird, Botanical, Geometric Shape, Cartoon, Weapon, Prison, Reptile, Sport, King, Rainbow, Currency, Snowflake, Coastal, Fish, Insect, Comic, Space, Zodiac, Fruit, Nautical, Queen, Religion, City, Animal, Love, Music, Dance, Fantasy, Scary, Skull, Sign
    }
    {
        field name: stone_cut
        question: What is the cut of the stone?
    }
    {
        field name: chain_type
        question: What is the chain type?
        options: Ball, Cable, Cobra, Herringbone, Tennis, Curb, Franco, Railroad, Singapore, Snake, Rolo, Byzantine, Box, Popcorn, Mariner, Omega, Figaro, French Rope, Bar, Crisscross, Wheat, Cord, Boston
    }
    {
        field name: lifecycle_supply_type
        question: What is the lifecycle supply type?
        options: Perennial, Year Round Replenishable, Seasonal Basic, Fashion
    }
    {
        field name: gem_type
        question: What type of gemstone is used?
        options: Created Alexandrite, Lava Stone, Abalone, Alexandrite, Amber, Larimar, Chrysoprase, Amazonite, Amethyst, Created Sapphire, Zircon, Aquamarine, Beryl, Andesine, No Gemstone, Ammolite, Created Pearl, Hematite, Tigers Eye, Sodalite, Obsidian, Serpentine, Labradorite, Created Emerald, Moonstone, Pearl, Selenite, Howlite, Bloodstone, Tanzanite, Jade, Morganite, Rhodonite, Tourmaline, Moldavite, Sunstone, Agate, Opal, Shungite, Created Diamond, Garnet, Cubic Zirconia, Sapphire, Emerald, Citrine, Rhinestone, Ruby, Turquoise, Sandstone, Aventurine, Diamond, Created Opal, Quartz, Carnelian, Lapis Lazuli, Cats Eye, Jasper, Kundan, Created Turquoise, Moissanite, Coral, Jet, Created Ruby, Onyx, Created Topaz, Topaz, Peridot
    }
    {
        field name: pearl_type
        question: What type of pearl is used (if any)?
        options: Saltwater Pearl, South Sea Cultured Pearl, Akoya Cultured Pearl, Conch Pearl, Blister Pearl, Fireballs Pearl, Scallop Pearl, Coin Pearl, Freshwater Cultured Pearl, Blue Mussel Pearl, Biwa Pearl, Seed Pearl, Keshi Cultured Pearl, Basra Pearl, Tahitian Cultured Pearl, Simulated Pearl, Tahitian Pearl, Melo Pearl, Shell Pearl, Abalone Pearl
    }
    {
        field name: pearl_minimum_color
        question: What is the minimum color of the pearl?
    }
    {
        field name: number_of_pearls
        question: How many pearls are there?
    }
    {
        field: is_scale,
        question: Is there a scale in the image besides the jwellery item? 
        options: Yes, No
    }

    output format: give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
}
"""

prompt_dimensions_bracelet_amz = """
    You are given an image containing a bracelet placed next to a metric scale (calibrated in centimeters). Using the scale as a visual reference, identify the markings of the scale, the full scale in length (vertically) is 21 cm and horizontally is 15 cm, carefully identify and measure the following dimensions of one of the earrings (specifically, the one that appears most clearly in the front view):

    item_width: Measure the maximum horizontal extent (breadth, broadness, or wideness) of the entire earring in the front view, from its leftmost to its rightmost point, in centimeters (cm).
    item_height: Measure the maximum vertical extent of the entire earring in the front view, from its topmost to its bottommost point, in centimeters (cm).
    item_length: Measure the depth or thickness of the earring — i.e., the front-to-back extent that would be visible if the earring were viewed from the side. This is typically the least prominent dimension in a front-facing image, so use visible shadows, curves, or hooks as reference to estimate it. Provide this value in centimeters (cm).

    Output Format:  give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""

prompt_description_earrings_meesho = """
        you are jewellery expert,you will be given an image of a jewellery item.
        for the jewellery item Create a meesho friendly description with top ranking keywords as per marketplace trends with 10k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footers
"""

prompt_questions_earrings_meesho = """
    you are a jewellery expert and an e-commerce catalog manager for Meesho. you will be given an image of a jewellery item.
answer the given questions. If options are provided for a question, choose the correct one based on the image.
if no options are given, respond concisely using one or two words. do not write "not applicable".

Questions {
    {
        field name: Product Name
        question: What is the product name? Give a concise and appealing title.
    }
    {
        field name: Base Metal
        question: What is the base metal of the jewellery?
        options: Alloy, Bentex, Brass, Brass & Copper, Bronze, Copper, Fabric, Five Metal, German Silver, Glass, Meta, Plastic, Pure 925 Sterling Silver, Synthetic, Terracota, Thread, Wood
    }
    {
        field name: Color
        question: What is the primary color of the item?
        options: Beige, Black, Blue, Brown, Coral, Cream, Gold, Green, Grey, Grey Melange, Khaki, Lavendar, Maroon, Metallic, Mint Green, Multicolor, Mustard, Navy Blue, Nude, Olive, Orange, Peach, Pink, Purple, Red, Rose, Rust, Silver, Teal, White, Yellow
    }
    {
        field name: Occasion
        question: What is the main occasion this jewellery is suited for?
        options: Bridal, Ethnic, Ethnic Casual, Ethnic Party, Festive, Western Casual, Western Party
    }
    {
        field name: Plating
        question: What is the plating of the jewellery?
        options: Brass Plated, Copper Plated, Gold Plated, Gold Plated - Matte, Micro Plating, No Plating, Oxidised Gold, Oxidised Silver, Rhodium Plated, Rose Gold Plated, Silver Plated
    }
    {
        field name: Sizing
        question: What is the sizing type?
        options: Adjustable, Non-Adjustable
    }
    {
        field name: Stone Type
        question: What is the stone type used?
        options: Agate, Alexandrite, American Diamond, Amethyst, Artificial Beads, Artificial Stones, Artificial Stones & Beads, Citrine, Crystals, Cubic Zirconia, Cubic Zirconia/American Diamond, Emerald, Garnet, Kundan, Labradorite, Lapis, No Stone, Onyx, Pearls, Peridot, Polki, Quartz, Rhinestone, Rhodolite, Ruby, Rudrakshi, Sapphire, Tigers Eyes, Topaz, Tourmaline, Turquoise
    }
    {
        field name: Trend
        question: What is the trend of the jewellery?
        options: Afghan, Bharatnatyam Jewellery, Butterfly, Chain Linked, Claywork, Coin Work, Enamelled, Evil Eye, Feather, Floral, Galaxy, Geometric, Gold Filigree, Gota Patti, Guttapusalu, Hand-crafted, Jali Jewellery, Kempu Work, Korean, Kundan, LakshmiDevi, Layered, Mango, Mantasha, Meenakari, Minimal, Mirror, Peacock, Pom Pom, Radha Krishna, Ram Parivar, Religious, Shellwork, Tasselled, Temple, Threadwork, Thuriya, Tribal
    }
    {
        field name: Type
        question: What type of jewellery is this?
        options: Bahubali, Bugadi Earrings, Butta Bomma, Chandbalis, Chandelier, Drop Earrings, Ear Cuff, Ear Jackets, Earrings with Chain, Half Hoop Earrings, Hoop Earrings, Huggie Earrings, Jhumkhas, Kaanphool, Long, Maangtika and Earrings, Oversized Studs, Studs, Suidhaga, Tassel
    }
}

For output format = give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""


prompt_description_necklace_meesho = """
        you are jewellery expert,you will be given an image of a necklace.
        for the jewellery item Create a meesho friendly description with top ranking keywords as per marketplace trends with 10k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footers
"""

prompt_questions_necklace_meesho = """
You are a jewellery expert and an e-commerce catalog manager for Meesho. You will be given an image of a jewellery item.
Answer the given questions. If options are provided for a question, choose the correct one based on the image.
If no options are given, respond concisely using one or two words. Do not write "not applicable".

Questions {
    {
        field name: Product Name
        question: What is the product name? Give a concise and appealing title.
    }
    {
        field name: Base Metal
        question: What is the base metal of the jewellery?
        options: Alloy, Bentex, Brass, Brass & Copper, Bronze, Copper, Fabric, Five Metal, German Silver, Glass, Meta, Plastic, Pure 925 Sterling Silver, Synthetic, Terracota, Thread, Wood
    }
    {
        field name: Color
        question: What is the primary color of the item?
        options: Beige, Black, Blue, Brown, Coral, Cream, Gold, Green, Grey, Grey Melange, Khaki, Lavendar, Maroon, Metallic, Mint Green, Multicolor, Mustard, Navy Blue, Nude, Olive, Orange, Peach, Pink, Purple, Red, Rust, Silver, Teal, White, Yellow
    }
    {
        field name: Occasion
        question: What is the main occasion this jewellery is suited for?
        options: Bridal, Casual, Ethnic, Ethnic Casual, Ethnic Party, Festive, Western, Western Casual, Western Party
    }
    {
        field name: Plating
        question: What is the plating of the jewellery?
        options: 1Gram Gold, Brass Plated, Copper Plated, Gold Plated, Gold Plated - Matte, Micro Plating, No Plating, Oxidised Gold, Oxidised Silver, Rhodium Plated, Rose Gold Plated, Silver Plated
    }
    {
        field name: Sizing
        question: What is the sizing type?
        options: Adjustable, Bib, Choker, Long, Non-Adjustable, Short
    }
    {
        field name: Stone Type
        question: What is the stone type used?
        options: Agate, Alexandrite, American Diamond, Amethyst, Artificial Beads, Artificial Stones, Artificial Stones & Beads, Citrine, Crystals, Cubic Zirconia, Cubic Zirconia/American Diamond, Emerald, Garnet, Kundan, Labradorite, Lapis, No Stone, Onyx, Pearls, Peridot, Polki, Quartz, Rhinestone, Rhodolite, Ruby, Rudrakshi, Sapphire, Tigers Eyes, Topaz, Tourmaline, Turquoise
    }
    {
        field name: Trend
        question: What is the trend of the jewellery?
        options: Afghan, Alphabet, Attiga, Ball Chain, Bharatnatyam Jewellery, Butterfly, Chain Linked, Chapla Haar, Chinchpeti, Choker, Claywork, Coin Work, Dholbiri, Dollar, Enamelled, Evil Eye, Feather, Floral, Geometric, Gold Filigree, Golpata, Gota Patti, Guttapusalu, Hand-crafted, Jali Jewellery, Jappi, Junbiri, Kempu Work, Kolhapuri Saj, Korean, Kumbla Thali, Kundan, LakshmiDevi, Layered, Lokaparo, Mango, Mantasha, Meenakari, Minimal, Mirror, Mohan Mala, Mugappu, Mullamottu, Palakka Necklace, Peacock, Pearl Work, Pepa, Pola, Pom Pom, Radha Krishna, Rajputi, Ram Parivar, Religious, Shellwork, Swan, Tanmani, Tasselled, Temple, Templed, Thali Necklace, Threadwork, Tribal, Tulsi Necklace
    }
    {
        field name: Type
        question: What type of jewellery is this?
        options: Chain, Layered, Mopu Necklace, Necklace, Rani Haar, Satlada, Thushi
    }
}

    For output format = give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""

prompt_description_bracelet_meesho = """
        you are jewellery expert,you will be given an image of a bracelet.
        for the jewellery item Create a meesho friendly description with top ranking keywords as per marketplace trends with 10k max characters
        in the description include all the key features of the item. it should be in one paragraph.
        give only description dont add any headers or footers
"""

prompt_questions_bracelet_meesho = """
    You are a jewellery expert and an e-commerce catalog manager. You will be given an image of a jewellery item.
    Answer the given questions. If options are provided for a question, choose the correct one based on the image.
    If no options are given, respond concisely using one or two words. Do not write "not applicable".

Questions {
    {
        field name: Product Name
        question: What is the product name? Give a concise and appealing title.
    }
    {
        field name: Base Metal
        question: What is the base metal of the jewellery?
        options: Alloy, Bentex, Brass, Brass & Copper, Bronze, Copper, Fabric, Five Metal, German Silver, Glass, Meta, Plastic, Pure 925 Sterling Silver, Synthetic, Terracota, Thread, Wood
    }
    {
        field name: Closure
        question: What type of closure or fastening does this item use?
        options: Button, Foldover, Hook, Interlock, Lobster Clasp, Slip On, Tang
    }
    {
        field name: Color
        question: What is the primary color of the item?
        options: Beige, Black, Blue, Brown, Coral, Cream, Gold, Green, Grey, Grey Melange, Khaki, Lavendar, Maroon, Metallic, Mint Green, Multicolor, Mustard, Navy Blue, Nude, Olive, Orange, Peach, Pink, Purple, Red, Rust, Silver, Teal, White, Yellow
    }
    {
        field name: Occasion
        question: What is the main occasion this jewellery is suited for?
        options: Bridal, Casual, Ethnic, Ethnic Casual, Ethnic Party, Festive, Western, Western Casual, Western Party
    }
    {
        field name: Plating
        question: What is the plating of the jewellery?
        options: 1Gram Gold, Brass Plated, Copper Plated, Gold Plated, Gold Plated - Matte, Micro Plating, No Plating, Oxidised Gold, Oxidised Silver, Rhodium Plated, Rose Gold Plated, Silver Plated
    }
    {
        field name: Sizing
        question: What is the sizing type?
        options: Adjustable, Bib, Choker, Long, Non-Adjustable, Short
    }
    {
        field name: Stone Type
        question: What is the stone type used?
        options: Agate, Alexandrite, American Diamond, Amethyst, Artificial Beads, Artificial Stones, Artificial Stones & Beads, Citrine, Crystals, Cubic Zirconia, Cubic Zirconia/American Diamond, Emerald, Garnet, Kundan, Labradorite, Lapis, No Stone, Onyx, Pearls, Peridot, Polki, Quartz, Rhinestone, Rhodolite, Ruby, Rudrakshi, Sapphire, Tigers Eyes, Topaz, Tourmaline, Turquoise
    }
    {
        field name: Trend
        question: What is the trend of the jewellery?
        options: Afghan, Alphabet, Attiga, Ball Chain, Bharatnatyam Jewellery, Butterfly, Chain Linked, Chapla Haar, Chinchpeti, Choker, Claywork, Coin Work, Dholbiri, Dollar, Enamelled, Evil Eye, Feather, Floral, Geometric, Gold Filigree, Golpata, Gota Patti, Guttapusalu, Hand-crafted, Jali Jewellery, Jappi, Junbiri, Kempu Work, Kolhapuri Saj, Korean, Kumbla Thali, Kundan, LakshmiDevi, Layered, Lokaparo, Mango, Mantasha, Meenakari, Minimal, Mirror, Mohan Mala, Mugappu, Mullamottu, Palakka Necklace, Peacock, Pearl Work, Pepa, Pola, Pom Pom, Radha Krishna, Rajputi, Ram Parivar, Religious, Shellwork, Swan, Tanmani, Tasselled, Temple, Templed, Thali Necklace, Threadwork, Tribal, Tulsi Necklace
    }
    {
        field name: Type
        question: What type of jewellery is this?
        options: Bangle Set, Bangle Style, Bracelet, Charm, Chooda, Contemporary, Couple, Cuff, Danglers, Elasticated, Haathphool/ Haathpanja, Kada, Kangan, Link, Multistrand, Wraparound
    }
}

    For output format = give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.
"""



order_prompt = """
You are an expert document parser. Extract the following fields from this Ecommeerce platform (amazon,flipkart etc) invoice PDF. Return the output strictly in JSON format with no extra text.

Fields to extract:

order_id: The order number.

customer_name: Full name of the customer.

order_date: The date the order was placed.

order_time: The time the order was placed (if it is in AM/PM format then add AM/PM and if in 24hrs format add hrs).

shipping_address: Full shipping address (single line).

tracking_id: The tracking number for the shipment.

payment_method: The payment method used (e.g., UPI, Credit Card, prepaid , cod etc.).

marketplace: The marketplace where the order was placed dont add the domain like .in , .com etc (e.g., Amazon, Flipkart, etc.).

total_price_order: The total price of the order sum of all items prices.

product_price: The price of each item this is the total amount of each item dont give the unit price look for total amount or gross amount of each product.

item_type: A brief description of the item ordered. , can be only earrings , bracelet or necklace

quantity: Number of units ordered.

sku_id: Extract the number from product descriptions like "earrings6" (e.g., if it says "earrings6", sku_id is 6).

if there are multiple sku_ids then in the output then return a string for product price, sku id , item type , quantites like the following example dont add multiple entries for any other fields.

product_price : "100, 200, 300"
item_type : "earrings , earrings, earrings"
quantities : "1, 2, 3"
sku_ids : "6, 7, 8"

Format the output like this:
{
  "order_id": "...",
  "customer_name": "...",
  "order_date": "...",
  "order_time": "...",
  "shipping_address": "...",
  "tracking_id": "...",
  "payment_method": "...",
  "marketplace": "...",
  "total_price_order": "...",
  "product_price": "...",
  "item_types": "...",
  "quantities": ...,
  "sku_ids": ...
}
"""

variations_prompt = """
        You are given multiple product images that all represent the same jewelry item design but in different colors. These variations are intended to be listed under a common product group on an e-commerce catalog.

    Your task is to analyze the images and generate the following:

    1. field name: item_sku
    A concise, style-oriented identifier for this jewelry design that can be used to group all its color variants together. This name should be free of specific colors and usable as a parent style name (e.g., "Leaf-Shaped Stud Earrings" or "Floral Drop Dangle").

    2. field name: item_name 
    A complete, customer-facing title for **one** of the images (you may pick any one). Include style, color, and appeal details. This name should follow marketplace best practices (e.g., "Elegant Gold Tone Leaf-Shaped Stud Earrings for Women, Lightweight, Statement Jewelry for Gifting and Parties").

Use the visual information from the images to infer material tone, motif, and overall design cues.

Output in the following format: give output in json format with field name and answer as key value pairs,Return only valid raw JSON. Do not wrap in triple backticks or markdown.

"""