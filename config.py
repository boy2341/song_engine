DB_PATH = "music_engine.db"

# Edit or add moods here (each mood averages multiple descriptive phrases into one centroid)
VIBE_ANCHORS: dict[str, list[str]] = {
    "energetic party hype": [
        "upbeat dance club party anthem high energy celebration jumping crowd",
        "wild night out loud bass adrenaline rush dancing till morning",
    ],
    "happy sunny feelgood": [
        "joyful carefree sunshine smiling warm summer day pure happiness",
        "lighthearted cheerful optimism bright morning good mood",
    ],
    "romantic dreamy love": [
        "deep romantic devotion tender kiss soul connection falling in love",
        "passionate intimacy sweet affection forever together",
    ],
    "ethereal hazy longing": [
        "dreamy slow motion hypnotic haze soft whisper floating midnight yearning",
        "intoxicating atmospheric reverie gentle ache quiet obsession",
    ],
    "sad heartbreak grief": [
        "devastating heartbreak crying alone shattered trust painful goodbye",
        "heavy sorrow mourning loss tears falling empty room",
    ],
    "bittersweet nostalgia": [
        "fading polaroid memories looking back at youth wistful autumn afternoon",
        "missing the past growing older bittersweet reflection on who we were",
    ],
    "calm acoustic warmth": [
        "peaceful gentle acoustic guitar quiet comfort safe sanctuary",
        "soft lullaby serene stillness warm coffee rainy window",
    ],
    "melancholic introspection": [
        "quiet loneliness overthinking at 3am numb detachment existential doubt",
        "fragile confession feeling invisible searching for meaning",
    ],
    "dark gothic brooding": [
        "haunting shadows gothic tragedy cursed fate eerie nocturnal mystery",
        "cold sorrowful cathedral organ velvet darkness tragic beauty",
    ],
    "epic symphonic grandeur": [
        "soaring operatic drama cinematic battle destiny triumph over darkness",
        "mythic storm rising phoenix powerful orchestral intensity",
    ],
    "raw angst rebellion": [
        "bitter anger screaming defiance distorted guitar refusing to conform",
        "visceral frustration betrayal raw grunge fury breaking free",
    ],
    "empowering triumphant rise": [
        "unstoppable confidence rising from the ashes fearless strength victory",
        "standing tall reclaiming power bold resilience",
    ],
    "sensual nocturnal allure": [
        "sultry midnight temptation velvet tension magnetic attraction",
        "late night mystery slow burn desire seductive rhythm",
    ],
    "playful witty irreverent": [
        "cheeky sarcastic humor clever banter carefree swagger",
        "teasing smirk ironic storytelling fun upbeat attitude",
    ],
    "cinematic wanderlust": [
        "open highway driving into the horizon freedom chasing distant lights",
        "restless spirit traveling across oceans wide open sky",
    ],
    "spiritual soulful awe": [
        "transcendent wonder sacred devotion surrender to the universe",
        "deep soulful gratitude inner awakening emotional catharsis",
    ],
}

# Add any artists here to pull their catalog + similar Artist Radio tracks
SEED_ARTISTS: list[str] = [
    "Within Temptation",
    "Lacuna Coil",
    "Nightwish",
    "Halestorm",
    "Flyleaf",
    "Delain",
    "Breaking Benjamin",
    "We Are The Fallen",
    "Epica",
    "After Forever",
    "Mazzy Star",
    "Cigarettes After Sex",
    "Ethel Cain",
    "Weyes Blood",
    "Florence + The Machine",
    "Angel Olsen",
    "Phoebe Bridgers",
    "Hole",
    "Twenty One Pilots",
    "Hayley Williams",
    "BoyWithUke",
]

# Format: (Playlist Search Query, Genre Tag, Storefront Country Code)
GENRE_PLAYLISTS: list[tuple[str, str, str]] = [
    ("Top Pop Hits 2023", "pop", "us"),
    ("Best Pop Songs 2024", "pop", "us"),
    ("Top Pop Hits 2025", "pop", "us"),
    ("Top Hindi Songs 2023", "hindi", "in"),
    ("Top Hindi Songs 2024", "hindi", "in"),
    ("Top Hindi Songs 2025", "hindi", "in"),
    ("Hindi Indie 2024", "hindi indie", "in"),
    ("Best Rock Songs 2023", "rock", "us"),
    ("Top Rock Hits 2024", "rock", "us"),
    ("Best Rock Songs 2025", "rock", "us"),
    ("Best Metal Songs 2023", "metal", "us"),
    ("Top Metal Hits 2024", "metal", "us"),
    ("Best Metal Songs 2025", "metal", "us"),
    ("Best Indie Songs 2023", "indie", "us"),
    ("Top Indie Hits 2024", "indie", "us"),
    ("Best Indie Songs 2025", "indie", "us"),
]