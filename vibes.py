import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DB_PATH = "music_engine.db"

# 16 distinct mood categories, each defined by multiple descriptive variants
VIBE_ANCHOR_VARIANTS: dict[str, list[str]] = {
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


def detect_db_model(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT LENGTH(lyrics_vec) FROM songs WHERE lyrics_vec IS NOT NULL LIMIT 1"
    ).fetchone()
    if row and row[0]:
        dim = row[0] // 4
        if dim == 768:
            return "BAAI/bge-base-en-v1.5"
        if dim == 384:
            return "all-MiniLM-L6-v2"
    return "BAAI/bge-base-en-v1.5"


def build_anchor_centroids(model: SentenceTransformer) -> tuple[list[str], np.ndarray]:
    """Encode multiple phrasing variants per mood and average them into unit centroids."""
    labels = list(VIBE_ANCHOR_VARIANTS.keys())
    centroids = []
    for label in labels:
        variants = VIBE_ANCHOR_VARIANTS[label]
        vecs = model.encode(
            variants, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        centroids.append(mean_vec)
    return labels, np.vstack(centroids).astype(np.float32)


def recompute_all_vibes() -> None:
    conn = sqlite3.connect(DB_PATH)
    model_name = detect_db_model(conn)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device.upper()} to encode {len(VIBE_ANCHOR_VARIANTS)} mood centroids...")
    model = SentenceTransformer(model_name, device=device)

    labels, anchor_vecs = build_anchor_centroids(model)
    expected_bytes = anchor_vecs.shape[1] * 4

    rows = conn.execute(
        "SELECT song_id, lyrics_vec FROM songs WHERE lyrics_vec IS NOT NULL"
    ).fetchall()

    valid_ids = []
    lyric_vecs = []
    for song_id, blob in rows:
        if blob and len(blob) == expected_bytes:
            valid_ids.append(song_id)
            lyric_vecs.append(np.frombuffer(blob, dtype=np.float32))

    if not lyric_vecs:
        print("No valid lyrics_vec entries found.")
        conn.close()
        return

    L = np.vstack(lyric_vecs)  # (N, D)
    sims = L @ anchor_vecs.T   # (N, 16)

    top2_idx = np.argsort(sims, axis=1)[:, ::-1][:, :2]
    top1 = top2_idx[:, 0]
    top2 = top2_idx[:, 1]

    V = 0.7 * anchor_vecs[top1] + 0.3 * anchor_vecs[top2]
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = (V / np.maximum(norms, 1e-12)).astype(np.float32)

    updates = [
        (labels[int(i1)], labels[int(i2)], V[row_i].tobytes(), int(sid))
        for row_i, (sid, i1, i2) in enumerate(zip(valid_ids, top1, top2))
    ]

    conn.executemany(
        """
        UPDATE songs
        SET primary_vibe = ?, secondary_vibe = ?, vibe_vec = ?
        WHERE song_id = ?
        """,
        updates,
    )
    conn.commit()
    conn.close()
    print(f"Updated vibe vectors across all {len(updates):,} songs in {DB_PATH}!")


if __name__ == "__main__":
    recompute_all_vibes()