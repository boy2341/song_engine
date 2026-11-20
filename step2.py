import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from step1 import get_connection
# High accuracy (768-dim). If running on CPU and you want 4x faster speed,
# you can swap this to "all-MiniLM-L6-v2" (384-dim).
MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 128
VIBE_ANCHORS: dict[str, str] = {
    "bittersweet nostalgia": "memories of past love, old photographs, rain, longing for youth and days gone by",
    "quiet heartbreak": "grief, distance, empty rooms, loneliness, tears, unspoken goodbyes",
    "euphoric celebration": "joy, dancing, golden sunlight, laughter, warmth, carefree party energy",
    "restless night drive": "midnight highways, neon city lights, fast heartbeat, urban escape",
    "calm acoustic warmth": "peaceful mornings, gentle guitar, stillness, comfort, quiet reflection",
    "defiant intensity": "rebellion, breaking free, storm clouds, anger, raw determination, fierce power",
    "romantic devotion": "deep love, holding close, forever promises, tenderness, soulmates",
}
def embed_all_songs(batch_size: int = BATCH_SIZE) -> None:
    conn = get_connection()
    rows = conn.execute(
        "SELECT song_id, lyrics FROM songs WHERE lyrics_vec IS NULL ORDER BY popularity DESC"
    ).fetchall()
    total = len(rows)
    if total == 0:
        print("All songs already have local embeddings.")
        conn.close()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{MODEL_NAME}' on [{device.upper()}] to embed {total:,} songs...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    anchor_labels = list(VIBE_ANCHORS.keys())
    anchor_vecs = model.encode(
        list(VIBE_ANCHORS.values()),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    t0 = time.perf_counter()
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        lyrics_batch = [r[1] for r in batch]
        lyric_vecs = model.encode(
            lyrics_batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        updates = []
        for i, (song_id, _) in enumerate(batch):
            l_vec = lyric_vecs[i]
            sims = anchor_vecs @ l_vec
            top_idx = np.argsort(sims)[::-1]
            primary_vibe = anchor_labels[top_idx[0]]
            secondary_vibe = anchor_labels[top_idx[1]]
            v_vec = (0.7 * anchor_vecs[top_idx[0]] + 0.3 * anchor_vecs[top_idx[1]]).astype(np.float32)
            v_norm = np.linalg.norm(v_vec)
            if v_norm > 0:
                v_vec /= v_norm
            updates.append(
                (primary_vibe, secondary_vibe, l_vec.tobytes(), v_vec.tobytes(), song_id)
            )
        conn.executemany(
            """
            UPDATE songs
            SET primary_vibe = ?, secondary_vibe = ?, lyrics_vec = ?, vibe_vec = ?
            WHERE song_id = ?
            """,
            updates,
        )
        conn.commit()
        done = min(start + batch_size, total)
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = ((total - done) / rate / 60) if rate > 0 else 0
        print(f"Embedded {done:,}/{total:,} ({rate:.1f} songs/sec | ETA: {eta_min:.1f} min)")
    conn.close()
    print("Finished generating all embeddings.")
if __name__ == "__main__":
    embed_all_songs()