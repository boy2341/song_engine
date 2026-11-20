from pathlib import Path
import pandas as pd

from step1 import clean_lyrics, get_connection, init_schema, parse_artists

CSV_PATH = Path("songs.csv")
TARGET_SONGS = 10_000
MIN_YEAR = 2015
MAX_YEAR = 2025


def sanitize_text(val: object) -> str:
    """Repair valid surrogate pairs and strip lone UTF-16 surrogates for SQLite."""
    if val is None or pd.isna(val):
        return ""
    text = str(val)
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    except Exception:
        pass
    return text.encode("utf-8", "ignore").decode("utf-8").strip()


def build_10k_from_local_csv() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH.resolve()}")

    usecols = [
        "id", "name", "album_name", "artists", "danceability", "energy",
        "key", "mode", "acousticness", "valence", "tempo",
        "lyrics", "year", "genre", "popularity",
    ]

    print(f"Scanning {CSV_PATH} for the top {TARGET_SONGS:,} most popular songs ({MIN_YEAR}-{MAX_YEAR})...")
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(CSV_PATH, usecols=lambda c: c in usecols, chunksize=100_000):
        chunk = chunk.dropna(
            subset=["name", "artists", "lyrics", "tempo", "energy", "acousticness", "valence", "danceability", "year"]
        )
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[(chunk["year"] >= MIN_YEAR) & (chunk["year"] <= MAX_YEAR)]
        chunk = chunk[chunk["lyrics"].astype(str).str.len() >= 80]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        df = df.sort_values(by="popularity", ascending=False)

    # Case-insensitive deduplication before slicing a small buffer above 10k
    df["_norm_title"] = df["name"].astype(str).str.strip().str.lower()
    df["_norm_artist"] = df["artists"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["_norm_title", "_norm_artist"]).head(TARGET_SONGS + 1_500)

    conn = get_connection()
    init_schema(conn)

    all_artist_lists = [
        [sanitize_text(n) for n in parse_artists(a) if sanitize_text(n)]
        for a in df["artists"]
    ]
    unique_names = sorted({n for sub in all_artist_lists for n in sub})
    conn.executemany("INSERT OR IGNORE INTO creators (name) VALUES (?)", [(n,) for n in unique_names])
    conn.commit()
    creator_map = dict(conn.execute("SELECT name, creator_id FROM creators").fetchall())

    inserted = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        if inserted >= TARGET_SONGS:
            break

        lyrics = sanitize_text(clean_lyrics(row["lyrics"]))
        title = sanitize_text(row["name"])
        if len(lyrics) < 80 or not title:
            continue

        artist_list = all_artist_lists[idx] or ["Unknown Artist"]
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO songs (
                spotify_id, title, artist, album_name, genre, release_year, popularity,
                lyrics, tempo_bpm, energy, acousticness, valence, danceability, musical_key, musical_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sanitize_text(row.get("id")) or None,
                title,
                ", ".join(artist_list),
                sanitize_text(row.get("album_name")) or None,
                sanitize_text(row.get("genre")) or None,
                int(row["year"]),
                int(row["popularity"]) if "popularity" in row else 0,
                lyrics[:4000],
                float(row["tempo"]),
                float(row["energy"]),
                float(row["acousticness"]),
                float(row["valence"]),
                float(row["danceability"]),
                int(row["key"]) if pd.notna(row.get("key")) else None,
                int(row["mode"]) if pd.notna(row.get("mode")) else None,
            ),
        )
        if cur.rowcount == 0:
            continue

        song_id = cur.lastrowid
        conn.executemany(
            "INSERT OR IGNORE INTO song_credits (song_id, creator_id, role) VALUES (?, ?, 'performer')",
            [(song_id, creator_map[n]) for n in artist_list if n in creator_map],
        )
        inserted += 1

    conn.commit()
    min_y, max_y, min_pop, max_pop = conn.execute(
        "SELECT MIN(release_year), MAX(release_year), MIN(popularity), MAX(popularity) FROM songs"
    ).fetchone()
    conn.close()

    print(f"Done! Inserted {inserted:,} most popular tracks into music_engine.db.")
    print(f"Year range: {min_y}–{max_y} | Popularity score range: {min_pop}–{max_pop}")


if __name__ == "__main__":
    build_10k_from_local_csv()