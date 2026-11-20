import ast
import re
import sqlite3
from pathlib import Path 
import pandas as pd
CSV_PATH = Path("songs.csv")
DB_PATH = Path("music_engine.db")
MAX_SONGS: int | None = 80_000

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL ;")
    conn.execute("PRAGMA synchronus = NORMAL;")
    conn.execute("PRAGMA  foreign_keys=ON ;")
    return conn

def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS songs (
            song_id INTEGER PRIMARY KEY AUTOINCREMENT,
            spotify_id TEXT UNIQUE,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            album_name TEXT,
            genre TEXT,
            release_year INTEGER,
            popularity INTEGER DEFAULT 0,
            lyrics TEXT NOT NULL,
            tempo_bpm REAL NOT NULL,
            energy REAL NOT NULL,
            acousticness REAL NOT NULL,
            valence REAL NOT NULL,
            danceability REAL NOT NULL,
            musical_key INTEGER,
            musical_mode INTEGER,
            primary_vibe TEXT,
            secondary_vibe TEXT,
            lyrics_vec BLOB,
            vibe_vec BLOB
        );
        CREATE TABLE IF NOT EXISTS creators (
            creator_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS song_credits (
            song_id INTEGER NOT NULL,
            creator_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            PRIMARY KEY (song_id, creator_id, role),
            FOREIGN KEY (song_id) REFERENCES songs(song_id) ON DELETE CASCADE,
            FOREIGN KEY (creator_id) REFERENCES creators(creator_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title);
        CREATE INDEX IF NOT EXISTS idx_songs_popularity ON songs(popularity DESC);
        CREATE INDEX IF NOT EXISTS idx_credits_song ON song_credits(song_id);
        CREATE INDEX IF NOT EXISTS idx_credits_creator ON song_credits(creator_id);
        """
    )
    conn.commit()

def parse_artists(raw: object) -> list[str]:
    if pd.isna(raw):
        return ["Unknown Artist"]
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                names = [str(x).strip() for x in parsed if str(x).strip()]
                return names or ["Unknown Artist"]
        except (ValueError, SyntaxError):
            pass
    names = [x.strip() for x in re.split(r"[;,]", text) if x.strip()]
    return names or ["Unknown Artist"]

def clean_lyrics(raw:object)-> str:
    if pd.isna(raw):
        return ""
    text = str(raw)
    text = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text
def load_550k_dataset() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"could not find")
    usecols = [
        "id", "name", "album_name", "artists", "danceability", "energy",
        "key", "mode", "acousticness", "valence", "tempo",
        "lyrics", "year", "genre", "popularity"
    ]
    print("Reading and filtering songs.csv in chunks...")
    filtered_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(CSV_PATH, usecols=lambda c: c in usecols, chunksize=50_000):
        chunk = chunk.dropna(
            subset=["name", "artists", "lyrics", "tempo", "energy", "acousticness", "valence", "danceability"]
        )
        chunk = chunk[chunk["lyrics"].astype(str).str.len() >= 80]
        filtered_chunks.append(chunk)
    df = pd.concat(filtered_chunks, ignore_index=True)
    if "popularity" in df.columns:
        df = df.sort_values(by="popularity", ascending=False)
    df = df.drop_duplicates(subset=["name", "artists"])
    if MAX_SONGS is not None:
        df = df.head(MAX_SONGS)
    conn = get_connection()
    init_schema(conn)
    # Pre-insert unique creators so credit linking is fast
    all_artist_lists = [parse_artists(a) for a in df["artists"]]
    unique_names = sorted({name for sublist in all_artist_lists for name in sublist})
    conn.executemany("INSERT OR IGNORE INTO creators (name) VALUES (?)", [(n,) for n in unique_names])
    conn.commit()
    creator_map = dict(conn.execute("SELECT name, creator_id FROM creators").fetchall())
    inserted = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        lyrics = clean_lyrics(row["lyrics"])
        if len(lyrics) < 80:
            continue
        artist_list = all_artist_lists[idx]
        primary_artist_str = ", ".join(artist_list)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO songs (
                spotify_id, title, artist, album_name, genre, release_year, popularity,
                lyrics, tempo_bpm, energy, acousticness, valence, danceability, musical_key, musical_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["id"]).strip() if pd.notna(row.get("id")) else None,
                str(row["name"]).strip(),
                primary_artist_str,
                str(row["album_name"]).strip() if pd.notna(row.get("album_name")) else None,
                str(row["genre"]).strip() if pd.notna(row.get("genre")) else None,
                int(row["year"]) if pd.notna(row.get("year")) else None,
                int(row["popularity"]) if pd.notna(row.get("popularity")) else 0,
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
        credit_rows = [
            (song_id, creator_map[name], "performer")
            for name in artist_list
            if name in creator_map
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO song_credits (song_id, creator_id, role) VALUES (?, ?, ?)",
            credit_rows,
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"Done! Stored {inserted:,} songs and {len(creator_map):,} creators in {DB_PATH}.")


if __name__ == "__main__":
    load_550k_dataset()