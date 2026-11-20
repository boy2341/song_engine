import argparse
import os
import re
import sqlite3
import tempfile
import time
import urllib.parse
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

from config import DB_PATH, GENRE_PLAYLISTS, SEED_ARTISTS, VIBE_ANCHORS

HEADERS = {"User-Agent": "SongFinderEngine/1.0 (open-source-music-recommender)"}
SPOTIFY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{22}$")
SKIP_KEYWORDS = ("instrumental", "karaoke", "commentary", "acapella", "sped up", "slowed + reverb")


def sanitize_text(val: object) -> str:
    if val is None:
        return ""
    text = str(val)
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    except Exception:
        pass
    return text.encode("utf-8", "ignore").decode("utf-8").strip()


def clean_lyrics(raw: str) -> str:
    text = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", " ", raw or "")
    return sanitize_text(re.sub(r"\s+", " ", text))


def strip_title_extras(title: str) -> str:
    cleaned = re.sub(r"\s*[\(\[].*?[\)\]]", "", title)
    cleaned = re.sub(
        r"\s+-\s+(?:remaster|live|radio edit|acoustic|demo|version|edit).*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def normalize_artist(name: str) -> str:
    text = sanitize_text(name).lower().replace("&", "and").replace("+", "and")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text)).strip()


def split_artists(raw: str) -> list[str]:
    parts = re.split(r"\s*(?:,|\bfeat\.?\b|\bft\.?\b|\bx\b)\s*", sanitize_text(raw), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def make_spotify_url(spotify_id: str | None, title: str, artist: str) -> str:
    clean_id = (spotify_id or "").strip()
    if SPOTIFY_ID_PATTERN.fullmatch(clean_id):
        return f"https://open.spotify.com/track/{clean_id}"
    return f"https://open.spotify.com/search/{urllib.parse.quote(f'{title} {artist}'.strip())}"


def detect_model_name(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT LENGTH(lyrics_vec) FROM songs WHERE lyrics_vec IS NOT NULL LIMIT 1").fetchone()
    if row and row[0] == 384 * 4:
        return "all-MiniLM-L6-v2"
    return "BAAI/bge-base-en-v1.5"


def load_embedder(conn: sqlite3.Connection) -> SentenceTransformer:
    model_name = detect_model_name(conn)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device.upper()}...")
    return SentenceTransformer(model_name, device=device)


def build_anchor_centroids(model: SentenceTransformer) -> tuple[list[str], np.ndarray]:
    labels = list(VIBE_ANCHORS.keys())
    centroids = []
    for label in labels:
        vecs = model.encode(VIBE_ANCHORS[label], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        centroids.append(mean_vec / norm if norm > 0 else mean_vec)
    return labels, np.vstack(centroids).astype(np.float32)


def ensure_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    if "spotify_url" not in cols:
        conn.execute("ALTER TABLE songs ADD COLUMN spotify_url TEXT;")
        conn.commit()


def fetch_lyrics(title: str, artist: str) -> str:
    primary_artist = split_artists(artist)[0] if split_artists(artist) else artist
    for t_var in dict.fromkeys([title, strip_title_extras(title)]):
        if not t_var:
            continue
        try:
            r = requests.get(
                "https://lrclib.net/api/get",
                params={"track_name": t_var, "artist_name": primary_artist},
                headers=HEADERS,
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json()
                txt = clean_lyrics(data.get("plainLyrics") or data.get("syncedLyrics") or "")
                if len(txt) >= 80:
                    return txt
        except Exception:
            pass

    try:
        r = requests.get(
            "https://lrclib.net/api/search",
            params={"q": f"{strip_title_extras(title)} {primary_artist}"},
            headers=HEADERS,
            timeout=8,
        )
        for item in r.json()[:3]:
            txt = clean_lyrics(item.get("plainLyrics") or item.get("syncedLyrics") or "")
            if len(txt) >= 80:
                return txt
    except Exception:
        pass
    return ""


def fetch_metadata(title: str, artist: str, country: str = "us") -> dict:
    storefronts = [country] if country == "us" else [country, "us"]
    base_title = strip_title_extras(title)
    for sf in storefronts:
        try:
            r = requests.get(
                "https://itunes.apple.com/search",
                params={"term": f"{base_title} {artist}", "entity": "song", "country": sf, "limit": 1},
                headers=HEADERS,
                timeout=8,
            )
            hits = r.json().get("results", [])
            if hits:
                h = hits[0]
                rd = h.get("releaseDate", "")
                yr = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
                return {
                    "year": yr,
                    "genre": sanitize_text(h.get("primaryGenreName", "alternative")).lower() or "alternative",
                    "preview_url": h.get("previewUrl"),
                }
        except Exception:
            continue
    return {}


def extract_audio(preview_url: str | None) -> tuple[float, float, float, float, float]:
    if preview_url:
        tmp_path = None
        try:
            import librosa

            raw = requests.get(preview_url, headers=HEADERS, timeout=10).content
            tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
            tmp_path = tmp.name
            tmp.write(raw)
            tmp.close()

            y, sr = librosa.load(tmp_path, sr=22050, duration=25.0)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(np.clip(float(np.atleast_1d(tempo)[0]), 60.0, 200.0))
            energy = float(np.clip(float(np.mean(librosa.feature.rms(y=y))) * 4.5, 0.05, 0.98))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            acousticness = float(np.clip(1.0 - (flatness * 18.0), 0.02, 0.98))
            danceability = float(np.clip(float(np.mean(librosa.onset.onset_strength(y=y, sr=sr))) / 3.5, 0.15, 0.95))
            return bpm, energy, acousticness, 0.55, danceability
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return 116.0, 0.70, 0.26, 0.55, 0.62


def add_candidates(candidates: list[dict], year_range: tuple[int, int] | None = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    existing = {
        (strip_title_extras(t).lower(), normalize_artist(a))
        for t, a in conn.execute("SELECT title, artist FROM songs").fetchall()
    }
    to_check = [
        c for c in candidates
        if (strip_title_extras(c["title"]).lower(), normalize_artist(c["artist"])) not in existing
    ]
    print(f"Found {len(to_check):,} new candidate tracks to check.")
    if not to_check:
        conn.close()
        return

    model = load_embedder(conn)
    labels, anchor_vecs = build_anchor_centroids(model)
    added = 0

    for idx, item in enumerate(to_check, start=1):
        title, artist = item["title"], item["artist"]
        norm_key = (strip_title_extras(title).lower(), normalize_artist(artist))
        if norm_key in existing:
            continue

        meta = item.get("meta") or fetch_metadata(title, artist, item.get("country", "us"))
        year = meta.get("year")
        if year_range and (year is None or not (year_range[0] <= year <= year_range[1])):
            continue

        lyrics = fetch_lyrics(title, artist)
        if not lyrics:
            continue

        genre = item.get("genre_hint") or meta.get("genre", "alternative")
        if meta.get("genre") and meta["genre"] not in {"music", "alternative", "bollywood", "indian"}:
            genre = meta["genre"]

        l_vec = model.encode([lyrics[:4000]], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        sims = anchor_vecs @ l_vec
        top2 = np.argsort(sims)[::-1][:2]
        v_vec = (0.7 * anchor_vecs[top2[0]] + 0.3 * anchor_vecs[top2[1]]).astype(np.float32)
        v_vec /= max(float(np.linalg.norm(v_vec)), 1e-12)

        bpm, energy, acousticness, valence, danceability = extract_audio(meta.get("preview_url"))
        sp_url = make_spotify_url(None, title, artist)

        cur = conn.execute(
            """
            INSERT OR IGNORE INTO songs (
                spotify_id, spotify_url, title, artist, album_name, genre, release_year, popularity,
                lyrics, tempo_bpm, energy, acousticness, valence, danceability,
                musical_key, musical_mode, primary_vibe, secondary_vibe, lyrics_vec, vibe_vec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"custom_{abs(hash((title.lower(), artist.lower())))}",
                sp_url,
                title,
                artist,
                None,
                genre,
                year,
                82,
                lyrics[:200],
                bpm,
                energy,
                acousticness,
                valence,
                danceability,
                None,
                1,
                labels[int(top2[0])],
                labels[int(top2[1])],
                l_vec.tobytes(),
                v_vec.tobytes(),
            ),
        )
        if cur.rowcount == 0:
            continue

        song_id = cur.lastrowid
        for name in split_artists(artist):
            conn.execute("INSERT OR IGNORE INTO creators (name) VALUES (?)", (name,))
            cid = conn.execute("SELECT creator_id FROM creators WHERE name = ?", (name,)).fetchone()[0]
            conn.execute(
                "INSERT OR IGNORE INTO song_credits (song_id, creator_id, role) VALUES (?, ?, 'performer')",
                (song_id, cid),
            )

        existing.add(norm_key)
        added += 1
        if added % 15 == 0:
            conn.commit()
        print(f"[{idx}/{len(to_check)}] + {title} — {artist} ({year or 'N/A'} | {genre})")
        time.sleep(0.18)

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    conn.close()
    print(f"\nAdded {added:,} new songs! Total library size: {total:,}.")


def harvest_artists(artists: list[str], include_radio: bool = True) -> None:
    tracks: dict[tuple[str, str], dict] = {}
    for artist in artists:
        target = normalize_artist(artist)
        print(f"Scanning artist & radio: {artist}...")
        try:
            r = requests.get("https://api.deezer.com/search/artist", params={"q": artist, "limit": 5}, headers=HEADERS, timeout=10)
            hits = r.json().get("data", [])
            aid = next((h["id"] for h in hits if normalize_artist(h.get("name", "")) == target), hits[0]["id"] if hits else None)
            if aid:
                endpoints = [f"https://api.deezer.com/artist/{aid}/top?limit=100"]
                if include_radio:
                    endpoints.append(f"https://api.deezer.com/artist/{aid}/radio?limit=50")
                for url in endpoints:
                    tr_list = requests.get(url, headers=HEADERS, timeout=10).json().get("data", [])
                    for tr in tr_list:
                        t = sanitize_text(tr.get("title_short") or tr.get("title"))
                        a = sanitize_text(tr.get("artist", {}).get("name")) or artist
                        if t and not any(k in t.lower() for k in SKIP_KEYWORDS):
                            tracks.setdefault((strip_title_extras(t).lower(), normalize_artist(a)), {"title": t, "artist": a})
        except Exception:
            pass

        try:
            it = requests.get("https://itunes.apple.com/search", params={"term": artist, "entity": "song", "limit": 100}, headers=HEADERS, timeout=10).json().get("results", [])
            for h in it:
                ha = sanitize_text(h.get("artistName"))
                if normalize_artist(ha) != target:
                    continue
                t = sanitize_text(h.get("trackName"))
                if t and not any(k in t.lower() for k in SKIP_KEYWORDS):
                    rd = h.get("releaseDate", "")
                    meta = {
                        "year": int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None,
                        "genre": sanitize_text(h.get("primaryGenreName", "rock")).lower(),
                        "preview_url": h.get("previewUrl"),
                    }
                    tracks.setdefault((strip_title_extras(t).lower(), target), {"title": t, "artist": ha, "meta": meta})
        except Exception:
            pass
        time.sleep(0.15)

    add_candidates(list(tracks.values()))


def harvest_playlists(playlists: list[tuple[str, str, str]], year_range: tuple[int, int] | None = (2023, 2025)) -> None:
    tracks: dict[tuple[str, str], dict] = {}
    for query, genre_tag, country in playlists:
        print(f"Scanning playlist query: {query}...")
        try:
            pls = requests.get("https://api.deezer.com/search/playlist", params={"q": query, "limit": 3}, headers=HEADERS, timeout=10).json().get("data", [])
            for pl in pls:
                if not pl.get("tracklist"):
                    continue
                for tr in requests.get(pl["tracklist"], params={"limit": 80}, headers=HEADERS, timeout=10).json().get("data", []):
                    t = sanitize_text(tr.get("title_short") or tr.get("title"))
                    a = sanitize_text(tr.get("artist", {}).get("name"))
                    if t and a and not any(k in t.lower() for k in SKIP_KEYWORDS):
                        tracks.setdefault(
                            (strip_title_extras(t).lower(), normalize_artist(a)),
                            {"title": t, "artist": a, "genre_hint": genre_tag, "country": country},
                        )
        except Exception:
            pass
    add_candidates(list(tracks.values()), year_range=year_range)


def recompute_all_vibes() -> None:
    conn = sqlite3.connect(DB_PATH)
    model = load_embedder(conn)
    labels, anchor_vecs = build_anchor_centroids(model)
    expected_bytes = anchor_vecs.shape[1] * 4

    rows = conn.execute("SELECT song_id, lyrics_vec FROM songs WHERE lyrics_vec IS NOT NULL").fetchall()
    valid_ids, vecs = [], []
    for sid, blob in rows:
        if blob and len(blob) == expected_bytes:
            valid_ids.append(sid)
            vecs.append(np.frombuffer(blob, dtype=np.float32))

    if not vecs:
        conn.close()
        return

    L = np.vstack(vecs)
    sims = L @ anchor_vecs.T
    top2 = np.argsort(sims, axis=1)[:, ::-1][:, :2]
    V = 0.7 * anchor_vecs[top2[:, 0]] + 0.3 * anchor_vecs[top2[:, 1]]
    V = (V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    conn.executemany(
        "UPDATE songs SET primary_vibe = ?, secondary_vibe = ?, vibe_vec = ? WHERE song_id = ?",
        [(labels[int(i1)], labels[int(i2)], V[r].tobytes(), int(sid)) for r, (sid, i1, i2) in enumerate(zip(valid_ids, top2[:, 0], top2[:, 1]))],
    )
    conn.commit()
    conn.close()
    print(f"Recomputed {len(labels)} vibe centroids across {len(valid_ids):,} songs!")


def compact_database() -> None:
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    conn.execute(
        """
        DELETE FROM songs WHERE song_id NOT IN (
            SELECT MIN(song_id) FROM songs GROUP BY LOWER(TRIM(title)), LOWER(TRIM(artist))
        )
        """
    )
    conn.execute("UPDATE songs SET lyrics = SUBSTR(lyrics, 1, 80) WHERE LENGTH(lyrics) > 80")
    rows = conn.execute("SELECT song_id, spotify_id, title, artist FROM songs WHERE spotify_url IS NULL OR spotify_url = ''").fetchall()
    if rows:
        conn.executemany(
            "UPDATE songs SET spotify_url = ? WHERE song_id = ?",
            [(make_spotify_url(sp_id, t, a), sid) for sid, sp_id, t, a in rows],
        )
    conn.commit()
    conn.execute("VACUUM;")
    count = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    conn.close()
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Database compacted: {size_mb:.2f} MB across {count:,} songs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage your Smart Song Finder music library.")
    parser.add_argument("--artists", nargs="*", help="Add specific artists (or omit names to use SEED_ARTISTS in config.py)")
    parser.add_argument("--playlists", action="store_true", help="Import tracks from GENRE_PLAYLISTS in config.py")
    parser.add_argument("--recompute-vibes", action="store_true", help="Recompute vibe vectors from VIBE_ANCHORS in config.py")
    parser.add_argument("--compact", action="store_true", help="Deduplicate, refresh Spotify links, and VACUUM database")
    args = parser.parse_args()

    if args.artists is not None:
        harvest_artists(args.artists if args.artists else SEED_ARTISTS, include_radio=True)
    elif args.playlists:
        harvest_playlists(GENRE_PLAYLISTS)
    elif args.recompute_vibes:
        recompute_all_vibes()
    elif args.compact:
        compact_database()
    else:
        print("\n=== Smart Song Finder — Library Manager ===")
        print("1. Add Artists + Radio (from config.py or custom input)")
        print("2. Add Genre / Year Playlists (from config.py)")
        print("3. Recompute Vibes (after editing VIBE_ANCHORS in config.py)")
        print("4. Compact Database & Refresh Spotify Links")
        choice = input("\nChoose an option (1-4): ").strip()

        if choice == "1":
            raw = input("Enter comma-separated artists (or press Enter for SEED_ARTISTS in config.py): ").strip()
            target_artists = [a.strip() for a in raw.split(",") if a.strip()] if raw else SEED_ARTISTS
            harvest_artists(target_artists, include_radio=True)
        elif choice == "2":
            harvest_playlists(GENRE_PLAYLISTS)
        elif choice == "3":
            recompute_all_vibes()
        elif choice == "4":
            compact_database()


if __name__ == "__main__":
    main()