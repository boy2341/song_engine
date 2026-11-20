import os
import re
import sqlite3
import tempfile
import time
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

DB_PATH = "music_engine.db"
HEADERS = {"User-Agent": "SongFinderEngine/1.0 (local-music-recommender)"}

# Set to True to also include similar tracks from each artist's Deezer Radio mix
INCLUDE_SIMILAR_RADIO_TRACKS = True

ARTISTS = [
    "Lana Del Rey"
]

VIBE_ANCHORS = {
    "energetic party hype": "upbeat party dance club energetic fun celebration hype",
    "sad heartbreak longing": "sad heartbreak crying miss you lonely pain tears",
    "calm acoustic warmth": "calm peaceful acoustic warm gentle soft love comfort",
    "dark moody intense": "dark moody intense shadow anger revenge midnight tension",
    "happy sunny feelgood": "happy sunshine smile joyful carefree bright summer",
    "romantic dreamy love": "romantic dreamy soul forever kiss passion devotion",
}

SKIP_TITLE_KEYWORDS = (
    "instrumental",
    "karaoke",
    "commentary",
    "acapella",
    "a cappella",
    "sped up",
    "slowed + reverb",
)


def sanitize_text(val: object) -> str:
    """Repair valid emoji pairs and strip broken UTF-16 surrogates for SQLite."""
    if val is None:
        return ""
    text = str(val)
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    except Exception:
        pass
    return text.encode("utf-8", "ignore").decode("utf-8").strip()


def clean_lyrics_text(raw: str) -> str:
    text = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", " ", raw or "")
    text = re.sub(r"\s+", " ", text)
    return sanitize_text(text)


def strip_title_extras(title: str) -> str:
    """Remove suffixes like '(Remastered)', '[Live]', or '- Radio Edit' for clean matching."""
    cleaned = re.sub(r"\s*[\(\[].*?[\)\]]", "", title)
    cleaned = re.sub(
        r"\s+-\s+(?:remaster|live|radio edit|acoustic|demo|version|edit).*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def normalize_artist_key(name: str) -> str:
    """Normalize '&', '+', and 'and' so 'Florence + The Machine' matches reliably."""
    text = sanitize_text(name).lower()
    text = text.replace("&", "and").replace("+", "and")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_artist_names(raw: str) -> list[str]:
    text = sanitize_text(raw)
    if not text:
        return []
    # Keep band names containing '+' intact while splitting featured artists
    parts = re.split(r"\s*(?:,|\bfeat\.?\b|\bft\.?\b|\bx\b)\s*", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


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


def find_deezer_artist_id(artist_name: str) -> int | None:
    target_key = normalize_artist_key(artist_name)
    try:
        resp = requests.get(
            "https://api.deezer.com/search/artist",
            params={"q": artist_name, "limit": 5},
            headers=HEADERS,
            timeout=10,
        )
        artists = resp.json().get("data", [])
        for item in artists:
            if normalize_artist_key(item.get("name", "")) == target_key:
                return item.get("id")
        if artists:
            return artists[0].get("id")
    except Exception:
        pass
    return None


def collect_all_artist_tracks() -> list[dict]:
    """Gather catalog tracks for all 21 artists first, followed by Artist Radio mixes."""
    primary_tracks: dict[tuple[str, str], dict] = {}
    radio_tracks: dict[tuple[str, str], dict] = {}

    for artist_name in ARTISTS:
        target_key = normalize_artist_key(artist_name)
        print(f"Scanning catalog for: {artist_name}...")

        # 1. Deezer Top / Catalog Tracks + Artist Radio
        artist_id = find_deezer_artist_id(artist_name)
        if artist_id:
            try:
                top_resp = requests.get(
                    f"https://api.deezer.com/artist/{artist_id}/top",
                    params={"limit": 100},
                    headers=HEADERS,
                    timeout=10,
                )
                for tr in top_resp.json().get("data", []):
                    t = sanitize_text(tr.get("title_short") or tr.get("title"))
                    a = sanitize_text(tr.get("artist", {}).get("name")) or artist_name
                    if not t or any(k in t.lower() for k in SKIP_TITLE_KEYWORDS):
                        continue
                    key = (strip_title_extras(t).lower(), normalize_artist_key(a))
                    if key[0] and key not in primary_tracks:
                        primary_tracks[key] = {"title": t, "artist": a, "meta": {}}
            except Exception:
                pass

            if INCLUDE_SIMILAR_RADIO_TRACKS:
                try:
                    rad_resp = requests.get(
                        f"https://api.deezer.com/artist/{artist_id}/radio",
                        params={"limit": 50},
                        headers=HEADERS,
                        timeout=10,
                    )
                    for tr in rad_resp.json().get("data", []):
                        t = sanitize_text(tr.get("title_short") or tr.get("title"))
                        a = sanitize_text(tr.get("artist", {}).get("name"))
                        if not t or not a or any(k in t.lower() for k in SKIP_TITLE_KEYWORDS):
                            continue
                        key = (strip_title_extras(t).lower(), normalize_artist_key(a))
                        if key[0] and key not in primary_tracks and key not in radio_tracks:
                            radio_tracks[key] = {"title": t, "artist": a, "meta": {}}
                except Exception:
                    pass

        # 2. iTunes Catalog Search (catches deep album cuts & pre-caches metadata)
        try:
            it_resp = requests.get(
                "https://itunes.apple.com/search",
                params={"term": artist_name, "entity": "song", "limit": 100},
                headers=HEADERS,
                timeout=10,
            )
            for hit in it_resp.json().get("results", []):
                hit_artist = sanitize_text(hit.get("artistName"))
                # Strict artist check so common names like "Hole" don't pull unrelated songs
                if normalize_artist_key(hit_artist) != target_key:
                    continue
                t = sanitize_text(hit.get("trackName"))
                if not t or any(k in t.lower() for k in SKIP_TITLE_KEYWORDS):
                    continue

                r_date = hit.get("releaseDate", "")
                year = int(r_date[:4]) if len(r_date) >= 4 and r_date[:4].isdigit() else None
                meta = {
                    "year": year,
                    "genre": sanitize_text(hit.get("primaryGenreName", "rock")).lower() or "rock",
                    "preview_url": hit.get("previewUrl"),
                }
                key = (strip_title_extras(t).lower(), target_key)
                if not key[0]:
                    continue
                if key not in primary_tracks:
                    primary_tracks[key] = {"title": t, "artist": hit_artist, "meta": meta}
                elif not primary_tracks[key]["meta"]:
                    primary_tracks[key]["meta"] = meta
        except Exception:
            pass

        time.sleep(0.2)

    combined = list(primary_tracks.values()) + list(radio_tracks.values())
    print(
        f"\nCollected {len(primary_tracks):,} tracks from the 21 artists"
        + (f" + {len(radio_tracks):,} artist-radio tracks." if INCLUDE_SIMILAR_RADIO_TRACKS else ".")
    )
    return combined


def fetch_itunes_metadata(title: str, artist: str) -> dict:
    base_title = strip_title_extras(title)
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": f"{base_title} {artist}", "entity": "song", "limit": 1},
            headers=HEADERS,
            timeout=8,
        )
        results = resp.json().get("results", [])
        if not results:
            return {}
        hit = results[0]
        r_date = hit.get("releaseDate", "")
        year = int(r_date[:4]) if len(r_date) >= 4 and r_date[:4].isdigit() else None
        return {
            "year": year,
            "genre": sanitize_text(hit.get("primaryGenreName", "alternative")).lower() or "alternative",
            "preview_url": hit.get("previewUrl"),
        }
    except Exception:
        return {}


def fetch_lrclib_lyrics(title: str, artist: str) -> str:
    primary_artist = parse_artist_names(artist)[0] if parse_artist_names(artist) else artist
    title_variants = list(dict.fromkeys([title, strip_title_extras(title)]))

    for t_var in title_variants:
        if not t_var:
            continue
        try:
            resp = requests.get(
                "https://lrclib.net/api/get",
                params={"track_name": t_var, "artist_name": primary_artist},
                headers=HEADERS,
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                cleaned = clean_lyrics_text(
                    data.get("plainLyrics") or data.get("syncedLyrics") or ""
                )
                if len(cleaned) >= 80:
                    return cleaned
        except Exception:
            pass

    try:
        resp = requests.get(
            "https://lrclib.net/api/search",
            params={"q": f"{strip_title_extras(title)} {primary_artist}"},
            headers=HEADERS,
            timeout=8,
        )
        for item in resp.json()[:3]:
            cleaned = clean_lyrics_text(
                item.get("plainLyrics") or item.get("syncedLyrics") or ""
            )
            if len(cleaned) >= 80:
                return cleaned
    except Exception:
        pass

    return ""


def extract_audio_features(preview_url: str | None) -> tuple[float, float, float, float, float]:
    if preview_url:
        tmp_path = None
        try:
            import librosa

            audio_bytes = requests.get(preview_url, headers=HEADERS, timeout=10).content
            tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
            tmp_path = tmp.name
            tmp.write(audio_bytes)
            tmp.close()

            y, sr = librosa.load(tmp_path, sr=22050, duration=25.0)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(np.clip(float(np.atleast_1d(tempo)[0]), 60.0, 200.0))
            energy = float(np.clip(float(np.mean(librosa.feature.rms(y=y))) * 4.5, 0.05, 0.98))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            acousticness = float(np.clip(1.0 - (flatness * 18.0), 0.02, 0.98))
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            danceability = float(np.clip(float(np.mean(onset_env)) / 3.5, 0.15, 0.95))
            return bpm, energy, acousticness, 0.50, danceability
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return 118.0, 0.72, 0.25, 0.50, 0.58


def main() -> None:
    candidates = collect_all_artist_tracks()

    conn = sqlite3.connect(DB_PATH)
    existing = {
        (strip_title_extras(t).lower(), normalize_artist_key(a))
        for t, a in conn.execute("SELECT title, artist FROM songs").fetchall()
    }
    to_check = [
        c
        for c in candidates
        if (strip_title_extras(c["title"]).lower(), normalize_artist_key(c["artist"])) not in existing
    ]
    print(f"{len(to_check):,} new candidate tracks to process.\n")

    if not to_check:
        conn.close()
        return

    model_name = detect_db_model(conn)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model ({model_name}) on {device.upper()}...")
    model = SentenceTransformer(model_name, device=device)

    anchor_labels = list(VIBE_ANCHORS.keys())
    anchor_vecs = model.encode(
        list(VIBE_ANCHORS.values()),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    added = 0
    for idx, item in enumerate(to_check, start=1):
        title = item["title"]
        artist = item["artist"]
        norm_key = (strip_title_extras(title).lower(), normalize_artist_key(artist))
        if norm_key in existing:
            continue

        lyrics = fetch_lrclib_lyrics(title, artist)
        if not lyrics:
            continue

        meta = item["meta"] or fetch_itunes_metadata(title, artist)
        year = meta.get("year")
        genre = meta.get("genre", "alternative")

        l_vec = model.encode(
            [lyrics[:4000]], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)

        sims = anchor_vecs @ l_vec
        top_idx = np.argsort(sims)[::-1]
        primary_vibe = anchor_labels[top_idx[0]]
        secondary_vibe = anchor_labels[top_idx[1]]

        v_vec = (
            0.7 * anchor_vecs[top_idx[0]] + 0.3 * anchor_vecs[top_idx[1]]
        ).astype(np.float32)
        v_norm = np.linalg.norm(v_vec)
        if v_norm > 0:
            v_vec /= v_norm

        bpm, energy, acousticness, valence, danceability = extract_audio_features(
            meta.get("preview_url")
        )

        cur = conn.execute(
            """
            INSERT OR IGNORE INTO songs (
                spotify_id, title, artist, album_name, genre, release_year, popularity,
                lyrics, tempo_bpm, energy, acousticness, valence, danceability,
                musical_key, musical_mode, primary_vibe, secondary_vibe, lyrics_vec, vibe_vec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"radio_{abs(hash((title.lower(), artist.lower())))}",
                title,
                artist,
                None,
                genre,
                year,
                80,
                lyrics[:4000],
                bpm,
                energy,
                acousticness,
                valence,
                danceability,
                None,
                1,
                primary_vibe,
                secondary_vibe,
                l_vec.tobytes(),
                v_vec.tobytes(),
            ),
        )
        if cur.rowcount == 0:
            continue

        song_id = cur.lastrowid
        for name in parse_artist_names(artist):
            conn.execute("INSERT OR IGNORE INTO creators (name) VALUES (?)", (name,))
            cid = conn.execute(
                "SELECT creator_id FROM creators WHERE name = ?", (name,)
            ).fetchone()[0]
            conn.execute(
                "INSERT OR IGNORE INTO song_credits (song_id, creator_id, role) VALUES (?, ?, 'performer')",
                (song_id, cid),
            )

        existing.add(norm_key)
        added += 1
        if added % 15 == 0:
            conn.commit()
        year_label = str(year) if year else "N/A"
        print(f"[{idx}/{len(to_check)}] + Added: {title} — {artist} ({year_label} | {genre})")
        time.sleep(0.2)

    conn.commit()
    total_now = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    conn.close()
    print(f"\nDone! Added {added:,} new songs. Total library size: {total_now:,} songs.")


if __name__ == "__main__":
    main()