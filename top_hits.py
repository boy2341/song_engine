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

VIBE_ANCHORS = {
    "energetic party hype": "upbeat party dance club energetic fun celebration hype",
    "sad heartbreak longing": "sad heartbreak crying miss you lonely pain tears",
    "calm acoustic warmth": "calm peaceful acoustic warm gentle soft love comfort",
    "dark moody intense": "dark moody intense shadow anger revenge midnight tension",
    "happy sunny feelgood": "happy sunshine smile joyful carefree bright summer",
    "romantic dreamy love": "romantic dreamy soul forever kiss passion devotion",
}

PLAYLIST_QUERIES = [
    "Top Hits 2026",
    "Global Top 50 2026",
    "Best Songs of 2026",
    "Top Hits 2025",
    "Best Songs of 2025",
    "Top Hits 2024",
    "Best Songs of 2024",
    "Pop Hits 2026",
    "Hip Hop Hits 2026",
    "R&B Hits 2025",
    "Indie Pop 2025",
    "Bollywood Hits 2026",
    "Top Hindi Songs 2025",
    "Punjabi Hits 2025",
]


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


def parse_artist_names(raw: str) -> list[str]:
    text = sanitize_text(raw)
    if not text:
        return []
    parts = re.split(r"\s*(?:,|&|\bfeat\.?\b|\bft\.?\b|\bx\b)\s*", text, flags=re.IGNORECASE)
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


def collect_top_chart_candidates() -> list[tuple[str, str]]:
    """Gather unique (title, artist) pairs from Apple Music Top 100 and Deezer charts/playlists."""
    candidates: dict[tuple[str, str], tuple[str, str]] = {}

    # 1. Apple Music Top 100 Most-Played Charts (US/Global, India, UK)
    for country in ["us", "in", "gb"]:
        url = f"https://rss.applemarketingtools.com/api/v2/{country}/music/most-played/100/songs.json"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            for item in resp.json().get("feed", {}).get("results", []):
                t = sanitize_text(item.get("name"))
                a = sanitize_text(item.get("artistName"))
                if t and a:
                    candidates[(t.lower(), a.lower())] = (t, a)
        except Exception:
            pass

    # 2. Deezer Global Top 100 Chart
    try:
        resp = requests.get(
            "https://api.deezer.com/chart/0/tracks",
            params={"limit": 100},
            headers=HEADERS,
            timeout=10,
        )
        for tr in resp.json().get("data", []):
            t = sanitize_text(tr.get("title_short") or tr.get("title"))
            a = sanitize_text(tr.get("artist", {}).get("name"))
            if t and a:
                candidates[(t.lower(), a.lower())] = (t, a)
    except Exception:
        pass

    # 3. Deezer Curated 2024-2026 Hit Playlists
    for query in PLAYLIST_QUERIES:
        try:
            resp = requests.get(
                "https://api.deezer.com/search/playlist",
                params={"q": query, "limit": 3},
                headers=HEADERS,
                timeout=10,
            )
            for pl in resp.json().get("data", []):
                tracklist_url = pl.get("tracklist")
                if not tracklist_url:
                    continue
                tr_resp = requests.get(
                    tracklist_url, params={"limit": 100}, headers=HEADERS, timeout=10
                )
                for tr in tr_resp.json().get("data", []):
                    t = sanitize_text(tr.get("title_short") or tr.get("title"))
                    a = sanitize_text(tr.get("artist", {}).get("name"))
                    if t and a:
                        candidates[(t.lower(), a.lower())] = (t, a)
                time.sleep(0.15)
        except Exception:
            continue

    return list(candidates.values())


def fetch_itunes_metadata(title: str, artist: str) -> dict:
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": f"{title} {artist}", "entity": "song", "limit": 1},
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
            "genre": sanitize_text(hit.get("primaryGenreName", "pop")).lower() or "pop",
            "preview_url": hit.get("previewUrl"),
        }
    except Exception:
        return {}


def fetch_lrclib_lyrics(title: str, artist: str) -> str:
    try:
        resp = requests.get(
            "https://lrclib.net/api/get",
            params={"track_name": title, "artist_name": artist},
            headers=HEADERS,
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            cleaned = clean_lyrics_text(data.get("plainLyrics") or data.get("syncedLyrics") or "")
            if len(cleaned) >= 80:
                return cleaned
    except Exception:
        pass

    try:
        resp = requests.get(
            "https://lrclib.net/api/search",
            params={"q": f"{title} {artist}"},
            headers=HEADERS,
            timeout=8,
        )
        for item in resp.json()[:2]:
            cleaned = clean_lyrics_text(item.get("plainLyrics") or item.get("syncedLyrics") or "")
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
            return bpm, energy, acousticness, 0.55, danceability
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return 115.0, 0.68, 0.28, 0.55, 0.68


def main() -> None:
    print("Fetching 2024–2026 top charts & hit playlists...")
    candidates = collect_top_chart_candidates()
    print(f"Collected {len(candidates):,} unique chart/playlist songs.")

    conn = sqlite3.connect(DB_PATH)
    existing = {
        (t.lower(), a.lower())
        for t, a in conn.execute("SELECT title, artist FROM songs").fetchall()
    }
    to_check = [(t, a) for (t, a) in candidates if (t.lower(), a.lower()) not in existing]
    print(f"{len(to_check):,} songs are new to your library.\n")

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
    for idx, (title, artist) in enumerate(to_check, start=1):
        meta = fetch_itunes_metadata(title, artist)
        year = meta.get("year")
        # Filter out older throwback tracks on playlists
        if year is not None and year < 2024:
            continue

        lyrics = fetch_lrclib_lyrics(title, artist)
        if not lyrics:
            continue

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
                f"chart_{abs(hash((title.lower(), artist.lower())))}",
                title,
                artist,
                None,
                meta.get("genre", "pop"),
                year or 2026,
                88,
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

        added += 1
        if added % 15 == 0:
            conn.commit()
        print(f"[{idx}/{len(to_check)}] + Added: {title} — {artist} ({year or '2026'})")
        time.sleep(0.2)

    conn.commit()
    total_now = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    conn.close()
    print(f"\nDone! Added {added:,} new top hits. Total library size: {total_now:,} songs.")


if __name__ == "__main__":
    main()