import ast
import html
import json
import os
import re
import sqlite3
import tempfile
import urllib.parse
import numpy as np
import requests
import streamlit as st

from config import DB_PATH, VIBE_ANCHORS
from step3 import MusicRecommender

HEADERS = {"User-Agent": "SongFinderEngine/1.0 (open-source-music-recommender)"}
SPOTIFY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{22}$")

SPOTIFY_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="#ffffff" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
</svg>
"""

st.set_page_config(page_title="Song Finder", page_icon="🎧", layout="wide")

st.markdown(
    """
    <style>
    .spotify-icon-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 38.4px;
        background-color: #1DB954;
        border-radius: 8px;
        text-decoration: none;
        transition: background-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 8px;
    }
    .spotify-icon-btn:hover {
        background-color: #1ed760;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- User Profiles & Playlists ---
def init_user_tables() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pin TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS playlists (
            playlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(profile_id, name),
            FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS playlist_songs (
            playlist_id INTEGER NOT NULL,
            song_title TEXT NOT NULL,
            song_artist TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (playlist_id, song_title, song_artist),
            FOREIGN KEY (playlist_id) REFERENCES playlists(playlist_id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()


def create_or_login_profile(username: str, pin: str = "") -> tuple[bool, str, int | None]:
    clean_user, clean_pin = username.strip(), pin.strip()
    if not clean_user:
        return False, "Please enter a profile name.", None

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT profile_id, COALESCE(pin, '') FROM profiles WHERE LOWER(username) = LOWER(?)",
        (clean_user,),
    ).fetchone()
    if row:
        pid, saved_pin = row
        conn.close()
        if saved_pin and saved_pin != clean_pin:
            return False, "Incorrect PIN for this profile.", None
        return True, f"Welcome back, {clean_user}!", pid

    cur = conn.execute("INSERT INTO profiles (username, pin) VALUES (?, ?)", (clean_user, clean_pin))
    pid = cur.lastrowid
    conn.execute("INSERT OR IGNORE INTO playlists (profile_id, name) VALUES (?, ?)", (pid, "Favorites"))
    conn.commit()
    conn.close()
    return True, f"Created profile '{clean_user}' with a 'Favorites' playlist!", pid


def get_user_playlists(profile_id: int) -> list[tuple[int, str]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT playlist_id, name FROM playlists WHERE profile_id = ? ORDER BY playlist_id ASC", (profile_id,)).fetchall()
    conn.close()
    return rows


def create_playlist(profile_id: int, name: str) -> tuple[bool, str]:
    clean_name = name.strip()
    if not clean_name:
        return False, "Enter a playlist name."
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO playlists (profile_id, name) VALUES (?, ?)", (profile_id, clean_name))
        conn.commit()
        return True, f"Created playlist '{clean_name}'."
    except sqlite3.IntegrityError:
        return False, "You already have a playlist with that name."
    finally:
        conn.close()


def get_playlist_songs(playlist_id: int) -> list[tuple[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT song_title, song_artist FROM playlist_songs WHERE playlist_id = ? ORDER BY added_at DESC", (playlist_id,)).fetchall()
    conn.close()
    return rows


def add_song_to_playlist(playlist_id: int, title: str, artist: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("INSERT OR IGNORE INTO playlist_songs (playlist_id, song_title, song_artist) VALUES (?, ?, ?)", (playlist_id, title.strip(), artist.strip()))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def remove_song_from_playlist(playlist_id: int, title: str, artist: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM playlist_songs WHERE playlist_id = ? AND song_title = ? AND song_artist = ?", (playlist_id, title, artist))
    conn.commit()
    conn.close()


def export_profile_json(profile_id: int, username: str) -> str:
    payload = {
        "username": username,
        "playlists": [
            {"name": pname, "songs": [{"title": t, "artist": a} for t, a in get_playlist_songs(pid)]}
            for pid, pname in get_user_playlists(profile_id)
        ],
    }
    return json.dumps(payload, indent=2)


# --- Helpers ---
def render_spotify_icon_button(url: str) -> None:
    safe_url = html.escape(url, quote=True)
    st.markdown(
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
        f'title="Open in Spotify" aria-label="Open in Spotify" class="spotify-icon-btn">'
        f"{SPOTIFY_SVG}</a>",
        unsafe_allow_html=True,
    )


def sanitize_text(val: object) -> str:
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
    return sanitize_text(re.sub(r"\s+", " ", text))


def parse_artist_names(raw: str) -> list[str]:
    text = sanitize_text(raw)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [sanitize_text(x) for x in parsed if sanitize_text(x)]
        except Exception:
            pass
    return [sanitize_text(p) for p in text.split(",") if sanitize_text(p)]


def make_spotify_url(spotify_id: str | None, title: str, artist: str) -> str:
    clean_id = (spotify_id or "").strip()
    if SPOTIFY_ID_PATTERN.fullmatch(clean_id):
        return f"https://open.spotify.com/track/{clean_id}"
    return f"https://open.spotify.com/search/{urllib.parse.quote(f'{title} {artist}'.strip())}"


@st.cache_resource
def load_spotify_url_map() -> dict[tuple[str, str], str]:
    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    if "spotify_url" in cols:
        rows = conn.execute("SELECT title, artist, spotify_id, spotify_url FROM songs").fetchall()
    else:
        rows = [(t, a, sid, None) for t, a, sid in conn.execute("SELECT title, artist, spotify_id FROM songs").fetchall()]
    conn.close()
    return {
        ((t or "").strip().lower(), (a or "").strip().lower()): (sp_url or make_spotify_url(sp_id, t or "", a or ""))
        for t, a, sp_id, sp_url in rows
    }


def get_song_spotify_url(title: str, artist: str, url_map: dict[tuple[str, str], str]) -> str:
    return url_map.get((title.strip().lower(), artist.strip().lower())) or make_spotify_url(None, title, artist)


def fetch_lrclib_lyrics(title: str, artist: str) -> str:
    try:
        r = requests.get("https://lrclib.net/api/get", params={"track_name": title, "artist_name": artist}, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            data = r.json()
            cleaned = clean_lyrics_text(data.get("plainLyrics") or data.get("syncedLyrics") or "")
            if len(cleaned) >= 80:
                return cleaned
    except Exception:
        pass
    try:
        r = requests.get("https://lrclib.net/api/search", params={"q": f"{title} {artist}"}, headers=HEADERS, timeout=8)
        for item in r.json()[:2]:
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
    return 115.0, 0.68, 0.28, 0.55, 0.68


def ingest_online_items(query_text: str, mode_type: str, model) -> list[dict[str, str]]:
    candidates: list[dict] = []
    if mode_type == "Artist + Radio":
        try:
            r = requests.get("https://api.deezer.com/search/artist", params={"q": query_text, "limit": 1}, headers=HEADERS, timeout=8).json()
            artists = r.get("data", [])
            if artists:
                aid = artists[0]["id"]
                for url in (f"https://api.deezer.com/artist/{aid}/top?limit=12", f"https://api.deezer.com/artist/{aid}/radio?limit=12"):
                    for tr in requests.get(url, headers=HEADERS, timeout=8).json().get("data", []):
                        t = sanitize_text(tr.get("title_short") or tr.get("title"))
                        a = sanitize_text(tr.get("artist", {}).get("name"))
                        if t and a:
                            candidates.append({"title": t, "artist": a})
        except Exception:
            pass
    else:
        try:
            hits = requests.get("https://itunes.apple.com/search", params={"term": query_text, "entity": "song", "limit": 5}, headers=HEADERS, timeout=8).json().get("results", [])
            for h in hits:
                rd = h.get("releaseDate", "")
                candidates.append({
                    "title": sanitize_text(h.get("trackName")),
                    "artist": sanitize_text(h.get("artistName")),
                    "year": int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else 2025,
                    "genre": sanitize_text(h.get("primaryGenreName", "pop")).lower() or "pop",
                    "preview_url": h.get("previewUrl"),
                })
        except Exception:
            pass

    if not candidates:
        return []

    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    if "spotify_url" not in cols:
        conn.execute("ALTER TABLE songs ADD COLUMN spotify_url TEXT;")

    existing = {(t.lower(), a.lower()) for t, a in conn.execute("SELECT title, artist FROM songs").fetchall()}

    anchor_labels = list(VIBE_ANCHORS.keys())
    centroids = []
    for label in anchor_labels:
        vecs = model.encode(VIBE_ANCHORS[label], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        m_vec = vecs.mean(axis=0)
        n = np.linalg.norm(m_vec)
        centroids.append(m_vec / n if n > 0 else m_vec)
    anchor_vecs = np.vstack(centroids).astype(np.float32)

    added: list[dict[str, str]] = []
    for item in candidates:
        title, artist = item["title"], item["artist"]
        if not title or not artist or (title.lower(), artist.lower()) in existing:
            continue

        lyrics = fetch_lrclib_lyrics(title, artist)
        if not lyrics:
            continue

        l_vec = model.encode([lyrics[:4000]], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        sims = anchor_vecs @ l_vec
        top2 = np.argsort(sims)[::-1][:2]
        v_vec = (0.7 * anchor_vecs[top2[0]] + 0.3 * anchor_vecs[top2[1]]).astype(np.float32)
        v_vec /= max(float(np.linalg.norm(v_vec)), 1e-12)

        bpm, energy, acousticness, valence, danceability = extract_audio_features(item.get("preview_url"))
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
                f"online_{abs(hash((title.lower(), artist.lower())))}",
                sp_url,
                title,
                artist,
                None,
                item.get("genre", "alternative"),
                item.get("year"),
                84,
                lyrics[:200],
                bpm,
                energy,
                acousticness,
                valence,
                danceability,
                None,
                1,
                anchor_labels[int(top2[0])],
                anchor_labels[int(top2[1])],
                l_vec.tobytes(),
                v_vec.tobytes(),
            ),
        )
        if cur.rowcount == 0:
            continue

        song_id = cur.lastrowid
        for name in parse_artist_names(artist):
            conn.execute("INSERT OR IGNORE INTO creators (name) VALUES (?)", (name,))
            cid = conn.execute("SELECT creator_id FROM creators WHERE name = ?", (name,)).fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO song_credits (song_id, creator_id, role) VALUES (?, ?, 'performer')", (song_id, cid))

        existing.add((title.lower(), artist.lower()))
        added.append({"title": title, "artist": artist})
        if len(added) >= 8:
            break

    conn.commit()
    conn.close()
    return added


def recommend_from_playlist(rec_engine: MusicRecommender, playlist_tracks: list[tuple[str, str]], weights: dict[str, int], top_k: int = 10) -> list[dict]:
    seed_indices: list[int] = []
    for title, artist in playlist_tracks:
        matches = rec_engine.search_catalog(title, limit=10)
        matched = next(
            (m["idx"] for m in matches if m["title"].strip().lower() == title.strip().lower() and m["artist"].strip().lower() == artist.strip().lower()),
            matches[0]["idx"] if matches else None,
        )
        if matched is not None and matched not in seed_indices:
            seed_indices.append(matched)

    if not seed_indices:
        return []

    seed_set = set(seed_indices)
    active_seeds = seed_indices[:20]
    pool_size = min(rec_engine.n_songs - 1, 250)
    candidates: dict[int, dict] = {}

    for s_idx in active_seeds:
        for item in rec_engine.recommend(seed_idx=s_idx, weights=weights, top_k=pool_size)["results"]:
            c_idx = item["idx"]
            if c_idx in seed_set:
                continue
            entry = candidates.setdefault(c_idx, {"item": item, "scores": [], "lyrics": [], "vibe": [], "tune": [], "reasons": []})
            entry["scores"].append(item["score"])
            entry["lyrics"].append(item["breakdown"]["lyrics"])
            entry["vibe"].append(item["breakdown"]["vibe"])
            entry["tune"].append(item["breakdown"]["tune"])
            entry["reasons"].extend(item["reasons"])

    ranked = []
    top_n = min(3, len(active_seeds))
    for entry in candidates.values():
        s_scores = sorted(entry["scores"], reverse=True)
        merged = dict(entry["item"])
        merged["score"] = min(0.99, float(np.mean(s_scores[:top_n])) * (1.0 + 0.04 * (len(s_scores) - 1)))
        merged["breakdown"] = {
            "lyrics": float(np.mean(sorted(entry["lyrics"], reverse=True)[:top_n])),
            "vibe": float(np.mean(sorted(entry["vibe"], reverse=True)[:top_n])),
            "tune": float(np.mean(sorted(entry["tune"], reverse=True)[:top_n])),
        }
        merged["reasons"] = list(dict.fromkeys(entry["reasons"]))[:3]
        ranked.append(merged)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


@st.cache_resource
def load_engine() -> MusicRecommender:
    return MusicRecommender()


@st.cache_resource
def load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT LENGTH(lyrics_vec) FROM songs WHERE lyrics_vec IS NOT NULL LIMIT 1").fetchone()
    conn.close()
    model_name = "all-MiniLM-L6-v2" if (row and row[0] == 384 * 4) else "BAAI/bge-base-en-v1.5"
    return SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")


init_user_tables()
engine = load_engine()
spotify_urls = load_spotify_url_map()

for key, default in [
    ("profile_id", None),
    ("profile_name", None),
    ("active_playlist_id", None),
    ("selected_song_idx", None),
    ("w_lyrics", 35),
    ("w_vibe", 25),
    ("w_tune", 25),
    ("w_creators", 15),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.selected_song_idx is not None and st.session_state.selected_song_idx >= engine.n_songs:
    st.session_state.selected_song_idx = None


def apply_preset(lyrics: int, vibe: int, tune: int, creators: int) -> None:
    st.session_state.w_lyrics = lyrics
    st.session_state.w_vibe = vibe
    st.session_state.w_tune = tune
    st.session_state.w_creators = creators


def format_reason(raw_reason: str) -> str:
    if raw_reason.startswith("vibe: "):
        return "💫 Similar emotional vibe"
    if raw_reason.startswith("lyrics"):
        return "📝 Similar story & lyrics"
    if raw_reason.startswith("tune"):
        return "🥁 Similar sound & rhythm"
    if raw_reason.startswith("same "):
        return f"🎤 Also by {raw_reason.split(': ', 1)[-1]}"
    return raw_reason


# --- Sidebar ---
with st.sidebar:
    st.header("👤 Your Profile & Playlists")

    if st.session_state.profile_id is None:
        with st.form("profile_form"):
            u_input = st.text_input("Profile Name", placeholder="e.g., Abhishank")
            pin_input = st.text_input("Optional PIN", type="password", placeholder="Leave blank if none")
            if st.form_submit_button("Sign In / Create Profile", use_container_width=True):
                ok, msg, pid = create_or_login_profile(u_input, pin_input)
                if ok and pid is not None:
                    st.session_state.profile_id = pid
                    st.session_state.profile_name = u_input.strip()
                    pls = get_user_playlists(pid)
                    st.session_state.active_playlist_id = pls[0][0] if pls else None
                    st.toast(msg, icon="🎉")
                    st.rerun()
                else:
                    st.error(msg)
    else:
        c_u, c_out = st.columns([2.6, 1.2])
        c_u.markdown(f"Signed in as **{st.session_state.profile_name}**")
        if c_out.button("Switch", use_container_width=True):
            st.session_state.profile_id = None
            st.session_state.profile_name = None
            st.session_state.active_playlist_id = None
            st.rerun()

        pls = get_user_playlists(st.session_state.profile_id)
        if pls:
            pl_map = {name: pid for pid, name in pls}
            pl_names = list(pl_map.keys())
            idx = next((i for i, (_, pid) in enumerate(pls) if pid == st.session_state.active_playlist_id), 0)
            chosen = st.selectbox("Active playlist:", pl_names, index=idx)
            st.session_state.active_playlist_id = pl_map[chosen]

        with st.expander("Create New Playlist"):
            new_pl = st.text_input("Playlist name", placeholder="e.g., Late Night Drive")
            if st.button("Create Playlist", use_container_width=True):
                ok, msg = create_playlist(st.session_state.profile_id, new_pl)
                if ok:
                    st.session_state.active_playlist_id = get_user_playlists(st.session_state.profile_id)[-1][0]
                    st.toast(msg, icon="✅")
                    st.rerun()
                else:
                    st.warning(msg)

        if st.session_state.active_playlist_id is not None:
            saved = get_playlist_songs(st.session_state.active_playlist_id)
            with st.expander(f"Saved Songs ({len(saved)})"):
                if not saved:
                    st.caption("Click '+ Save' on any song card to add it here.")
                for s_t, s_a in saved:
                    ct, cd = st.columns([4, 1])
                    ct.caption(f"**{s_t}** — {s_a}")
                    if cd.button("✕", key=f"del_{s_t}_{s_a}"):
                        remove_song_from_playlist(st.session_state.active_playlist_id, s_t, s_a)
                        st.rerun()

            st.download_button(
                "Download Profile Backup (.json)",
                data=export_profile_json(st.session_state.profile_id, st.session_state.profile_name),
                file_name=f"{st.session_state.profile_name.lower()}_playlists.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    st.header("How do you want to discover?")
    mode = st.radio(
        "Choose a search style:",
        ["Find Songs Like One I Love", "Recommend From My Playlist", "Describe a Mood or Scene"],
    )

    if mode != "Describe a Mood or Scene":
        st.divider()
        st.subheader("Quick Presets")
        p1, p2, p3 = st.columns(3)
        p1.button("Balanced", use_container_width=True, on_click=apply_preset, args=(35, 25, 25, 15))
        p2.button("Deep Feel", use_container_width=True, on_click=apply_preset, args=(45, 40, 10, 5))
        p3.button("Same Beat", use_container_width=True, on_click=apply_preset, args=(10, 20, 60, 10))

        st.subheader("Fine-Tune Your Mix")
        lyrics_w = st.slider("Story & Lyrics", 0, 100, key="w_lyrics")
        vibe_w = st.slider("Mood & Atmosphere", 0, 100, key="w_vibe")
        tune_w = st.slider("Beat & Sound", 0, 100, key="w_tune")
        creators_w = st.slider("Same Artist", 0, 100, key="w_creators")
    else:
        lyrics_w, vibe_w, tune_w, creators_w = 35, 25, 25, 15

    st.divider()
    top_k = st.slider("Number of recommendations", 5, 20, 10)

    st.divider()
    st.subheader("Expand Music Library")
    import_type = st.radio("Import type:", ["Single Song", "Artist + Radio"], horizontal=True)
    import_q = st.text_input("Name:", placeholder="e.g., Sleep Token" if import_type == "Artist + Radio" else "e.g., Die With A Smile")
    if st.button("Fetch & Add to Library", use_container_width=True) and import_q.strip():
        with st.spinner("Fetching lyrics, audio & vibes..."):
            added = ingest_online_items(import_q.strip(), import_type, load_embedder())
            if added:
                load_engine.clear()
                load_spotify_url_map.clear()
                fresh = load_engine()
                m = fresh.search_catalog(added[0]["title"], limit=5)
                if m:
                    st.session_state.selected_song_idx = m[0]["idx"]
                st.toast(f"Added {len(added)} new track(s)!", icon="✅")
                st.rerun()
            else:
                st.warning("No new tracks with lyrics found (or already in library).")

    st.caption(f"Library size: **{engine.n_songs:,}** songs")

# --- Main Page ---
st.title("🎧 Smart Song Finder")
st.caption("Discover music by its story, emotional mood, sound, and creators.")

active_saved = (
    {(t.strip().lower(), a.strip().lower()) for t, a in get_playlist_songs(st.session_state.active_playlist_id)}
    if st.session_state.active_playlist_id is not None
    else set()
)
results = []

if mode == "Find Songs Like One I Love":
    q = st.text_input("Search for a song or artist to start from", placeholder="Type a song title or artist...")
    if q.strip():
        matches = engine.search_catalog(q, limit=15)
        if matches:
            placeholder = "— Pick a song from the matches below —"
            opts = {f"{m['title']} — {m['artist']}": m["idx"] for m in matches}

            def on_pick() -> None:
                c = st.session_state.get("song_dropdown")
                if c in opts:
                    st.session_state.selected_song_idx = opts[c]

            st.selectbox("Select a track:", [placeholder] + list(opts.keys()), key="song_dropdown", on_change=on_pick)
        else:
            st.info(f"'{q}' isn't in your library yet — use **Expand Music Library** in the sidebar to add it!")

    if st.session_state.selected_song_idx is None:
        st.info("Search and select a song above to see personalized recommendations.")
    else:
        data = engine.recommend(
            seed_idx=st.session_state.selected_song_idx,
            weights={"lyrics": lyrics_w, "vibe": vibe_w, "tune": tune_w, "creators": creators_w},
            top_k=top_k,
        )
        cur = data["seed"]
        with st.container(border=True):
            st.caption("CURRENTLY MATCHING AGAINST")
            c1, c2, c3, c4 = st.columns([3.8, 0.65, 0.95, 0.85])
            c1.markdown(f"### {cur['title']}\n**{cur['artist']}**")
            with c2:
                st.write("")
                render_spotify_icon_button(get_song_spotify_url(cur["title"], cur["artist"], spotify_urls))
            with c3:
                st.write("")
                if st.session_state.active_playlist_id is not None:
                    is_saved = (cur["title"].strip().lower(), cur["artist"].strip().lower()) in active_saved
                    if st.button("Saved ✓" if is_saved else "+ Save", key="save_seed", use_container_width=True, disabled=is_saved):
                        add_song_to_playlist(st.session_state.active_playlist_id, cur["title"], cur["artist"])
                        st.rerun()
            with c4:
                st.write("")
                if st.button("Clear", use_container_width=True):
                    st.session_state.selected_song_idx = None
                    st.rerun()
        results = data["results"]

elif mode == "Recommend From My Playlist":
    if st.session_state.profile_id is None or st.session_state.active_playlist_id is None:
        st.info("Sign in or create a profile in the sidebar to save songs and get playlist-based recommendations.")
    else:
        pl_tracks = get_playlist_songs(st.session_state.active_playlist_id)
        if not pl_tracks:
            st.info("Your active playlist is empty. Save a few songs using '+ Save' first!")
        else:
            st.success(f"Matching against **{len(pl_tracks)}** saved song(s) in your active playlist.")
            results = recommend_from_playlist(
                engine, pl_tracks, {"lyrics": lyrics_w, "vibe": vibe_w, "tune": tune_w, "creators": creators_w}, top_k
            )

else:
    prompt_text = st.text_input("What kind of vibe, story, or moment are you looking for?", placeholder="e.g., driving alone on a rainy highway at midnight...")
    if prompt_text.strip():
        results = engine.recommend_by_prompt(prompt=prompt_text, top_k=top_k)["results"]
    else:
        st.info("Describe a mood, lyric theme, or scene above to find matching songs.")

if results:
    st.subheader("Recommended For You")
    for rank, item in enumerate(results, start=1):
        with st.container(border=True):
            col_song, col_why, col_btn = st.columns([3.2, 2.1, 1.3])
            with col_song:
                yr = f" ({item['year']})" if item.get("year") else ""
                gn = f" • *{(item.get('genre') or '').strip().title()}*" if (item.get("genre") or "").strip() else ""
                st.markdown(f"#### {rank}. {item['title']}")
                st.markdown(f"**{item['artist']}**{yr}{gn}")
                st.write("  \n".join(dict.fromkeys(format_reason(r) for r in item["reasons"])))

            with col_why:
                pct = min(100, int(round(item["score"] * 100)))
                st.markdown(f"**{pct}% Match**")
                st.progress(pct / 100.0)
                b = item["breakdown"]
                st.caption(f"Story: {int(b['lyrics']*100)}% • Mood: {int(b['vibe']*100)}% • Beat: {int(b['tune']*100)}%")

            with col_btn:
                b_sp, b_sv = st.columns(2)
                with b_sp:
                    render_spotify_icon_button(get_song_spotify_url(item["title"], item["artist"], spotify_urls))
                with b_sv:
                    if st.session_state.active_playlist_id is not None:
                        is_saved = (item["title"].strip().lower(), item["artist"].strip().lower()) in active_saved
                        if st.button("✓" if is_saved else "+ Save", key=f"sv_{item['idx']}_{rank}", use_container_width=True, disabled=is_saved):
                            add_song_to_playlist(st.session_state.active_playlist_id, item["title"], item["artist"])
                            st.rerun()
                    else:
                        st.button("+ Save", key=f"svd_{item['idx']}_{rank}", use_container_width=True, disabled=True)

                if st.button("More Like This", key=f"more_{item['idx']}_{rank}", use_container_width=True):
                    st.session_state.selected_song_idx = item["idx"]
                    st.rerun()