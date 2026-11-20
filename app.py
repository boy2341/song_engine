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

from step3 import MusicRecommender

DB_PATH = "music_engine.db"
HEADERS = {"User-Agent": "SongFinderEngine/1.0 (local-music-recommender)"}
SPOTIFY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{22}$")

VIBE_ANCHORS = {
    "energetic party hype": "upbeat party dance club energetic fun celebration hype",
    "sad heartbreak longing": "sad heartbreak crying miss you lonely pain tears",
    "calm acoustic warmth": "calm peaceful acoustic warm gentle soft love comfort",
    "dark moody intense": "dark moody intense shadow anger revenge midnight tension",
    "happy sunny feelgood": "happy sunshine smile joyful carefree bright summer",
    "romantic dreamy love": "romantic dreamy soul forever kiss passion devotion",
}

SPOTIFY_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="#ffffff" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
</svg>
"""

st.set_page_config(
    page_title="Song Finder",
    page_icon="🎧",
    layout="wide",
)

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
    .spotify-icon-btn:active {
        transform: translateY(0px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Database Helpers for Profiles & Playlists ---
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


def get_all_profiles() -> list[tuple[int, str, str]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT profile_id, username, COALESCE(pin, '') FROM profiles ORDER BY username COLLATE NOCASE"
    ).fetchall()
    conn.close()
    return rows


def create_or_login_profile(username: str, pin: str = "") -> tuple[bool, str, int | None]:
    clean_user = username.strip()
    clean_pin = pin.strip()
    if not clean_user:
        return False, "Please enter a profile name.", None

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT profile_id, COALESCE(pin, '') FROM profiles WHERE LOWER(username) = LOWER(?)",
        (clean_user,),
    ).fetchone()

    if row:
        pid, saved_pin = row
        if saved_pin and saved_pin != clean_pin:
            conn.close()
            return False, "Incorrect PIN for this profile.", None
        conn.close()
        return True, f"Welcome back, {clean_user}!", pid

    cur = conn.execute(
        "INSERT INTO profiles (username, pin) VALUES (?, ?)",
        (clean_user, clean_pin),
    )
    pid = cur.lastrowid
    conn.execute(
        "INSERT OR IGNORE INTO playlists (profile_id, name) VALUES (?, ?)",
        (pid, "Favorites"),
    )
    conn.commit()
    conn.close()
    return True, f"Profile '{clean_user}' created with a 'Favorites' playlist!", pid


def get_user_playlists(profile_id: int) -> list[tuple[int, str]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT playlist_id, name FROM playlists WHERE profile_id = ? ORDER BY playlist_id ASC",
        (profile_id,),
    ).fetchall()
    conn.close()
    return rows


def create_playlist(profile_id: int, name: str) -> tuple[bool, str]:
    clean_name = name.strip()
    if not clean_name:
        return False, "Enter a playlist name."
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO playlists (profile_id, name) VALUES (?, ?)",
            (profile_id, clean_name),
        )
        conn.commit()
        return True, f"Created playlist '{clean_name}'."
    except sqlite3.IntegrityError:
        return False, "You already have a playlist with that name."
    finally:
        conn.close()


def get_playlist_songs(playlist_id: int) -> list[tuple[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT song_title, song_artist FROM playlist_songs WHERE playlist_id = ? ORDER BY added_at DESC",
        (playlist_id,),
    ).fetchall()
    conn.close()
    return rows


def add_song_to_playlist(playlist_id: int, title: str, artist: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT OR IGNORE INTO playlist_songs (playlist_id, song_title, song_artist) VALUES (?, ?, ?)",
        (playlist_id, title.strip(), artist.strip()),
    )
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def remove_song_from_playlist(playlist_id: int, title: str, artist: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM playlist_songs WHERE playlist_id = ? AND song_title = ? AND song_artist = ?",
        (playlist_id, title, artist),
    )
    conn.commit()
    conn.close()


def export_profile_json(profile_id: int, username: str) -> str:
    playlists = get_user_playlists(profile_id)
    payload = {
        "username": username,
        "playlists": [
            {
                "name": pname,
                "songs": [{"title": t, "artist": a} for t, a in get_playlist_songs(pid)],
            }
            for pid, pname in playlists
        ],
    }
    return json.dumps(payload, indent=2)


# --- General & Recommendation Helpers ---
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
    text = re.sub(r"\s+", " ", text)
    return sanitize_text(text)


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
    return [sanitize_text(part) for part in text.split(",") if sanitize_text(part)]


def make_spotify_url(spotify_id: str | None, title: str, artist: str) -> str:
    clean_id = (spotify_id or "").strip()
    if SPOTIFY_ID_PATTERN.fullmatch(clean_id):
        return f"https://open.spotify.com/track/{clean_id}"
    query = urllib.parse.quote(f"{title} {artist}".strip())
    return f"https://open.spotify.com/search/{query}"


@st.cache_resource
def load_spotify_url_map() -> dict[tuple[str, str], str]:
    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    if "spotify_url" in cols:
        rows = conn.execute(
            "SELECT title, artist, spotify_id, spotify_url FROM songs"
        ).fetchall()
    else:
        rows = [
            (t, a, sid, None)
            for t, a, sid in conn.execute("SELECT title, artist, spotify_id FROM songs").fetchall()
        ]
    conn.close()

    url_map: dict[tuple[str, str], str] = {}
    for title, artist, sp_id, sp_url in rows:
        key = ((title or "").strip().lower(), (artist or "").strip().lower())
        url_map[key] = sp_url or make_spotify_url(sp_id, title or "", artist or "")
    return url_map


def get_song_spotify_url(title: str, artist: str, url_map: dict[tuple[str, str], str]) -> str:
    key = ((title or "").strip().lower(), (artist or "").strip().lower())
    return url_map.get(key) or make_spotify_url(None, title, artist)


def detect_db_model() -> str:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT LENGTH(lyrics_vec) FROM songs WHERE lyrics_vec IS NOT NULL LIMIT 1"
    ).fetchone()
    conn.close()
    if row and row[0]:
        dim = row[0] // 4
        if dim == 768:
            return "BAAI/bge-base-en-v1.5"
        if dim == 384:
            return "all-MiniLM-L6-v2"
    return "BAAI/bge-base-en-v1.5"


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


def ingest_online_query(query_text: str, model, limit: int = 4) -> list[dict[str, str]]:
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query_text, "entity": "song", "limit": limit},
            headers=HEADERS,
            timeout=10,
        )
        hits = resp.json().get("results", [])
    except Exception:
        return []

    if not hits:
        return []

    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    if "spotify_url" not in cols:
        conn.execute("ALTER TABLE songs ADD COLUMN spotify_url TEXT;")

    existing = {
        (t.lower(), a.lower())
        for t, a in conn.execute("SELECT title, artist FROM songs").fetchall()
    }

    anchor_labels = list(VIBE_ANCHORS.keys())
    anchor_vecs = model.encode(
        list(VIBE_ANCHORS.values()),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    added: list[dict[str, str]] = []
    for hit in hits:
        title = sanitize_text(hit.get("trackName"))
        artist = sanitize_text(hit.get("artistName"))
        if not title or not artist or (title.lower(), artist.lower()) in existing:
            continue

        lyrics = fetch_lrclib_lyrics(title, artist)
        if not lyrics:
            continue

        r_date = hit.get("releaseDate", "")
        year = int(r_date[:4]) if len(r_date) >= 4 and r_date[:4].isdigit() else 2025
        genre = sanitize_text(hit.get("primaryGenreName", "pop")).lower() or "pop"
        sp_url = make_spotify_url(None, title, artist)

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
            hit.get("previewUrl")
        )

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
                genre,
                year,
                85,
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

        existing.add((title.lower(), artist.lower()))
        added.append({"title": title, "artist": artist})

    conn.commit()
    conn.close()
    return added


def recommend_from_playlist(
    rec_engine: MusicRecommender,
    playlist_tracks: list[tuple[str, str]],
    weights: dict[str, int],
    top_k: int = 10,
) -> list[dict]:
    """Aggregate recommendations across all songs in a user's playlist."""
    seed_indices: list[int] = []
    for title, artist in playlist_tracks:
        matches = rec_engine.search_catalog(title, limit=10)
        matched_idx = None
        for m in matches:
            if (
                m["title"].strip().lower() == title.strip().lower()
                and m["artist"].strip().lower() == artist.strip().lower()
            ):
                matched_idx = m["idx"]
                break
        if matched_idx is None and matches:
            matched_idx = matches[0]["idx"]
        if matched_idx is not None and matched_idx not in seed_indices:
            seed_indices.append(matched_idx)

    if not seed_indices:
        return []

    seed_set = set(seed_indices)
    active_seeds = seed_indices[:20]
    pool_size = min(rec_engine.n_songs - 1, 250)

    candidates: dict[int, dict] = {}
    for s_idx in active_seeds:
        res = rec_engine.recommend(seed_idx=s_idx, weights=weights, top_k=pool_size)
        for item in res["results"]:
            c_idx = item["idx"]
            if c_idx in seed_set:
                continue
            if c_idx not in candidates:
                candidates[c_idx] = {
                    "item": item,
                    "scores": [],
                    "lyrics": [],
                    "vibe": [],
                    "tune": [],
                    "reasons": [],
                }
            candidates[c_idx]["scores"].append(item["score"])
            candidates[c_idx]["lyrics"].append(item["breakdown"]["lyrics"])
            candidates[c_idx]["vibe"].append(item["breakdown"]["vibe"])
            candidates[c_idx]["tune"].append(item["breakdown"]["tune"])
            candidates[c_idx]["reasons"].extend(item["reasons"])

    ranked_results = []
    top_n = min(3, len(active_seeds))
    for entry in candidates.values():
        sorted_scores = sorted(entry["scores"], reverse=True)
        base_score = float(np.mean(sorted_scores[:top_n]))
        # Reward tracks that align with multiple songs in the playlist
        consistency_boost = 1.0 + 0.04 * (len(sorted_scores) - 1)
        final_score = min(0.99, base_score * consistency_boost)

        merged_item = dict(entry["item"])
        merged_item["score"] = final_score
        merged_item["breakdown"] = {
            "lyrics": float(np.mean(sorted(entry["lyrics"], reverse=True)[:top_n])),
            "vibe": float(np.mean(sorted(entry["vibe"], reverse=True)[:top_n])),
            "tune": float(np.mean(sorted(entry["tune"], reverse=True)[:top_n])),
        }
        merged_item["reasons"] = list(dict.fromkeys(entry["reasons"]))[:3]
        ranked_results.append(merged_item)

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results[:top_k]


@st.cache_resource
def load_engine() -> MusicRecommender:
    return MusicRecommender()


@st.cache_resource
def load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    model_name = detect_db_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


init_user_tables()
engine = load_engine()
spotify_urls = load_spotify_url_map()

if "profile_id" not in st.session_state:
    st.session_state.profile_id = None
if "profile_name" not in st.session_state:
    st.session_state.profile_name = None
if "active_playlist_id" not in st.session_state:
    st.session_state.active_playlist_id = None

if "selected_song_idx" not in st.session_state:
    st.session_state.selected_song_idx = None
if (
    st.session_state.selected_song_idx is not None
    and st.session_state.selected_song_idx >= engine.n_songs
):
    st.session_state.selected_song_idx = None

if "w_lyrics" not in st.session_state:
    st.session_state.w_lyrics = 35
if "w_vibe" not in st.session_state:
    st.session_state.w_vibe = 25
if "w_tune" not in st.session_state:
    st.session_state.w_tune = 25
if "w_creators" not in st.session_state:
    st.session_state.w_creators = 15


def apply_preset(lyrics: int, vibe: int, tune: int, creators: int) -> None:
    st.session_state.w_lyrics = lyrics
    st.session_state.w_vibe = vibe
    st.session_state.w_tune = tune
    st.session_state.w_creators = creators


def clear_selection() -> None:
    st.session_state.selected_song_idx = None


def import_and_select(query_text: str) -> None:
    embedder = load_embedder()
    added = ingest_online_query(query_text, model=embedder, limit=4)
    if added:
        load_engine.clear()
        load_spotify_url_map.clear()
        fresh_engine = load_engine()
        matches = fresh_engine.search_catalog(added[0]["title"], limit=5)
        if matches:
            st.session_state.selected_song_idx = matches[0]["idx"]
        st.toast(f"Added '{added[0]['title']}' by {added[0]['artist']}!", icon="✅")
        st.rerun()
    else:
        st.warning("No new lyrics found for that search (or the song is already in your library).")


def format_reason(raw_reason: str) -> str:
    if raw_reason.startswith("vibe: "):
        return "💫 Similar emotional vibe"
    if raw_reason.startswith("lyrics"):
        return "📝 Similar story & lyrics"
    if raw_reason.startswith("tune"):
        return "🥁 Similar sound & rhythm"
    if raw_reason.startswith("same "):
        who = raw_reason.split(": ", 1)[-1]
        return f"🎤 Also by {who}"
    return raw_reason


# --- Sidebar ---
with st.sidebar:
    st.header("👤 Your Profile & Playlists")

    if st.session_state.profile_id is None:
        with st.form("profile_form", clear_on_submit=False):
            u_input = st.text_input("Profile Name", placeholder="e.g., Abhishank")
            pin_input = st.text_input(
                "Optional PIN", type="password", placeholder="Leave blank if none"
            )
            submitted = st.form_submit_button("Sign In / Create Profile", use_container_width=True)
            if submitted:
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
        col_u, col_out = st.columns([2.6, 1.2])
        col_u.markdown(f"Signed in as **{st.session_state.profile_name}**")
        if col_out.button("Switch", use_container_width=True):
            st.session_state.profile_id = None
            st.session_state.profile_name = None
            st.session_state.active_playlist_id = None
            st.rerun()

        user_playlists = get_user_playlists(st.session_state.profile_id)
        if user_playlists:
            pl_map = {name: pid for pid, name in user_playlists}
            pl_names = list(pl_map.keys())
            default_idx = 0
            if st.session_state.active_playlist_id in pl_map.values():
                for idx, (_, pid) in enumerate(user_playlists):
                    if pid == st.session_state.active_playlist_id:
                        default_idx = idx
                        break

            chosen_pl_name = st.selectbox(
                "Active playlist (for saving & recommendations):",
                pl_names,
                index=default_idx,
            )
            st.session_state.active_playlist_id = pl_map[chosen_pl_name]

        with st.expander("Create New Playlist"):
            new_pl_name = st.text_input(
                "Playlist name", placeholder="e.g., Late Night Drive", key="new_pl_input"
            )
            if st.button("Create Playlist", use_container_width=True):
                ok, msg = create_playlist(st.session_state.profile_id, new_pl_name)
                if ok:
                    pls = get_user_playlists(st.session_state.profile_id)
                    st.session_state.active_playlist_id = pls[-1][0]
                    st.toast(msg, icon="✅")
                    st.rerun()
                else:
                    st.warning(msg)

        if st.session_state.active_playlist_id is not None:
            saved_tracks = get_playlist_songs(st.session_state.active_playlist_id)
            with st.expander(f"Saved Songs ({len(saved_tracks)})"):
                if not saved_tracks:
                    st.caption("Click '+ Save' on any song card to add it here.")
                else:
                    for s_title, s_artist in saved_tracks:
                        c_t, c_del = st.columns([4, 1])
                        c_t.caption(f"**{s_title}** — {s_artist}")
                        if c_del.button("✕", key=f"del_{s_title}_{s_artist}"):
                            remove_song_from_playlist(
                                st.session_state.active_playlist_id, s_title, s_artist
                            )
                            st.rerun()

            st.download_button(
                "Download Profile Backup (.json)",
                data=export_profile_json(
                    st.session_state.profile_id, st.session_state.profile_name
                ),
                file_name=f"{st.session_state.profile_name.lower()}_playlists.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    st.header("How do you want to discover?")
    mode = st.radio(
        "Choose a search style:",
        [
            "Find Songs Like One I Love",
            "Recommend From My Playlist",
            "Describe a Mood or Scene",
        ],
        index=0,
    )

    if mode in {"Find Songs Like One I Love", "Recommend From My Playlist"}:
        st.divider()
        st.subheader("Quick Presets")
        p1, p2, p3 = st.columns(3)
        p1.button(
            "Balanced",
            use_container_width=True,
            on_click=apply_preset,
            args=(35, 25, 25, 15),
        )
        p2.button(
            "Deep Feel",
            use_container_width=True,
            on_click=apply_preset,
            args=(45, 40, 10, 5),
        )
        p3.button(
            "Same Beat",
            use_container_width=True,
            on_click=apply_preset,
            args=(10, 20, 60, 10),
        )

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
    st.subheader("Add New / Top Chart Songs")
    import_query = st.text_input(
        "Import any song or artist online:",
        placeholder="e.g., Die With A Smile Bruno Mars",
    )
    if st.button("Fetch & Add to Library", use_container_width=True) and import_query.strip():
        with st.spinner("Fetching lyrics & audio features..."):
            import_and_select(import_query.strip())

    st.caption(f"Library size: **{engine.n_songs:,}** songs")

# --- Main Page ---
st.title("🎧 Smart Song Finder")
st.caption("Discover music by its story, emotional mood, sound, and creators.")

active_saved_set: set[tuple[str, str]] = set()
if st.session_state.active_playlist_id is not None:
    active_saved_set = {
        (t.strip().lower(), a.strip().lower())
        for t, a in get_playlist_songs(st.session_state.active_playlist_id)
    }

results = []

if mode == "Find Songs Like One I Love":
    search_query = st.text_input(
        "Search for a song or artist to start from",
        placeholder="Type a song title or artist (e.g., Blinding Lights, Nightwish, Mazzy Star)...",
    )

    if search_query.strip():
        matches = engine.search_catalog(search_query, limit=15)
        if matches:
            placeholder = "— Pick a song from the matches below —"
            options = {
                f"{m['title']} — {m['artist']}": m["idx"]
                for m in matches
            }

            def on_pick() -> None:
                choice = st.session_state.get("song_dropdown")
                if choice in options:
                    st.session_state.selected_song_idx = options[choice]

            st.selectbox(
                "Select a track:",
                [placeholder] + list(options.keys()),
                key="song_dropdown",
                on_change=on_pick,
            )
        else:
            col_msg, col_fetch = st.columns([3, 1.3])
            col_msg.info(f"'{search_query}' isn't in your local library yet.")
            if col_fetch.button("Import It Online Now", type="primary", use_container_width=True):
                with st.spinner(f"Fetching '{search_query}'..."):
                    import_and_select(search_query.strip())

    if st.session_state.selected_song_idx is None:
        st.info("Search and select a song above to see personalized recommendations.")
    else:
        data = engine.recommend(
            seed_idx=st.session_state.selected_song_idx,
            weights={
                "lyrics": lyrics_w,
                "vibe": vibe_w,
                "tune": tune_w,
                "creators": creators_w,
            },
            top_k=top_k,
        )
        current = data["seed"]
        seed_spotify_url = get_song_spotify_url(
            current["title"], current["artist"], spotify_urls
        )

        with st.container(border=True):
            st.caption("CURRENTLY MATCHING AGAINST")
            c1, c2, c3, c4 = st.columns([3.8, 0.65, 0.95, 0.85])
            c1.markdown(f"### {current['title']}\n**{current['artist']}**")
            with c2:
                st.write("")
                render_spotify_icon_button(seed_spotify_url)
            with c3:
                st.write("")
                if st.session_state.active_playlist_id is not None:
                    is_saved = (
                        current["title"].strip().lower(),
                        current["artist"].strip().lower(),
                    ) in active_saved_set
                    if st.button(
                        "Saved ✓" if is_saved else "+ Save",
                        key="save_current_song",
                        use_container_width=True,
                        disabled=is_saved,
                    ):
                        add_song_to_playlist(
                            st.session_state.active_playlist_id,
                            current["title"],
                            current["artist"],
                        )
                        st.toast(f"Saved '{current['title']}'!", icon="💾")
                        st.rerun()
            with c4:
                st.write("")
                st.button("Clear", use_container_width=True, on_click=clear_selection)

        results = data["results"]

elif mode == "Recommend From My Playlist":
    if st.session_state.profile_id is None or st.session_state.active_playlist_id is None:
        st.info("Sign in or create a profile in the sidebar to build playlists and get personalized recommendations.")
    else:
        pl_tracks = get_playlist_songs(st.session_state.active_playlist_id)
        if not pl_tracks:
            st.info(
                "Your active playlist is empty. Switch to 'Find Songs Like One I Love' or 'Describe a Mood or Scene' and click '+ Save' on songs you enjoy!"
            )
        else:
            st.success(f"Generating recommendations based on **{len(pl_tracks)}** saved song(s) in your active playlist.")
            results = recommend_from_playlist(
                rec_engine=engine,
                playlist_tracks=pl_tracks,
                weights={
                    "lyrics": lyrics_w,
                    "vibe": vibe_w,
                    "tune": tune_w,
                    "creators": creators_w,
                },
                top_k=top_k,
            )

else:
    prompt_text = st.text_input(
        "What kind of vibe, story, or moment are you looking for?",
        placeholder="e.g., driving alone on a rainy highway at midnight thinking about old memories",
    )
    if prompt_text.strip():
        data = engine.recommend_by_prompt(prompt=prompt_text, top_k=top_k)
        results = data["results"]
    else:
        st.info("Describe a mood, lyric theme, or scene above to find matching songs.")

if results:
    st.subheader("Recommended For You")

    for rank, item in enumerate(results, start=1):
        with st.container(border=True):
            col_song, col_why, col_btn = st.columns([3.2, 2.1, 1.3])

            with col_song:
                year_str = f" ({item['year']})" if item.get("year") else ""
                genre_clean = (item.get("genre") or "").strip()
                genre_str = f" • *{genre_clean.title()}*" if genre_clean else ""
                st.markdown(f"#### {rank}. {item['title']}")
                st.markdown(f"**{item['artist']}**{year_str}{genre_str}")
                friendly_tags = list(dict.fromkeys(format_reason(r) for r in item["reasons"]))
                st.write("  \n".join(friendly_tags))

            with col_why:
                match_pct = min(100, int(round(item["score"] * 100)))
                st.markdown(f"**{match_pct}% Match**")
                st.progress(match_pct / 100.0)
                b = item["breakdown"]
                st.caption(
                    f"Story: {int(b['lyrics']*100)}% • Mood: {int(b['vibe']*100)}% • "
                    f"Beat: {int(b['tune']*100)}%"
                )

            with col_btn:
                item_spotify_url = get_song_spotify_url(
                    item["title"], item["artist"], spotify_urls
                )
                b_sp, b_sv = st.columns(2)
                with b_sp:
                    render_spotify_icon_button(item_spotify_url)
                with b_sv:
                    if st.session_state.active_playlist_id is not None:
                        is_saved = (
                            item["title"].strip().lower(),
                            item["artist"].strip().lower(),
                        ) in active_saved_set
                        if st.button(
                            "✓" if is_saved else "+ Save",
                            key=f"save_{item['idx']}_{rank}",
                            use_container_width=True,
                            disabled=is_saved,
                        ):
                            add_song_to_playlist(
                                st.session_state.active_playlist_id,
                                item["title"],
                                item["artist"],
                            )
                            st.toast(f"Saved '{item['title']}'!", icon="💾")
                            st.rerun()
                    else:
                        st.button(
                            "+ Save",
                            key=f"save_disabled_{item['idx']}_{rank}",
                            use_container_width=True,
                            disabled=True,
                            help="Sign in to a profile in the sidebar to save songs",
                        )

                if st.button(
                    "More Like This",
                    key=f"more_{item['idx']}_{rank}",
                    use_container_width=True,
                ):
                    st.session_state.selected_song_idx = item["idx"]
                    st.rerun()