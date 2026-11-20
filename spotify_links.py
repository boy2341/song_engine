import re
import sqlite3
import time
import urllib.parse
import requests

DB_PATH = "music_engine.db"
SPOTIFY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{22}$")

# Optional: Leave blank for instant zero-API-key links, or add free Spotify Developer
# credentials to resolve exact /track/ URLs for Deezer/Apple-imported songs too.
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""


def build_spotify_url(spotify_id: str | None, title: str, artist: str) -> str:
    """Return a direct /track/ URL if a valid Spotify ID exists, else a Spotify search URL."""
    clean_id = (spotify_id or "").strip()
    if SPOTIFY_ID_PATTERN.fullmatch(clean_id):
        return f"https://open.spotify.com/track/{clean_id}"
    query = urllib.parse.quote(f"{title} {artist}".strip())
    return f"https://open.spotify.com/search/{query}"


def get_spotify_token(client_id: str, client_secret: str) -> str | None:
    if not client_id or not client_secret:
        return None
    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except Exception:
        pass
    return None


def search_exact_spotify_track(title: str, artist: str, token: str) -> tuple[str | None, str | None]:
    try:
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            params={"q": f"track:{title} artist:{artist}", "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        if resp.status_code == 200:
            items = resp.json().get("tracks", {}).get("items", [])
            if items:
                tid = items[0].get("id")
                turl = items[0].get("external_urls", {}).get("spotify")
                return tid, turl
    except Exception:
        pass
    return None, None


def populate_spotify_links() -> None:
    conn = sqlite3.connect(DB_PATH)

    # Add spotify_url column if it does not exist yet
    cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    if "spotify_url" not in cols:
        conn.execute("ALTER TABLE songs ADD COLUMN spotify_url TEXT;")
        conn.commit()

    rows = conn.execute(
        "SELECT song_id, spotify_id, title, artist FROM songs"
    ).fetchall()

    token = get_spotify_token(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    direct_count = 0
    search_count = 0
    updates: list[tuple[str, str | None, int]] = []

    for song_id, sp_id, title, artist in rows:
        clean_id = (sp_id or "").strip()
        if SPOTIFY_ID_PATTERN.fullmatch(clean_id):
            url = f"https://open.spotify.com/track/{clean_id}"
            direct_count += 1
            updates.append((url, clean_id, song_id))
        elif token:
            found_id, found_url = search_exact_spotify_track(title, artist, token)
            if found_id and found_url:
                direct_count += 1
                updates.append((found_url, found_id, song_id))
            else:
                search_count += 1
                updates.append((build_spotify_url(None, title, artist), sp_id, song_id))
            time.sleep(0.08)
        else:
            search_count += 1
            updates.append((build_spotify_url(None, title, artist), sp_id, song_id))

    conn.executemany(
        "UPDATE songs SET spotify_url = ?, spotify_id = ? WHERE song_id = ?",
        updates,
    )
    conn.commit()
    conn.close()

    print(f"Updated {len(updates):,} songs in {DB_PATH}:")
    print(f"  - Direct Spotify track links (/track/...): {direct_count:,}")
    print(f"  - Spotify search links (/search/...):      {search_count:,}")


if __name__ == "__main__":
    populate_spotify_links()