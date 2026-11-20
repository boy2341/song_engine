import os
import shutil
import sqlite3

DB_PATH = "music_engine.db"
BACKUP_PATH = "music_engine_full_backup.db"
TARGET_MAX_SONGS = 9200  # ~68-75 MB with full 768-dim float32 vectors

PROTECTED_ARTISTS = [
    "within temptation",
    "lacuna coil",
    "nightwish",
    "halestorm",
    "flyleaf",
    "delain",
    "breaking benjamin",
    "we are the fallen",
    "epica",
    "after forever",
    "mazzy star",
    "cigarettes after sex",
    "ethel cain",
    "weyes blood",
    "florence + the machine",
    "angel olsen",
    "phoebe bridgers",
    "hole",
    "twenty one pilots",
    "hayley williams",
    "boywithuke",
]


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def compact_database() -> None:
    if not os.path.exists(DB_PATH):
        print(f"Could not find {DB_PATH}.")
        return

    # Preserve the original unshrunk backup if it already exists
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print(f"Saved full local backup to: {BACKUP_PATH}")

    before_mb = file_size_mb(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    # 1. Remove exact duplicate (title, artist) rows
    conn.execute(
        """
        DELETE FROM songs
        WHERE song_id NOT IN (
            SELECT MIN(song_id)
            FROM songs
            GROUP BY LOWER(TRIM(title)), LOWER(TRIM(artist))
        )
        """
    )

    total_songs = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    print(f"Current size: {before_mb:.2f} MB across {total_songs:,} unique songs.")

    # 2. Prune only low-popularity older generic tracks if above TARGET_MAX_SONGS
    if total_songs > TARGET_MAX_SONGS:
        rows = conn.execute(
            """
            SELECT song_id, spotify_id, artist, genre, release_year, popularity
            FROM songs
            """
        ).fetchall()

        protected_ids: set[int] = set()
        prunable_rows: list[tuple[int, int, int]] = []

        for song_id, sp_id, artist, genre, year, pop in rows:
            sp_str = (sp_id or "").lower()
            art_str = (artist or "").lower()
            gen_str = (genre or "").lower()
            yr = int(year) if year else 0
            popularity = int(pop) if pop else 0

            is_custom_import = sp_str.startswith(("genre23_25_", "radio_", "online_"))
            is_fav_artist = any(fav in art_str for fav in PROTECTED_ARTISTS)
            is_recent = yr >= 2022
            is_core_genre = any(
                g in gen_str
                for g in ("hindi", "metal", "rock", "indie", "gothic", "symphonic", "grunge")
            )

            if is_custom_import or is_fav_artist or is_recent or is_core_genre:
                protected_ids.add(song_id)
            else:
                prunable_rows.append((popularity, yr, song_id))

        # Keep highest-popularity / newest general tracks up to TARGET_MAX_SONGS
        slots_left = max(0, TARGET_MAX_SONGS - len(protected_ids))
        prunable_rows.sort(reverse=True)
        keep_ids = protected_ids | {sid for _, _, sid in prunable_rows[:slots_left]}

        delete_ids = [(sid,) for _, _, sid in prunable_rows[slots_left:]]
        if delete_ids:
            conn.executemany("DELETE FROM songs WHERE song_id = ?", delete_ids)
            print(
                f"Trimmed {len(delete_ids):,} older low-popularity tracks "
                f"(kept all {len(protected_ids):,} priority/favorite-artist tracks)."
            )

    # 3. Clean up orphaned credits/creators and trim redundant lyric strings
    conn.execute("DELETE FROM song_credits WHERE song_id NOT IN (SELECT song_id FROM songs)")
    conn.execute("DELETE FROM creators WHERE creator_id NOT IN (SELECT DISTINCT creator_id FROM song_credits)")
    conn.execute("UPDATE songs SET lyrics = SUBSTR(lyrics, 1, 80) WHERE LENGTH(lyrics) > 80")
    conn.commit()

    # 4. Compact SQLite pages
    print("Running VACUUM to reclaim disk space...")
    conn.execute("VACUUM;")
    conn.execute("PRAGMA optimize;")
    final_count = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    conn.close()

    after_mb = file_size_mb(DB_PATH)
    print(f"New size: {after_mb:.2f} MB across {final_count:,} songs (saved {before_mb - after_mb:.2f} MB).")


if __name__ == "__main__":
    compact_database()