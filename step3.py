import time
import numpy as np

from step1 import get_connection


class MusicRecommender:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        conn = get_connection()

        rows = conn.execute(
            """
            SELECT song_id, title, artist, genre, release_year, popularity,
                   tempo_bpm, energy, acousticness, valence, danceability, musical_mode,
                   primary_vibe, secondary_vibe, lyrics_vec, vibe_vec
            FROM songs
            WHERE lyrics_vec IS NOT NULL AND vibe_vec IS NOT NULL
            ORDER BY popularity DESC, song_id ASC
            """
        ).fetchall()

        if not rows:
            conn.close()
            raise RuntimeError("No embedded songs found. Run step1.py and step2.py first.")

        self.n_songs = len(rows)
        self.song_ids: list[int] = []
        self.titles: list[str] = []
        self.artists: list[str] = []
        self.genres: list[str] = []
        self.years: list[int | None] = []
        self.popularities: list[int] = []
        self.primary_vibes: list[str] = []
        self.secondary_vibes: list[str] = []
        self.title_to_idx: dict[str, int] = {}

        lyrics_list: list[np.ndarray] = []
        vibe_list: list[np.ndarray] = []
        tune_list: list[list[float]] = []

        for idx, r in enumerate(rows):
            (
                song_id, title, artist, genre, year, pop,
                bpm, energy, acousticness, valence, danceability, mode,
                p_vibe, s_vibe, lyrics_blob, vibe_blob
            ) = r

            self.song_ids.append(song_id)
            self.titles.append(title)
            self.artists.append(artist)
            self.genres.append(genre or "Unknown")
            self.years.append(year)
            self.popularities.append(pop or 0)
            self.primary_vibes.append(p_vibe or "")
            self.secondary_vibes.append(s_vibe or "")

            key = title.strip().lower()
            if key not in self.title_to_idx:
                self.title_to_idx[key] = idx

            lyrics_list.append(np.frombuffer(lyrics_blob, dtype=np.float32))
            vibe_list.append(np.frombuffer(vibe_blob, dtype=np.float32))

            mode_val = float(mode) * 0.5 if mode is not None else 0.25
            tune_list.append([bpm / 200.0, energy, acousticness, valence, danceability, mode_val])

        self.lyrics_matrix = np.vstack(lyrics_list).astype(np.float32)
        self.vibe_matrix = np.vstack(vibe_list).astype(np.float32)
        self.tune_matrix = np.array(tune_list, dtype=np.float32)
        self.embed_dim = self.lyrics_matrix.shape[1]
        self._text_model = None

        # Build sparse creator graph (performers now; composers/lyricists automatically work later)
        id_to_idx = {sid: idx for idx, sid in enumerate(self.song_ids)}
        credit_rows = conn.execute(
            """
            SELECT sc.song_id, c.creator_id, c.name, sc.role
            FROM song_credits sc
            JOIN creators c ON sc.creator_id = c.creator_id
            """
        ).fetchall()
        conn.close()

        self.song_credits: dict[int, list[tuple[int, str, str]]] = {}
        self.creator_songs: dict[int, list[tuple[int, str, str]]] = {}

        for sid, cid, cname, role in credit_rows:
            s_idx = id_to_idx.get(sid)
            if s_idx is None:
                continue
            self.song_credits.setdefault(s_idx, []).append((cid, cname, role))
            self.creator_songs.setdefault(cid, []).append((s_idx, cname, role))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"Engine ready: {self.n_songs:,} songs ({self.embed_dim}-dim) loaded in {elapsed_ms:.1f} ms.")

    def _get_text_model(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = "BAAI/bge-base-en-v1.5" if self.embed_dim == 768 else "all-MiniLM-L6-v2"
            self._text_model = SentenceTransformer(model_name)
        return self._text_model

    def search_catalog(self, query: str, limit: int = 12) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        exact, prefix, partial = [], [], []
        for idx, (t, a) in enumerate(zip(self.titles, self.artists)):
            tl, al = t.lower(), a.lower()
            item = {
                "idx": idx,
                "title": t,
                "artist": a,
                "genre": self.genres[idx],
                "year": self.years[idx],
                "vibe": self.primary_vibes[idx],
            }
            if tl == q:
                exact.append(item)
            elif tl.startswith(q):
                prefix.append(item)
            elif q in tl or q in al:
                partial.append(item)
            if len(exact) + len(prefix) + len(partial) >= limit * 3:
                break
        return (exact + prefix + partial)[:limit]

    def find_seed_index(self, title: str, artist: str | None = None) -> int:
        t_clean = title.strip().lower()
        if artist:
            a_clean = artist.strip().lower()
            for idx, (t, a) in enumerate(zip(self.titles, self.artists)):
                if t.lower() == t_clean and a_clean in a.lower():
                    return idx
        if t_clean in self.title_to_idx:
            return self.title_to_idx[t_clean]
        # Fallback to highest-popularity substring match
        matches = self.search_catalog(title, limit=1)
        if matches:
            return matches[0]["idx"]
        raise ValueError(f"Track '{title}' not found in embedded songs.")

    def recommend(
        self,
        seed_title: str | None = None,
        seed_artist: str | None = None,
        seed_idx: int | None = None,
        weights: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> dict:
        if seed_idx is None:
            if not seed_title:
                raise ValueError("Provide either seed_title or seed_idx.")
            seed_idx = self.find_seed_index(seed_title, seed_artist)

        t0 = time.perf_counter()
        raw_w = weights or {"lyrics": 0.35, "vibe": 0.25, "tune": 0.25, "creators": 0.15}
        w_sum = sum(raw_w.values()) or 1.0
        w = {k: v / w_sum for k, v in raw_w.items()}

        # 1. Lyric Semantic Similarity
        raw_lyric = self.lyrics_matrix @ self.lyrics_matrix[seed_idx]
        lyric_scores = np.clip((raw_lyric - 0.30) / 0.60, 0.0, 1.0)

        # 2. Emotional Vibe Similarity
        raw_vibe = self.vibe_matrix @ self.vibe_matrix[seed_idx]
        vibe_scores = np.clip((raw_vibe - 0.30) / 0.60, 0.0, 1.0)
        same_vibe = np.array(
            [1.0 if v == self.primary_vibes[seed_idx] else 0.0 for v in self.primary_vibes],
            dtype=np.float32,
        )
        vibe_scores = np.clip(vibe_scores + 0.12 * same_vibe, 0.0, 1.0)

        # 3. Tune DNA Similarity
        diff = self.tune_matrix - self.tune_matrix[seed_idx]
        sq_dist = np.sum(diff * diff, axis=1)
        tune_scores = np.exp(-2.5 * sq_dist)

        # 4. Creator Graph Similarity
        creator_scores = np.zeros(self.n_songs, dtype=np.float32)
        creator_notes: dict[int, list[str]] = {}
        role_weights = {"composer": 0.60, "lyricist": 0.60, "writer": 0.50, "performer": 0.45}

        for cid, cname, _seed_role in self.song_credits.get(seed_idx, []):
            for cand_idx, _, cand_role in self.creator_songs.get(cid, []):
                if cand_idx == seed_idx:
                    continue
                bonus = role_weights.get(cand_role, 0.40)
                creator_scores[cand_idx] = min(1.0, creator_scores[cand_idx] + bonus)
                creator_notes.setdefault(cand_idx, []).append(f"same {cand_role}: {cname}")

        total = (
            w["lyrics"] * lyric_scores
            + w["vibe"] * vibe_scores
            + w["tune"] * tune_scores
            + w["creators"] * creator_scores
        )
        total[seed_idx] = -1.0

        top_indices = np.argsort(total)[::-1][:top_k]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        for idx in top_indices:
            idx = int(idx)
            reasons = list(dict.fromkeys(creator_notes.get(idx, [])))
            if self.primary_vibes[idx] == self.primary_vibes[seed_idx]:
                reasons.append(f"vibe: {self.primary_vibes[idx]}")
            if lyric_scores[idx] >= 0.55:
                reasons.append(f"lyrics ({lyric_scores[idx]:.2f})")
            if tune_scores[idx] >= 0.80:
                bpm = int(round(self.tune_matrix[idx, 0] * 200))
                reasons.append(f"tune ({bpm} BPM)")
            if not reasons:
                reasons.append(f"vibe: {self.primary_vibes[idx]}")

            results.append(
                {
                    "idx": idx,
                    "title": self.titles[idx],
                    "artist": self.artists[idx],
                    "genre": self.genres[idx],
                    "year": self.years[idx],
                    "vibe": self.primary_vibes[idx],
                    "bpm": int(round(self.tune_matrix[idx, 0] * 200)),
                    "score": round(float(total[idx]), 4),
                    "breakdown": {
                        "lyrics": round(float(lyric_scores[idx]), 3),
                        "vibe": round(float(vibe_scores[idx]), 3),
                        "tune": round(float(tune_scores[idx]), 3),
                        "creators": round(float(creator_scores[idx]), 3),
                    },
                    "reasons": reasons,
                }
            )

        seed_info = {
            "idx": seed_idx,
            "title": self.titles[seed_idx],
            "artist": self.artists[seed_idx],
            "genre": self.genres[seed_idx],
            "year": self.years[seed_idx],
            "vibe": self.primary_vibes[seed_idx],
            "bpm": int(round(self.tune_matrix[seed_idx, 0] * 200)),
        }
        return {"seed": seed_info, "latency_ms": round(elapsed_ms, 2), "results": results}

    def recommend_by_prompt(self, prompt: str, top_k: int = 10) -> dict:
        t0 = time.perf_counter()
        model = self._get_text_model()
        q_vec = model.encode([prompt], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

        raw_lyric = self.lyrics_matrix @ q_vec
        lyric_scores = np.clip((raw_lyric - 0.25) / 0.65, 0.0, 1.0)
        raw_vibe = self.vibe_matrix @ q_vec
        vibe_scores = np.clip((raw_vibe - 0.25) / 0.65, 0.0, 1.0)

        total = 0.65 * lyric_scores + 0.35 * vibe_scores
        top_indices = np.argsort(total)[::-1][:top_k]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        for idx in top_indices:
            idx = int(idx)
            results.append(
                {
                    "idx": idx,
                    "title": self.titles[idx],
                    "artist": self.artists[idx],
                    "genre": self.genres[idx],
                    "year": self.years[idx],
                    "vibe": self.primary_vibes[idx],
                    "bpm": int(round(self.tune_matrix[idx, 0] * 200)),
                    "score": round(float(total[idx]), 4),
                    "breakdown": {
                        "lyrics": round(float(lyric_scores[idx]), 3),
                        "vibe": round(float(vibe_scores[idx]), 3),
                        "tune": 0.0,
                        "creators": 0.0,
                    },
                    "reasons": [f"vibe: {self.primary_vibes[idx]}", f"theme match ({lyric_scores[idx]:.2f})"],
                }
            )
        return {"query": prompt, "latency_ms": round(elapsed_ms, 2), "results": results}


if __name__ == "__main__":
    engine = MusicRecommender()
    out = engine.recommend(seed_idx=0)
    print(f"Seed: {out['seed']['title']} — {out['seed']['artist']} ({out['latency_ms']} ms)")
    for r in out["results"]:
        print(f"  {r['score']:.3f} | {r['title']} — {r['artist']} ({', '.join(r['reasons'])})")