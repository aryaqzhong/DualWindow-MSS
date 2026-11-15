"""MUSDB18-HQ access: official train/valid split, memory-mapped stems, random-mix stream."""
from __future__ import annotations

import os

import numpy as np
import torch
from scipy.io import wavfile

SR = 44100
STEMS = ("vocals", "drums", "bass", "other")

# musdb's official validation split (sigsep-mus-db, musdb/configs/mus.yaml).
VALIDATION_TRACKS = (
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
)


def list_tracks(root: str, subset: str) -> list[str]:
    d = os.path.join(root, subset)
    return sorted(t for t in os.listdir(d) if os.path.isdir(os.path.join(d, t)))


def train_valid_split(root: str) -> tuple[list[str], list[str]]:
    tracks = list_tracks(root, "train")
    valid = [t for t in tracks if t in VALIDATION_TRACKS]
    missing = sorted(set(VALIDATION_TRACKS) - set(valid))
    if missing:
        raise RuntimeError(f"validation tracks not found under {root}/train: {missing}")
    return [t for t in tracks if t not in VALIDATION_TRACKS], valid


class StemStore:
    """Memory-maps the stereo int16 stems of MUSDB18-HQ tracks and returns mono float32."""

    def __init__(self, root: str, subset: str, tracks):
        self.dir = os.path.join(root, subset)
        self.tracks = list(tracks)
        self._maps = {}
        self.lengths = {t: min(self._map(t, s).shape[0] for s in STEMS) for t in self.tracks}
        self._maps = {}  # reopened lazily (e.g. inside each DataLoader worker)

    def _map(self, track: str, stem: str):
        key = (track, stem)
        mm = self._maps.get(key)
        if mm is None:
            sr, mm = wavfile.read(os.path.join(self.dir, track, f"{stem}.wav"), mmap=True)
            if sr != SR or mm.dtype != np.int16:
                raise ValueError(f"{track}/{stem}: expected {SR} Hz int16, got {sr} Hz {mm.dtype}")
            self._maps[key] = mm
        return mm

    def read(self, track: str, stem: str, start: int, n: int) -> np.ndarray:
        x = np.asarray(self._map(track, stem)[start:start + n], dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return x / 32768.0


def rms_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


class RandomMixStream(torch.utils.data.IterableDataset):
    """Endless stream of (mixture, vocals) mono crops.

    Augmentation: with probability ``p_remix`` every stem comes from a random track and
    offset (otherwise all stems share one track and offset); each stem gets a random gain
    in ``gain_db``; with probability ``p_bleed`` the accompaniment is attenuated by
    ``bleed_db`` so that vocal-dominant "live microphone" mixtures are also covered.
    Crops whose vocals are silent are mostly resampled.
    """

    def __init__(self, store: StemStore, seconds=3.0, seed=0, p_remix=0.5, gain_db=(-6.0, 3.0),
                 p_bleed=0.2, bleed_db=(6.0, 20.0), silent_vocal_db=-50.0, p_keep_silent=0.1):
        super().__init__()
        self.store = store
        self.n = int(round(seconds * SR))
        self.seed = int(seed)
        self.p_remix, self.gain_db = float(p_remix), tuple(gain_db)
        self.p_bleed, self.bleed_db = float(p_bleed), tuple(bleed_db)
        self.silent_vocal_db, self.p_keep_silent = float(silent_vocal_db), float(p_keep_silent)
        self.tracks = [t for t in store.tracks if store.lengths[t] > self.n]

    def _pick(self, rng):
        t = self.tracks[rng.integers(len(self.tracks))]
        return t, int(rng.integers(0, self.store.lengths[t] - self.n))

    def sample(self, rng):
        for _ in range(20):
            if rng.random() < self.p_remix:
                src = {s: self._pick(rng) for s in STEMS}
            else:
                t, off = self._pick(rng)
                src = {s: (t, off) for s in STEMS}
            stems = {
                s: self.store.read(src[s][0], s, src[s][1], self.n)
                * np.float32(10.0 ** (rng.uniform(*self.gain_db) / 20.0))
                for s in STEMS
            }
            if rms_db(stems["vocals"]) > self.silent_vocal_db or rng.random() < self.p_keep_silent:
                break
        voc = stems["vocals"]
        acc = stems["drums"] + stems["bass"] + stems["other"]
        if rng.random() < self.p_bleed:
            acc = acc * np.float32(10.0 ** (-rng.uniform(*self.bleed_db) / 20.0))
        mix = voc + acc
        return torch.from_numpy(mix.astype(np.float32)), torch.from_numpy(voc.astype(np.float32))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield self.sample(rng)


def load_tracks_mono(root: str, subset: str, tracks, max_seconds=None):
    """Full mono tracks as (name, mixture, vocals); mixture = sum of the four stems."""
    store = StemStore(root, subset, tracks)
    out = []
    for t in tracks:
        n = store.lengths[t] if max_seconds is None else min(store.lengths[t], int(max_seconds * SR))
        voc = store.read(t, "vocals", 0, n)
        acc = store.read(t, "drums", 0, n) + store.read(t, "bass", 0, n) + store.read(t, "other", 0, n)
        out.append((t, (voc + acc).astype(np.float32), voc.astype(np.float32)))
    return out
