"""Three-granularity dedup (report 9.2).

1. **Exact**        - sha256 over whitespace-normalized artifact text.
2. **Near-dup**     - MinHash over word 5-shingles with LSH banding;
                      Jaccard threshold 0.80 (configurable).
3. **Isomorphism**  - model-graph signature via ``prismacoder.hashing.
                      model_graph_signature``: same relation graph + field type
                      profile under renamed entities -> collision (prevents the
                      sampler emitting fifty isomorphic e-commerce schemas).

``Deduplicator`` is corpus-wide (not per-track) so a synthetic near-copy of a
mined schema cannot silently double.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Iterable

from prismacoder.constants import (
    DEDUP_ISOMORPHISM,
    DEDUP_MINHASH_JACCARD,
    MINHASH_PERMUTATIONS,
    MINHASH_SHINGLE_K,
)
from prismacoder.hashing import exact_hash, model_graph_signature, shingles


# --- MinHash ----------------------------------------------------------------
def _hash64(data: str, salt: int) -> int:
    h = hashlib.blake2b(data.encode("utf-8"), digest_size=8, key=str(salt).encode())
    return struct.unpack("<Q", h.digest())[0]


def minhash_signature(
    text: str, num_perm: int = MINHASH_PERMUTATIONS, k: int = MINHASH_SHINGLE_K
) -> list[int]:
    sh = shingles(text, k)
    if not sh:
        return [0] * num_perm
    return [min(_hash64(s, salt) for s in sh) for salt in range(num_perm)]


def lsh_bands(sig: list[int], bands: int = 32) -> list[str]:
    """Split a MinHash signature into LSH bands; each band hashes to a bucket key."""
    rows = max(1, len(sig) // bands)
    keys = []
    for b in range(bands):
        chunk = sig[b * rows : (b + 1) * rows]
        keys.append(hashlib.md5(str(chunk).encode()).hexdigest()[:16])
    return keys


def estimated_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    eq = sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
    return eq / len(sig_a)


@dataclass
class DedupStats:
    seen: int = 0
    exact_dupes: int = 0
    near_dupes: int = 0
    isomorphic_dupes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "exact_dupes": self.exact_dupes,
            "near_dupes": self.near_dupes,
            "isomorphic_dupes": self.isomorphic_dupes,
            "kept": self.seen - self.exact_dupes - self.near_dupes - self.isomorphic_dupes,
        }


@dataclass
class Deduplicator:
    """Streaming dedup: feed records via ``consider()``; kept ones come back."""

    jaccard_threshold: float = DEDUP_MINHASH_JACCARD
    use_isomorphism: bool = DEDUP_ISOMORPHISM
    text_key: str = "dedup_text"  # record key holding the canonical text
    stats: DedupStats = dc_field(default_factory=DedupStats)
    _exact: set[str] = dc_field(default_factory=set)
    _lsh: dict[str, set[str]] = dc_field(default_factory=dict)  # band key -> ids
    _sigs: dict[str, list[int]] = dc_field(default_factory=dict)
    _iso: set[str] = dc_field(default_factory=set)

    def consider(self, rec: dict[str, Any]) -> bool:
        """True -> record is novel, keep it. False -> duplicate, drop."""
        self.stats.seen += 1
        rid = rec.get("id") or f"rec-{self.stats.seen:08d}"

        text = rec.get(self.text_key) or rec.get("schema") or rec.get("reference") or ""
        if not text:
            return True  # nothing to compare on; keep (downstream gates decide)

        eh = exact_hash(text)
        if eh in self._exact:
            self.stats.exact_dupes += 1
            return False
        self._exact.add(eh)

        if self.use_isomorphism and rec.get("schema"):
            iso = model_graph_signature(rec["schema"])
            if iso and iso in self._iso:
                self.stats.isomorphic_dupes += 1
                return False
            if iso:
                self._iso.add(iso)

        sig = minhash_signature(text)
        bands = lsh_bands(sig)
        for key in bands:
            for other_id in self._lsh.get(key, ()):  # candidate set
                other_sig = self._sigs.get(other_id)
                if other_sig and estimated_jaccard(sig, other_sig) >= self.jaccard_threshold:
                    self.stats.near_dupes += 1
                    return False
        for key in bands:
            self._lsh.setdefault(key, set()).add(rid)
        self._sigs[rid] = sig
        return True

    def dedup(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in records if self.consider(r)]
