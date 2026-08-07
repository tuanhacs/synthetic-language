"""Bit-level tokenizer: vocabulary {0, 1, BOS, EOS, PAD}.

Tokenisation is deliberately bit-level. Codeword-level tokens would hand the
segmentation to the model for free, which is exactly the difficulty we study.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from .language import Sample


class BitTokenizer:
    """Maps bit strings to token ids and packs sentences into context windows."""

    BIT0, BIT1, BOS, EOS, PAD = 0, 1, 2, 3, 4
    ID_TO_TOKEN = {0: "0", 1: "1", 2: "<bos>", 3: "<eos>", 4: "<pad>"}

    @property
    def vocab_size(self) -> int:
        return len(self.ID_TO_TOKEN)

    @property
    def vocab(self) -> dict[str, int]:
        return {"0": self.BIT0, "1": self.BIT1, "<bos>": self.BOS, "<eos>": self.EOS, "<pad>": self.PAD}

    def encode(self, bits: str, bos: bool = True, eos: bool = True) -> list[int]:
        """``bits`` -> token ids, optionally wrapped in BOS/EOS."""
        ids = [self.BOS] if bos else []
        for ch in bits:
            if ch == "0":
                ids.append(self.BIT0)
            elif ch == "1":
                ids.append(self.BIT1)
            else:
                raise ValueError(f"not a bit: {ch!r}")
        if eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int], keep_special: bool = False) -> str:
        """Token ids -> string; special tokens are dropped unless asked for."""
        out = []
        for i in ids:
            token = self.ID_TO_TOKEN[i]
            if len(token) == 1 or keep_special:
                out.append(token)
        return "".join(out)

    def pack(
        self,
        samples: Iterable[Sample | str],
        context_len: int,
        drop_last: bool = True,
    ) -> Iterator[list[int]]:
        """Concatenate ``BOS + bits + EOS`` streams and cut fixed-size windows.

        Sentences are not aligned to window boundaries (as in the CFG paper), so
        a window may start mid-sentence. A trailing partial window is dropped
        (``drop_last=True``) or PAD-filled.
        """
        if context_len < 2:
            raise ValueError("context_len must be >= 2")
        buffer: list[int] = []
        for item in samples:
            bits = item if isinstance(item, str) else item.bits
            buffer.extend(self.encode(bits))
            while len(buffer) >= context_len:
                yield buffer[:context_len]
                del buffer[:context_len]
        if buffer and not drop_last:
            yield buffer + [self.PAD] * (context_len - len(buffer))

    def pack_all(self, samples: Sequence[Sample | str], context_len: int) -> list[list[int]]:
        return list(self.pack(samples, context_len))
