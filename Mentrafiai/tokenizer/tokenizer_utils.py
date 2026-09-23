"""Thin wrapper around a trained SentencePiece model, with chat-turn helpers."""

import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()

        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()
        self.user_id = self.sp.piece_to_id("<user>")
        self.assistant_id = self.sp.piece_to_id("<assistant>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def encode_chat_turn(self, user_text: str, assistant_text: str):
        """
        Encode a single <user>...<assistant>... turn for SFT.
        Returns (input_ids, loss_mask) where loss_mask is 0 on the user
        portion (no loss) and 1 on the assistant portion (loss computed).
        """
        user_ids = [self.user_id] + self.encode(user_text)
        assistant_ids = [self.assistant_id] + self.encode(assistant_text) + [self.eos_id]

        input_ids = [self.bos_id] + user_ids + assistant_ids
        loss_mask = [0] * (1 + len(user_ids)) + [1] * len(assistant_ids)
        return input_ids, loss_mask
