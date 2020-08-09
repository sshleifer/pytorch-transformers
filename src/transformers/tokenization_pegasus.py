from typing import Dict, List, Optional

from transformers.tokenization_reformer import ReformerTokenizer

from .tokenization_utils_base import BatchEncoding


class PegasusTokenizer(ReformerTokenizer):
    offset = 103  # to make embedding size a multiple of 128 I think
    vocab_files_names = {"vocab_file": "spiece.model"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dont use reserved words added_token_encoder, added_tokens_decoder because they
        # get checked in from_pretrained
        assert len(self.added_tokens_decoder) == 0
        self.encoder: Dict[int, str] = {0: self.pad_token, 1: self.eos_token}
        self.encoder.update({i: f"unk_{i}" for i in range(2, self.offset + 2)})
        # self.added_tokens_decoder = self._added_toks
        self.decoder: Dict[str, int] = {v: k for k, v in self.encoder.items()}
        assert 104 in self.encoder
        assert 105 not in self.encoder
        assert self.pad_token_id == 0, "pad should be 0"
        assert self.eos_token_id == 1, "eos should be 1"
        assert self.unk_token_id != 2

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """

        if token in self.decoder:
            return self.decoder[token]
        elif token in self.added_tokens_decoder:
            return self.added_tokens_decoder[token]
        sp_id = self.sp_model.piece_to_id(token)
        return sp_id + self.offset

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.encoder:
            return self.encoder[index]
        elif index in self.added_tokens_encoder:
            return self.added_tokens_encoder[index]
        else:
            # assert index > self.offset, f"cannot decode ids between 2 and {self.offset}. Got {index}"
            token = self.sp_model.IdToPiece(index - self.offset)
        return token

    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + self.offset

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def num_special_tokens_to_add(self, pair=False):
        """Just EOS"""
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
        assert all_special_ids == set([0, 1])
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
        truncation=True,
        padding="longest",
    ) -> BatchEncoding:
        """Prepare model inputs for translation. For best performance, translate one sentence at a time.
        Arguments:
            src_texts: list of src language texts
            tgt_texts: list of tgt language texts
            max_length: (None) defer to config (1024 for mbart-large-en-ro)
            pad_to_max_length: (bool)
            return_tensors: (str) default "pt" returns pytorch tensors, pass None to return lists.

        Returns:
            BatchEncoding: with keys [input_ids, attention_mask, decoder_input_ids,  decoder_attention_mask]
            all shaped bs, seq_len. (BatchEncoding is a dict of string -> tensor or lists).
            If no tgt_text is specified, the only keys will be input_ids and attention_mask.
        """
        if "" in src_texts:
            raise ValueError(f"found empty string in src_texts: {src_texts}")

        tokenizer_kwargs = dict(
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            truncation=truncation,
            padding=padding,
        )
        model_inputs: BatchEncoding = self(src_texts, **tokenizer_kwargs)
        if tgt_texts is None:
            return model_inputs
        if max_target_length is not None:
            tokenizer_kwargs["max_length"] = max_target_length
        tgt_texts = ["<pad> " + x for x in tgt_texts]
        decoder_inputs: BatchEncoding = self(tgt_texts, **tokenizer_kwargs)
        for k, v in decoder_inputs.items():
            model_inputs[f"decoder_{k}"] = v
        return model_inputs
