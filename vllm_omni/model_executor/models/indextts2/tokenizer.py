# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from transformers import PreTrainedTokenizer
from vllm.logger import init_logger

from vllm_omni.model_executor.models.indextts2.utils.front import TextNormalizer, TextTokenizer

logger = init_logger(__name__)


class IndexTTS2Tokenizer(PreTrainedTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        vocab_file = kwargs.pop("vocab_file", None)
        if vocab_file is None:
            vocab_file = os.path.join(pretrained_model_name_or_path, "bpe.model")
        kwargs.setdefault("model_dir", pretrained_model_name_or_path)
        logger.info(f"IndexTTS2Tokenizer.from_pretrained: {pretrained_model_name_or_path}")
        return cls(vocab_file, **kwargs)

    def __init__(self, vocab_file: str, **kwargs):
        logger.info(f"IndexTTS2Tokenizer.__init__: vocab_file={vocab_file}")
        self.vocab_file = vocab_file
        model_dir = kwargs.pop("model_dir", None)
        enable_glossary = kwargs.pop("enable_glossary", True)
        normalizer = TextNormalizer(enable_glossary=enable_glossary)
        glossary_path = None
        if model_dir is not None:
            glossary_path = os.path.join(model_dir, "glossary.yaml")
        elif vocab_file:
            glossary_path = os.path.join(os.path.dirname(vocab_file), "glossary.yaml")
        if enable_glossary and glossary_path and os.path.exists(glossary_path):
            normalizer.load_glossary_from_yaml(glossary_path)
            logger.info("IndexTTS2Tokenizer loaded glossary from %s", glossary_path)
        self._tok = TextTokenizer(vocab_file, normalizer=normalizer)
        logger.info("IndexTTS2Tokenizer initialized successfully")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self._tok.vocab_size

    @property
    def max_token_id(self):
        return self.vocab_size - 1

    @property
    def max_chars_per_token(self):
        return max(len(tok) for tok in self._tok.get_vocab())

    def get_vocab(self):
        return self._tok.get_vocab()

    def _tokenize(self, text):
        return self._tok.tokenize(text)

    def _convert_token_to_id(self, token):
        return self._tok.convert_tokens_to_ids(token)[0]

    def convert_tokens_to_string(self, tokens):
        return self._tok.decode(self._tok.convert_tokens_to_ids(tokens))
