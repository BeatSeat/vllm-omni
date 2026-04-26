# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight text frontend aligned with the official GLM-TTS preprocessing."""

from __future__ import annotations

import re

import emoji
import inflect

try:
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
except ImportError:
    ZhNormalizer = None

try:
    from tn.english.normalizer import Normalizer as EnNormalizer
except ImportError:
    EnNormalizer = None


CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
PUNCTUATION_CHARS = r"。？！；：、.?!;:，,"


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_CHAR_PATTERN.search(text))


def markdown_norm(markdown_text: str) -> str:
    markdown_text = re.sub(r"^(\d+)\. ", r"\1。", markdown_text)
    return markdown_text.replace("\\n", "\n")


def multi_line_process(plain_text: str) -> str:
    lines: list[str] = []
    for line in plain_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[-1] not in {".", "!", "?", ";", ":", "：", "。", "！", "？", "；", "，"}:
            line = f"{line}{'。' if contains_chinese(line) else '. '}"
        lines.append(line)
    return "".join(lines) if contains_chinese("".join(lines)) else " ".join(lines)


def remove_bracket(text: str, lang: str = "zh") -> str:
    brackets_to_remove = [
        ("(", ")"),
        ("（", "）"),
        ("【", "】"),
        ("「", "」"),
        ("`", "`"),
        ("《", "》"),
        ("『", "』"),
        ("{", "}"),
        ("[", "]"),
    ]
    if lang != "en":
        brackets_to_remove.append(("‘", "’"))
    for left, right in brackets_to_remove:
        text = text.replace(left, "").replace(right, "")
    return text


def replace_asterisk_with_multiply(text: str, lang: str) -> str:
    if lang == "zh":
        return re.sub(r"(?<=\d)\*(?=\d)", "乘", text)
    return re.sub(r"(?<=\d)\*(?=\d)", " times ", text)


def spell_out_number(text: str, inflect_parser: inflect.engine) -> str:
    new_text: list[str] = []
    start = None
    for idx, char in enumerate(text):
        if not char.isdigit():
            if start is not None:
                new_text.append(inflect_parser.number_to_words(text[start:idx]))
                start = None
            new_text.append(char)
        elif start is None:
            start = idx
    if start is not None and start < len(text):
        new_text.append(inflect_parser.number_to_words(text[start:]))
    return "".join(new_text)


def replace_space(text: str) -> str:
    alphanumeric_pattern = r"[a-zA-Z0-9]"
    punctuation_pattern = r"[.,!?;:]"
    text = re.sub(r" +", " ", text)
    result = ""
    idx = 0
    while idx < len(text):
        current_char = text[idx]
        if current_char != " ":
            result += current_char
            idx += 1
            continue
        prev_char = text[idx - 1] if idx > 0 else ""
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if re.match(alphanumeric_pattern, prev_char) and re.match(alphanumeric_pattern, next_char):
            result += " "
        elif re.match(punctuation_pattern, prev_char) and re.match(alphanumeric_pattern, next_char):
            result += " "
        idx += 1
    return result


def normalize_punctuation(text: str, punctuation_chars: str) -> str:
    text = replace_space(text)
    text = re.sub(rf"([{punctuation_chars}])\1+", r"\1", text)
    text = re.sub(rf"([{punctuation_chars}])(?=[{punctuation_chars}])", "", text)
    text = text.replace("#", "")
    text = text.replace("！", "。")
    text = text.replace("!", ".")
    return text


def ensure_proper_ending(text: str) -> str:
    if not text:
        return text
    if text[-1] in "？?":
        return text
    if text[-1] in PUNCTUATION_CHARS:
        return f"{text[:-1]}{'。' if contains_chinese(text) else '.'}"
    return f"{text}{'。' if contains_chinese(text) else '.'}"


class GLMTTSTextFrontend:
    """Subset of the official GLM-TTS text frontend used by the adapter."""

    def __init__(self) -> None:
        self.inflect_parser = inflect.engine()
        self.zh_tn_model = (
            ZhNormalizer(
                remove_erhua=False,
                full_to_half=True,
                remove_interjections=False,
                overwrite_cache=True,
            )
            if ZhNormalizer is not None
            else None
        )
        self.en_tn_model = EnNormalizer() if EnNormalizer is not None else None

    def text_normalize(self, text: str | None) -> str | None:
        if text is None:
            return None
        text = self._preprocess_text(text)
        if contains_chinese(text):
            text = self._normalize_chinese_text(text).lower()
        else:
            text = self._normalize_english_text(text)
        text = normalize_punctuation(text, PUNCTUATION_CHARS)
        return ensure_proper_ending(text)

    def _preprocess_text(self, text: str) -> str:
        text = markdown_norm(text)
        text = multi_line_process(text)
        text = emoji.replace_emoji(text, replace="")
        return re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text)

    def _normalize_chinese_text(self, text: str) -> str:
        if self.zh_tn_model is not None:
            text = self.zh_tn_model.normalize(text)
        text = remove_bracket(text)
        text = replace_asterisk_with_multiply(text, "zh")
        text = text.replace(" - ", "，")
        text = text.replace("——", "，")
        text = re.sub(r"[,:：;；、]+", "，", text)
        text = re.sub(r"[.…]+", "。", text)
        text = re.sub(r"[_·]+", "", text)
        text = re.sub(r"""['"‘’“”|]+""", "", text)
        return text.strip()

    def _normalize_english_text(self, text: str) -> str:
        text = text.replace("'", "’")
        if self.en_tn_model is not None:
            text = self.en_tn_model.normalize(text)
        text = remove_bracket(text, "en")
        text = replace_asterisk_with_multiply(text, "en")
        text = text.replace("—", " ")
        text = text.replace("’", "'")
        text = spell_out_number(text, self.inflect_parser)
        text = re.sub(r"\s+", " ", text)
        keep_punctuation = r"\.,!\?'\:;"
        pattern = rf"[^\w\s{keep_punctuation}]"
        text = re.sub(pattern, "", text)
        text = text.lower()
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"!+", "!", text)
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"'+", "'", text)
        text = re.sub(r":+", ":", text)
        text = re.sub(r";+", ";", text)
        text = re.sub(r"\s*([.,?!':;])\s*", r"\1 ", text)
        return text.strip()
