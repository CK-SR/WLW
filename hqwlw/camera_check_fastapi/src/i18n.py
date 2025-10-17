"""Lightweight internationalization utilities for the FastAPI service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class I18NConfig:
    """Configuration parameters for language resolution."""

    default_language: str = "en"
    fallback_language: str = "en"
    supported_languages: tuple[str, ...] = ("en",)


class LanguageManager:
    """Simple dictionary based translation helper."""

    def __init__(
        self,
        translations: Mapping[str, Mapping[str, str]],
        config: I18NConfig | None = None,
    ) -> None:
        self._translations: Dict[str, Dict[str, str]] = {
            lang: dict(values) for lang, values in translations.items()
        }
        self._config = config or I18NConfig()
        for lang in self._config.supported_languages:
            self._translations.setdefault(lang, {})
        if self._config.default_language not in self._translations:
            object.__setattr__(self._config, "default_language", self._config.fallback_language)
        if self._config.fallback_language not in self._translations:
            object.__setattr__(self._config, "fallback_language", "en")

    @property
    def supported_languages(self) -> set[str]:
        return set(self._translations)

    def _resolve_language(self, language: str | None) -> str:
        if language and language in self._translations:
            return language
        if self._config.default_language in self._translations:
            return self._config.default_language
        return self._config.fallback_language

    def translate(self, key: str, language: str | None = None, **fmt: Any) -> str:
        lang = self._resolve_language(language)
        translation = self._translations.get(lang, {}).get(key)
        if translation is None:
            translation = self._translations.get(self._config.fallback_language, {}).get(key, key)
        return translation.format(**fmt)

    def bundle(self, key: str, **fmt: Any) -> Dict[str, str]:
        """Return all translations for a key."""

        bundle: Dict[str, str] = {}
        for lang, mapping in self._translations.items():
            template = mapping.get(key)
            if template is not None:
                bundle[lang] = template.format(**fmt)
        if self._config.fallback_language not in bundle:
            fallback = self._translations.get(self._config.fallback_language, {}).get(key)
            if fallback is not None:
                bundle[self._config.fallback_language] = fallback.format(**fmt)
        return bundle

    def message(self, key: str, **fmt: Any) -> Dict[str, Any]:
        translations = self.bundle(key, **fmt)
        return {
            "msg": translations.get(self._config.fallback_language, key),
            "i18n": translations,
        }


DEFAULT_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "service.started": "Service started",
        "service.already_started": "Service already started",
        "service.stopping": "Service stopping...",
        "service.stopped": "Service stopped and resources released",
        "service.already_stopped": "Service already stopped",
        "error.service_not_started": "Service not started",
        "error.streams_not_list": "streams must be a list",
        "state.normal": "Normal",
        "state.black_screen": "Black Screen",
        "state.occluded": "Occluded",
        "state.tampered": "Tampered (Violent Motion)",
        "state.error": "Error",
        "error.decoding_failed": "Decoding failed",
    },
    "zh-CN": {
        "service.started": "服务已启动",
        "service.already_started": "服务已启动",
        "service.stopping": "服务停止中...",
        "service.stopped": "服务已停止并释放资源",
        "service.already_stopped": "服务已停止",
        "error.service_not_started": "服务尚未启动",
        "error.streams_not_list": "streams 参数必须是列表",
        "state.normal": "正常",
        "state.black_screen": "黑屏",
        "state.occluded": "遮挡",
        "state.tampered": "篡改（剧烈运动）",
        "state.error": "错误",
        "error.decoding_failed": "解码失败",
    },
}
