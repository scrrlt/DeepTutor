"""
Prompt Types.
Encapsulates raw YAML data into a robust object.
"""

from typing import Any


class PromptBundle:
    def __init__(self, data: dict[str, Any], source_id: str = "unknown"):
        self._data = data
        self.source_id = source_id  # Useful for debugging (e.g., "chat_agent_en")

    def get(self, key: str, fallback: str = "") -> str:
        """
        Robust lookup supporting dot notation (e.g., 'system.main').
        """
        if "." in key:
            section, subkey = key.split(".", 1)
            # Safe nested lookup
            section_data = self._data.get(section, {})
            if isinstance(section_data, dict):
                return str(section_data.get(subkey, fallback))
            return fallback

        val = self._data.get(key, fallback)
        return str(val) if val is not None else fallback

    def get_template(self, key: str, **kwargs) -> str:
        """
        Get and immediately format a template.
        Raises KeyError if arguments are missing (Fail Fast).
        """
        tmpl = self.get(key)
        if not tmpl:
            return ""
        try:
            return tmpl.format(**kwargs)
        except KeyError as e:
            # Observability: Log exactly which variable was missing in which prompt
            raise ValueError(f"Missing var {e} in prompt '{key}' (source: {self.source_id})") from e
