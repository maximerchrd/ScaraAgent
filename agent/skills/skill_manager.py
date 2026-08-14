# agent/skills/skill_manager.py
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

class SkillManager:
    def __init__(self, library_path: str = "agent/skills/library.json"):
        self.skills: List[Dict[str, Any]] = []
        self._load(library_path)

    def _load(self, path: str):
        file = Path(path)
        if not file.exists():
            logging.warning(f"Skill library not found: {path}")
            return
        with open(file, 'r') as f:
            self.skills = json.load(f)
        logging.info(f"Loaded {len(self.skills)} skills.")

    def get_skill_names(self) -> str:
        """Return a compact list for the system prompt."""
        names = [s["name"] for s in self.skills]
        return "Available skills: " + ", ".join(names)

    def get_skill_text(self, skill_name: str) -> Optional[str]:
        """Return the full instructions for a skill."""
        for s in self.skills:
            if s["name"] == skill_name:
                parts = [f"SKILL: {s['name']}"]
                parts.append("")
                for step in s.get("instructions", []):
                    parts.append(f"- {step}")
                return "\n".join(parts)
        return None