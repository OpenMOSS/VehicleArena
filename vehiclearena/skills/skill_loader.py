"""Skill loader for VehicleArena — discovers and loads .md skill files.

Skills are behavior guides that LLM agents load on-demand via the
load_skill() tool. Each .md file in skills/ is a self-contained skill
with YAML frontmatter for metadata.

Usage:
    from skills.skill_loader import SkillLoader
    loader = SkillLoader()
    print(loader.generate_catalog())
    content = loader.load_skill("weather_transition")
"""

import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class SkillMeta:
    """Metadata extracted from a skill file's YAML frontmatter."""
    name: str
    description: str
    trigger: str
    modules: List[str] = field(default_factory=list)
    file_path: str = ""


class SkillLoader:
    """Discovers and loads .md skill files from the skills directory."""

    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir is None:
            skills_dir = os.path.dirname(__file__)
        self.skills_dir = skills_dir
        self._skills: Dict[str, SkillMeta] = {}
        self._scan()

    def _scan(self) -> None:
        """Glob for *.md files in skills_dir and parse their YAML frontmatter."""
        pattern = os.path.join(self.skills_dir, "*.md")
        for path in sorted(glob.glob(pattern)):
            meta = self._parse_frontmatter(path)
            if meta:
                self._skills[meta.name] = meta

    @staticmethod
    def _parse_frontmatter(path: str) -> Optional[SkillMeta]:
        """Parse ``---`` delimited YAML header from a .md file, return SkillMeta or None."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            return None

        # Frontmatter must start with '---'
        if not content.startswith("---"):
            return None

        end_idx = content.find("---", 3)
        if end_idx == -1:
            return None

        raw_fm = content[3:end_idx].strip()

        if HAS_YAML:
            try:
                data = yaml.safe_load(raw_fm)
            except yaml.YAMLError:
                return None
        else:
            # Simple fallback parser for key: value lines
            data = {}
            for line in raw_fm.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Handle list notation [a, b, c]
                    if value.startswith("[") and value.endswith("]"):
                        value = [
                            item.strip().strip('"').strip("'")
                            for item in value[1:-1].split(",")
                            if item.strip()
                        ]
                    data[key] = value

        if not isinstance(data, dict):
            return None

        name = data.get("name")
        if not name:
            return None

        modules = data.get("modules", [])
        if isinstance(modules, str):
            modules = [m.strip() for m in modules.split(",") if m.strip()]

        return SkillMeta(
            name=str(name),
            description=str(data.get("description", "")),
            trigger=str(data.get("trigger", "")),
            modules=list(modules),
            file_path=path,
        )

    def list_skills(self) -> List[SkillMeta]:
        """Return a list of all discovered SkillMeta objects."""
        return list(self._skills.values())

    def load_skill(
        self, skill_name: str, available_modules=None,
    ) -> str:
        """Return the body (after frontmatter) of the named skill, or an error message."""
        meta = self._skills.get(skill_name)
        if meta is None:
            available = ", ".join(sorted(self._skills.keys())) or "(none)"
            return f"[ERROR] Unknown skill '{skill_name}'. Available skills: {available}"

        try:
            with open(meta.file_path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except OSError as exc:
            return f"[ERROR] Could not read skill file: {exc}"

        # Strip frontmatter — find closing '---' after the opening one
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx != -1:
                body = content[end_idx + 3:]
                return self._filter_body(
                    self._expand_body(body.lstrip("\n")), available_modules)

        return self._filter_body(self._expand_body(content), available_modules)

    @staticmethod
    def _expand_body(body: str) -> str:
        """Keep operational weather instructions tied to the shared policy."""
        marker = "{{WEATHER_SAFETY_RULES}}"
        if marker in body:
            from weather_safety import render_weather_transition_rules
            body = body.replace(marker, render_weather_transition_rules())
        return body

    @staticmethod
    def _filter_body(body: str, available_modules=None) -> str:
        """Remove instructions that target equipment absent from this car."""
        if available_modules is None:
            return body
        available = set(available_modules)
        try:
            from registry import ModuleRegistry
            known = set(ModuleRegistry.instance().all_modules())
        except Exception:
            known = set()
        unavailable = known - available
        kept = []
        for line in body.splitlines():
            lowered = line.lower()
            if any(module.lower() in lowered for module in unavailable):
                continue
            if "all " in lowered and " actions required" in lowered:
                kept.append(
                    "**Execute only the applicable installed-equipment "
                    "actions below:**")
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def generate_catalog(self, available_modules=None) -> str:
        """Return a markdown table summarising all available skills."""
        available = (
            set(available_modules)
            if available_modules is not None else None)
        lines = [
            "| Skill | Trigger | Description |",
            "|-------|---------|-------------|",
        ]
        for meta in self.list_skills():
            description = meta.description
            if (
                available is not None
                and set(meta.modules) - available
            ):
                description = (
                    f"{meta.trigger} handling using installed equipment")
            lines.append(
                f"| {meta.name} | {meta.trigger} | {description} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_loader_instance: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Return (and lazily create) a module-level SkillLoader singleton."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = SkillLoader()
    return _loader_instance
