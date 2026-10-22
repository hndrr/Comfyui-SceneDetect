from __future__ import annotations
from typing import Any, Dict, List, Tuple

DEFAULT_SCENE_PROMPT = (
    "Scene {index}/{scene_count}: {start_time}–{end_time} ({duration_sec}s). Describe this shot."
)


class _TemplateMap(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_scenes_for_llm(
    rows: List[Dict[str, Any]],
    template: str = "",
) -> Tuple[str, List[str]]:
    prompt_template = (template or "").strip() or DEFAULT_SCENE_PROMPT
    scene_count = len(rows)
    prompts: List[str] = []
    lines = [f"# Scenes ({scene_count})"]
    for row in rows:
        values = {
            "index": row["index"],
            "scene_count": scene_count,
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration_sec": float(row["duration_sec"]),
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "duration_frames": row["duration_frames"],
            "clip_path": row.get("clip_path") or "",
        }
        try:
            prompts.append(prompt_template.format_map(_TemplateMap(values)))
        except (ValueError, IndexError):
            prompts.append(prompt_template)
        lines.append(
            f"{row['index']}. {row['start_time']} – {row['end_time']} | "
            f"{float(row['duration_sec']):.3f}s | frames {row['start_frame']}-{row['end_frame']}"
        )
    return "\n".join(lines), prompts
