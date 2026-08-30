"""Lock sample workflows to the current DynamicCombo widget order.

ComfyUI stores widget values by position. If schema_v3.py changes field
order or defaults, these graphs must be rewritten the same way.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO / "workflow"

# Flattened widget order for method=content, matching schema_v3.py.
# Nested DynamicCombo children use parent.child names in the live UI;
# the workflow JSON still serializes them as a positional array.
CONTENT_METHOD_WIDGETS = [
    ("method", "content"),
    ("method.threshold", 27.0),
    ("method.luma_only", True),
    ("method.delta_hue", 1.0),
    ("method.delta_sat", 1.0),
    ("method.delta_lum", 1.0),
    ("method.delta_edges", 0.0),
    ("method.kernel_size", 0),
]

COMMON_WIDGETS = [
    ("min_scene_len_sec", 0.0),
    ("min_scene_len_frames", 15),
    ("representative", "start"),
]

SPLIT_CLIPS_ON = [
    ("split_clips", "true"),
    ("split_clips.split_reencode", False),
]

SHOW_ALL_OFF = [
    ("show_all_settings", "false"),
]

VIDEO_WIDGETS = CONTENT_METHOD_WIDGETS + COMMON_WIDGETS + SPLIT_CLIPS_ON + SHOW_ALL_OFF
LEGACY_WIDGETS = CONTENT_METHOD_WIDGETS + COMMON_WIDGETS + SHOW_ALL_OFF

VIDEO_OUTPUTS = [
    "images",
    "scenes_json",
    "scene_count",
    "all_scenes_text",
    "per_scene_prompt_list",
    "videos",
]
LEGACY_OUTPUTS = [
    "images",
    "scenes_json",
    "scene_count",
    "all_scenes_text",
    "per_scene_prompt_list",
]


def _load_workflow(name: str) -> dict:
    path = WORKFLOW_DIR / name
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _node_by_type(graph: dict, node_type: str) -> dict:
    matches = [node for node in graph["nodes"] if node["type"] == node_type]
    if len(matches) != 1:
        raise AssertionError(f"expected one {node_type} node, found {len(matches)}")
    return matches[0]


def _output_names(node: dict) -> list[str]:
    return [slot["name"] for slot in node.get("outputs", [])]


def _wired_output_names(node: dict) -> list[str]:
    names = []
    for slot in node.get("outputs", []):
        links = slot.get("links") or []
        if links:
            names.append(slot["name"])
    return names


def _link_map(graph: dict) -> dict[int, list]:
    return {row[0]: row for row in graph["links"]}


class SampleWorkflowTests(unittest.TestCase):
    def test_recommended_graph_matches_current_video_node(self) -> None:
        graph = _load_workflow("pyscene_workflow.json")
        detect = _node_by_type(graph, "PySceneDetectVideo")
        preview = _node_by_type(graph, "PySceneDetectPreviewVideos")
        save = _node_by_type(graph, "SaveVideo")
        load = _node_by_type(graph, "LoadVideo")

        self.assertEqual(_output_names(detect), VIDEO_OUTPUTS)
        self.assertEqual(_wired_output_names(detect), VIDEO_OUTPUTS)
        self.assertEqual(
            [slot["name"] for slot in detect["inputs"]],
            ["video"] + [name for name, _value in VIDEO_WIDGETS],
        )
        for slot in detect["inputs"][1:]:
            self.assertEqual(slot["widget"]["name"], slot["name"])
            self.assertIsNone(slot["link"])
        self.assertEqual(
            detect["widgets_values"],
            [value for _name, value in VIDEO_WIDGETS],
        )
        self.assertNotIn("PySceneDetectToImages", {node["type"] for node in graph["nodes"]})
        self.assertNotIn("VHS_LoadVideo", {node["type"] for node in graph["nodes"]})

        links = _link_map(graph)
        video_in = next(slot for slot in detect["inputs"] if slot["name"] == "video")
        self.assertEqual(links[video_in["link"]][1], load["id"])
        self.assertEqual(links[preview["inputs"][0]["link"]][1], detect["id"])
        self.assertEqual(links[preview["inputs"][0]["link"]][2], VIDEO_OUTPUTS.index("videos"))
        self.assertEqual(links[save["inputs"][0]["link"]][1], detect["id"])
        self.assertEqual(links[save["inputs"][0]["link"]][2], VIDEO_OUTPUTS.index("videos"))

        preview_any_targets = {
            slot["name"]: slot["links"][0]
            for slot in detect["outputs"]
            if slot["name"]
            in {"scenes_json", "scene_count", "all_scenes_text", "per_scene_prompt_list"}
        }
        for output_name, link_id in preview_any_targets.items():
            target_id = links[link_id][3]
            target = next(node for node in graph["nodes"] if node["id"] == target_id)
            self.assertEqual(target["type"], "PreviewAny", output_name)
            self.assertIn(output_name, target.get("title", ""), output_name)

        image_link = next(
            slot["links"][0] for slot in detect["outputs"] if slot["name"] == "images"
        )
        image_target = next(
            node for node in graph["nodes"] if node["id"] == links[image_link][3]
        )
        self.assertEqual(image_target["type"], "PreviewImage")

    def test_legacy_graph_matches_current_vhs_node(self) -> None:
        graph = _load_workflow("pyscene_workflow_legacy_vhs.json")
        detect = _node_by_type(graph, "PySceneDetectToImages")
        load = _node_by_type(graph, "VHS_LoadVideo")

        self.assertEqual(_output_names(detect), LEGACY_OUTPUTS)
        self.assertEqual(_wired_output_names(detect), LEGACY_OUTPUTS)
        self.assertEqual(
            [slot["name"] for slot in detect["inputs"]],
            ["image", "video_info"] + [name for name, _value in LEGACY_WIDGETS],
        )
        for slot in detect["inputs"][2:]:
            self.assertEqual(slot["widget"]["name"], slot["name"])
            self.assertIsNone(slot["link"])
        self.assertEqual(
            detect["widgets_values"],
            [value for _name, value in LEGACY_WIDGETS],
        )
        self.assertNotIn("PySceneDetectVideo", {node["type"] for node in graph["nodes"]})
        self.assertNotIn(
            "PySceneDetectPreviewVideos", {node["type"] for node in graph["nodes"]}
        )

        links = _link_map(graph)
        image_in = next(slot for slot in detect["inputs"] if slot["name"] == "image")
        info_in = next(slot for slot in detect["inputs"] if slot["name"] == "video_info")
        self.assertEqual(links[image_in["link"]][1], load["id"])
        self.assertEqual(links[info_in["link"]][1], load["id"])
        self.assertEqual(links[info_in["link"]][2], 3)

        for slot in detect["outputs"]:
            target_id = links[slot["links"][0]][3]
            target = next(node for node in graph["nodes"] if node["id"] == target_id)
            if slot["name"] == "images":
                self.assertEqual(target["type"], "PreviewImage")
            else:
                self.assertEqual(target["type"], "PreviewAny")
                self.assertIn(slot["name"], target.get("title", ""))


if __name__ == "__main__":
    unittest.main()
