import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


SPEC = importlib.util.spec_from_file_location("release_notes", Path(__file__).parents[1] / "scripts/release_notes.py")
notes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notes)
SHA = "a" * 40
REPO = "hndrr/Comfyui-SceneDetect"


class GitHub:
    def __init__(self):
        self.refs = {}
        self.releases = {}
        self.tags = {}
        self.calls = []

    def __call__(self, method, path, data=None, missing_ok=False):
        self.calls.append((method, path, data))
        route = path.removeprefix(f"/repos/{REPO}/")
        if method == "GET" and route.startswith("git/ref/tags/"):
            return self.refs.get(route.removeprefix("git/ref/tags/"))
        if method == "GET" and route.startswith("git/tags/"):
            return self.tags[route.removeprefix("git/tags/")]
        if method == "GET" and route.startswith("releases/tags/"):
            return self.releases.get(route.removeprefix("releases/tags/"))
        if method == "POST" and route == "git/refs":
            tag = data["ref"].removeprefix("refs/tags/")
            self.refs[tag] = {"object": {"type": "commit", "sha": data["sha"]}}
            return self.refs[tag]
        if method == "POST" and route == "releases":
            self.releases[data["tag_name"]] = data.copy()
            return data.copy()
        if method == "POST" and route == "releases/generate-notes":
            return {"body": f"## Changes\n\nGenerated for {data['tag_name']} at {data['target_commitish']}"}
        raise AssertionError((method, path, data))


class Registry:
    def __init__(self, versions):
        self.versions = versions
        self.calls = []

    def __call__(self, method, path, data=None, missing_ok=False):
        self.calls.append((method, path, data))
        if method == "GET" and path == "/nodes/scenedetect/versions":
            return self.versions
        if method == "PUT":
            for version in self.versions:
                if path == f"/publishers/hndr/nodes/scenedetect/versions/{version['id']}":
                    version.update(data)
                    return version.copy()
        raise AssertionError((method, path, data))


def node(version="1.2.2", deprecated=False):
    return {"version": version, "id": f"uuid-{version}", "changelog": "", "deprecated": deprecated,
            "status": "NodeVersionStatusActive"}


class ReleaseNotesTests(unittest.TestCase):
    def setUp(self):
        self.github = GitHub()
        self.item = node(deprecated=True)
        self.registry = Registry([self.item])

    def sync(self, body="## Changes\n\n- シーン検出を修正。"):
        with contextlib.redirect_stdout(io.StringIO()):
            notes.sync_release(self.github, self.registry, REPO, "hndr", "scenedetect", self.item, SHA, body)

    def test_new_release_has_same_body_and_published_commit(self):
        self.sync()
        release = self.github.releases["v1.2.2"]
        self.assertEqual(release["body"], self.item["changelog"])
        self.assertEqual(self.github.refs["v1.2.2"]["object"]["sha"], SHA)
        self.assertEqual(release["target_commitish"], SHA)
        self.assertTrue(self.item["deprecated"])
        self.assertEqual([call[0] for call in self.registry.calls], ["PUT"])

    def test_rerun_does_not_create_or_update_twice(self):
        self.sync()
        github_writes = len([call for call in self.github.calls if call[0] == "POST"])
        registry_writes = len(self.registry.calls)
        self.sync()
        self.assertEqual(len([call for call in self.github.calls if call[0] == "POST"]), github_writes)
        self.assertEqual(len(self.registry.calls), registry_writes)

    def test_edited_github_notes_are_copied_to_registry(self):
        self.sync()
        self.github.releases["v1.2.2"]["body"] = "## Fixes\n\n- Edited release comment."
        self.sync("old generated notes")
        self.assertEqual(self.item["changelog"], self.github.releases["v1.2.2"]["body"])
        self.assertEqual(len([call for call in self.github.calls if call[0] == "POST"]), 2)

    def test_wrong_tag_stops_before_any_write(self):
        self.github.refs["v1.2.2"] = {"object": {"type": "commit", "sha": "b" * 40}}
        with self.assertRaisesRegex(ValueError, "does not point"):
            self.sync()
        self.assertEqual(self.registry.calls, [])
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))

    def test_annotated_tag_is_resolved_to_commit(self):
        self.github.refs["v1.2.2"] = {"object": {"type": "tag", "sha": "b" * 40}}
        self.github.tags["b" * 40] = {"object": {"type": "commit", "sha": SHA}}
        self.sync()
        self.assertIn("v1.2.2", self.github.releases)
        self.assertFalse(any(call[1].endswith("git/refs") and call[0] == "POST" for call in self.github.calls))

    def test_draft_is_not_overwritten_or_published(self):
        self.github.refs["v1.2.2"] = {"object": {"type": "commit", "sha": SHA}}
        self.github.releases["v1.2.2"] = {"draft": True, "body": "Unfinished notes"}
        with self.assertRaisesRegex(ValueError, "published tag"):
            self.sync()
        self.assertEqual(self.registry.calls, [])

    def test_prepare_uses_existing_notes_without_regenerating(self):
        self.sync()
        body = notes.release_body(self.github, REPO, "1.2.2", SHA)
        self.assertEqual(body, self.item["changelog"])
        self.assertFalse(any(call[1].endswith("generate-notes") for call in self.github.calls))

    def test_backfill_consolidates_deleted_version_and_preserves_old_versions(self):
        history = json.loads((Path(__file__).parents[1] / "release-history.json").read_text())
        versions = [node(version, deprecated=version != "1.2.2") for version in history["releases"]]
        versions += [node("1.2.1"), dict(node("8.0.0"), status="NodeVersionStatusDeleted")]
        registry = Registry(versions)
        version_by_sha = {item["sha"]: version for version, item in history["releases"].items()}
        def fake_git(*args):
            if args[0] == "show":
                return f'[project]\nversion = "{version_by_sha[args[1].split(":")[0]]}"\n'
            return ""
        with patch.object(notes, "git", side_effect=fake_git), contextlib.redirect_stdout(io.StringIO()):
            notes.backfill(self.github, registry, REPO, "hndr", "scenedetect", history["releases"], history["superseded"])
        self.assertEqual(set(self.github.releases), {f"v{v}" for v in history["releases"]})
        self.assertNotIn("v1.2.1", self.github.releases)
        self.assertIn("1.2.1", self.github.releases["v1.2.2"]["body"])
        self.assertEqual(self.github.releases["v1.2.2"]["make_latest"], "true")
        for item in registry.versions:
            if item["version"] in history["releases"]:
                self.assertEqual(item["changelog"], self.github.releases[f"v{item['version']}"]["body"])
                self.assertEqual(item["deprecated"], item["version"] != "1.2.2")
        self.assertTrue(all(call[0] in ("GET", "PUT") for call in registry.calls))

    def test_api_only_treats_404_as_missing_and_hides_error_body(self):
        api = notes.API("https://api.example.invalid", "secret-token")
        for status in (401, 403, 500):
            error = HTTPError("https://api.example.invalid/test", status, "failure", {}, io.BytesIO(b"secret-token"))
            with patch.object(notes, "urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as caught:
                    api("GET", "/test", missing_ok=True)
                self.assertNotIn("secret-token", str(caught.exception))
        with patch.object(notes, "urlopen", side_effect=HTTPError("url", 404, "missing", {}, None)):
            self.assertIsNone(api("GET", "/test", missing_ok=True))

    def test_multiline_output_preserves_note_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output"
            body = "## Changes\n\n- 日本語の更新内容。\n- Another change."
            notes.write_outputs(path, {"body": body})
            lines = path.read_text().splitlines()
            delimiter = lines[0].split("<<", 1)[1]
            self.assertEqual(lines[-1], delimiter)
            self.assertEqual("\n".join(lines[1:-1]), body)


if __name__ == "__main__":
    unittest.main()
