import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs


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
            release = self.releases.get(route.removeprefix("releases/tags/"))
            return None if release is None or release["draft"] else release
        if method == "GET" and route.startswith("releases?"):
            query = parse_qs(route.split("?", 1)[1])
            page, size = int(query["page"][0]), int(query["per_page"][0])
            releases = [dict(release, tag_name=tag) for tag, release in self.releases.items()]
            return releases[(page - 1) * size:page * size]
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
        with self.assertRaisesRegex(ValueError, "is a draft"):
            self.sync()
        self.assertEqual(self.registry.calls, [])

    def test_draft_on_second_page_stops_before_any_write(self):
        self.github.releases = {f"v0.0.{index}": {"draft": False, "body": "Older notes"}
                                for index in range(100)}
        self.github.releases["v1.2.2"] = {"draft": True, "body": "Unfinished notes"}
        self.assertIsNone(self.github("GET", f"/repos/{REPO}/releases/tags/v1.2.2"))
        with self.assertRaisesRegex(ValueError, "is a draft"):
            self.sync()
        self.assertEqual(self.registry.calls, [])
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))
        self.assertTrue(any("page=2" in call[1] for call in self.github.calls))

    def test_prepare_rejects_draft_without_creating_notes(self):
        self.github.releases["v1.2.2"] = {"draft": True, "body": "Unfinished notes"}
        with self.assertRaisesRegex(ValueError, "is a draft"):
            notes.release_body(self.github, REPO, "1.2.2", SHA, "Replacement notes")
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))

    def test_backfill_checks_all_drafts_before_any_write(self):
        history = {version: {"sha": SHA, "body": "Historical notes"} for version in ("1.0.0", "1.2.2")}
        registry = Registry([node(version) for version in history])
        self.github.releases["v1.2.2"] = {"draft": True, "body": "Unfinished notes"}
        configs = iter(['[project]\nversion = "1.0.0"\n', '[project]\nversion = "1.2.2"\n'])
        def fake_git(*args):
            return next(configs) if args[0] == "show" else ""
        with patch.object(notes, "git", side_effect=fake_git):
            with self.assertRaisesRegex(ValueError, "is a draft"):
                notes.backfill(self.github, registry, REPO, "hndr", "scenedetect", history, {})
        self.assertFalse(any(call[0] == "PUT" for call in registry.calls))
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))

    def test_prepare_uses_existing_notes_without_regenerating(self):
        self.sync()
        body = notes.release_body(self.github, REPO, "1.2.2", SHA)
        self.assertEqual(body, self.item["changelog"])
        self.assertFalse(any(call[1].endswith("generate-notes") for call in self.github.calls))

    def test_new_release_uses_the_summary_for_github_and_registry(self):
        summary = "- Fix Registry distribution of the updated scene detection nodes."
        body = notes.release_body(self.github, REPO, "1.2.2", SHA, summary)
        self.sync(body)
        self.assertEqual(self.github.releases["v1.2.2"]["body"], summary)
        self.assertEqual(self.item["changelog"], summary)
        self.assertFalse(any(call[1].endswith("generate-notes") for call in self.github.calls))

    def test_missing_or_blank_summary_stops_before_writes(self):
        for summary in (None, "", " \n "):
            with self.subTest(summary=summary), self.assertRaisesRegex(ValueError, "Release notes"):
                notes.release_body(self.github, REPO, "1.2.2", SHA, summary)
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))
        self.assertEqual(self.registry.calls, [])

    def test_release_commit_resolves_summary_without_recorded_sha(self):
        with patch.object(notes, "git", return_value=SHA):
            self.assertEqual(notes.release_commit("1.2.4", {"1.2.4": {"body": "- Fix scene detection."}}), SHA)

    def test_backfill_consolidates_deleted_version_and_preserves_old_versions(self):
        history = json.loads((Path(__file__).parents[1] / "release-history.json").read_text())
        history["releases"] = {version: item for version, item in history["releases"].items() if item.get("sha")}
        latest = max(history["releases"], key=lambda version: tuple(map(int, version.split("."))))
        versions = [node(version, deprecated=version != latest) for version in history["releases"]]
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
        self.assertEqual(history["superseded"]["1.2.1"], "1.2.2")
        self.assertEqual(self.github.releases[f"v{latest}"]["make_latest"], "true")
        for item in registry.versions:
            if item["version"] in history["releases"]:
                self.assertEqual(item["changelog"], self.github.releases[f"v{item['version']}"]["body"])
                self.assertEqual(item["deprecated"], item["version"] != latest)
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


class PreparedReleaseTests(unittest.TestCase):
    OBJECT_FORMAT = "sha1"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.github = GitHub()
        self.registry = Registry([])
        self.sequence = 0
        self.git("init", "-q", "--initial-branch=master", f"--object-format={self.OBJECT_FORMAT}")
        self.before = self.commit_version("1.2.3")

    def git(self, *arguments):
        return subprocess.check_output([
            "git", "-c", "user.name=Release test", "-c", "user.email=release@example.invalid",
            "-c", "commit.gpgsign=false", *arguments,
        ], cwd=self.root, text=True).strip()

    def commit_version(self, version, summary=None):
        self.sequence += 1
        (self.root / "pyproject.toml").write_text(
            f'[project]\nname = "scenedetect"\nversion = "{version}"\n'
            f'description = "Metadata update {self.sequence}"\n', encoding="utf-8",
        )
        (self.root / ".github").mkdir(exist_ok=True)
        releases = {} if summary is None else {version: {"body": summary}}
        (self.root / ".github/release-history.json").write_text(
            json.dumps({"releases": releases}), encoding="utf-8",
        )
        self.git("add", ".")
        self.git("commit", "-qm", f"Update release metadata to {version}")
        return self.git("rev-parse", "HEAD")

    def prepare(self, event="push", supplied=""):
        def api(base_url, token):
            return self.registry if base_url == "https://api.comfy.org" else self.github
        with contextlib.chdir(self.root), patch.object(notes, "API", side_effect=api), \
             patch.dict(notes.os.environ, {
                 "GITHUB_REPOSITORY": REPO, "GH_TOKEN": "test-token", "GITHUB_EVENT_NAME": event,
                 "REGISTRY_ACCESS_TOKEN": "test-registry-token",
                 "BEFORE_SHA": self.before, "GITHUB_OUTPUT": str(self.root / "output"),
                 "RELEASE_NOTES_INPUT": supplied,
             }), patch("sys.argv", ["release_notes.py", "prepare"]):
            notes.main()
        lines = (self.root / "output").read_text(encoding="utf-8").splitlines()
        result = {}
        while lines:
            name, delimiter = lines.pop(0).split("<<", 1)
            end = lines.index(delimiter)
            result[name] = "\n".join(lines[:end])
            lines = lines[end + 1:]
        return result

    def test_same_version_metadata_push_does_not_publish_or_request_notes(self):
        self.commit_version("1.2.3")
        self.assertEqual(self.prepare(), {"publish": "false"})
        self.assertEqual(self.github.calls, [])
        self.assertEqual(self.registry.calls, [])

    def test_manual_metadata_only_revision_does_not_publish(self):
        self.commit_version("1.2.3")
        self.assertEqual(self.prepare("workflow_dispatch"), {"publish": "false"})
        self.assertEqual(self.github.calls, [])
        self.assertEqual(self.registry.calls, [])

    def test_manual_version_decrease_stops_before_any_api_call(self):
        self.commit_version("1.2.2", "- Fix scene detection.")
        with self.assertRaisesRegex(ValueError, "must increase"):
            self.prepare("workflow_dispatch")
        self.assertEqual(self.github.calls, [])
        self.assertEqual(self.registry.calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_manual_initial_revision_requires_a_previous_revision(self):
        with self.assertRaisesRegex(ValueError, "previous.*revision"):
            self.prepare("workflow_dispatch")
        self.assertEqual(self.registry.calls, [])

    def test_prepare_uses_recorded_summary_and_manual_input_overrides_it(self):
        recorded = "- Fix scene detection for trimmed inputs."
        manual = "- Fix scene detection and representative image output."
        self.commit_version("1.2.4", recorded)
        self.registry = Registry([node("1.2.3")])
        for supplied, expected in (("", recorded), (manual, manual)):
            with self.subTest(supplied=supplied):
                (self.root / "output").unlink(missing_ok=True)
                self.assertEqual(self.prepare("workflow_dispatch", supplied), {
                    "publish": "true", "version": "1.2.4", "body": expected,
                })
        self.assertTrue(all(call[0] == "GET" for call in self.registry.calls))

    def test_existing_registry_version_is_not_uploaded_again_in_any_status(self):
        self.commit_version("1.2.4", "- Fix scene detection.")
        for event in ("push", "workflow_dispatch"):
            for status in ("Active", "Pending", "Flagged", "Deleted"):
                with self.subTest(event=event, status=status):
                    (self.root / "output").unlink(missing_ok=True)
                    self.registry = Registry([dict(node("1.2.4"), status=f"NodeVersionStatus{status}")])
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(self.prepare(event), {"publish": "false"})
                    self.assertEqual(self.github.calls, [])
                    self.assertTrue(all(call[0] == "GET" for call in self.registry.calls))

    def test_release_older_than_registry_stops_before_publication(self):
        self.commit_version("1.2.4", "- Fix scene detection.")
        self.registry = Registry([node("1.2.10")])
        for event in ("push", "workflow_dispatch"):
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, "older than Registry"):
                self.prepare(event)
        self.assertEqual(self.github.calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_registry_failure_stops_before_publication(self):
        self.commit_version("1.2.4", "- Fix scene detection.")
        def unavailable(*arguments):
            raise RuntimeError("Registry unavailable")
        self.registry = unavailable
        with self.assertRaisesRegex(RuntimeError, "Registry unavailable"):
            self.prepare("workflow_dispatch")
        self.assertEqual(self.github.calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_version_increase_publishes_the_prepared_summary(self):
        summary = "- Fix scene detection for trimmed inputs.\n- Improve representative image output."
        for version in ("1.2.4", "1.2.10", "1.3.0", "2.0.0"):
            with self.subTest(version=version):
                (self.root / "output").unlink(missing_ok=True)
                self.commit_version(version, summary)
                self.assertEqual(self.prepare(), {"publish": "true", "version": version, "body": summary})
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))

    def test_version_decrease_stops_before_any_api_call(self):
        self.commit_version("1.2.2", "- Fix scene detection.")
        with self.assertRaisesRegex(ValueError, "must increase"):
            self.prepare()
        self.assertEqual(self.github.calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_missing_or_unreadable_previous_revision_stops_with_a_clear_error(self):
        self.commit_version("1.2.4", "- Fix scene detection.")
        for before in ("0" * 40, "0" * 64, "f" * 40, "f" * 64, "", "invalid"):
            with self.subTest(before=before):
                self.before = before
                with self.assertRaisesRegex(ValueError, "previous.*revision") as caught:
                    self.prepare()
                if before.startswith("f"):
                    self.assertTrue(caught.exception.__suppress_context__)
                self.assertEqual(self.github.calls, [])
                self.assertFalse((self.root / "output").exists())

    def test_version_increase_requires_a_nonempty_summary(self):
        for summary in (None, "", " \n "):
            with self.subTest(summary=summary):
                self.commit_version("1.2.4", summary)
                with self.assertRaisesRegex(ValueError, "Release notes"):
                    self.prepare()
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))
        self.assertFalse((self.root / "output").exists())

    def test_backfill_resolves_a_prepared_release_from_its_tag(self):
        summary = "- Fix scene detection for trimmed inputs."
        sha = self.commit_version("1.2.4", summary)
        self.git("tag", "-a", "v1.2.4", "-m", "Release 1.2.4")
        self.commit_version("1.2.4", "- Unreleased changes after publication.")
        self.github.refs["v1.2.4"] = {"object": {"type": "commit", "sha": sha}}
        self.github.releases["v1.2.4"] = {"draft": False, "body": summary}
        item = node("1.2.4", deprecated=True)
        registry = Registry([item])
        with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
            notes.backfill(self.github, registry, REPO, "hndr", "scenedetect", {}, {})
        self.assertEqual(item["changelog"], summary)
        self.assertTrue(item["deprecated"])
        self.assertFalse(any(call[0] == "POST" for call in self.github.calls))


class Sha256PreparedReleaseTests(PreparedReleaseTests):
    OBJECT_FORMAT = "sha256"


class CommitIdTests(unittest.TestCase):
    def test_supported_object_formats_require_full_nonzero_lowercase_hex_ids(self):
        for size in (40, 64):
            notes.validate_commit_sha("a" * size)
        for value in ("", "a" * 7, "a" * 41, "a" * 63, "a" * 65, "0" * 40,
                      "0" * 64, "g" * 40, "A" * 64, "HEAD", "refs/heads/master"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                notes.validate_commit_sha(value)


if __name__ == "__main__":
    unittest.main()
