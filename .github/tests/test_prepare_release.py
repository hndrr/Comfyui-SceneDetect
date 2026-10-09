import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("prepare_release", SCRIPTS / "prepare_release.py")
prepare = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(prepare)
SHA = "a" * 40
REPO = "hndrr/Comfyui-SceneDetect"


class GitHub:
    def __init__(self):
        self.calls = []
        self.pulls = []
        self.refs = {}
        self.commits = {SHA: {"sha": SHA, "tree": {"sha": "base-tree"}}}
        self.fail_pull = False

    def __call__(self, method, path, data=None, missing_ok=False):
        self.calls.append((method, path, data))
        route = path.removeprefix(f"/repos/{REPO}/")
        if method == "GET" and route.startswith("pulls?"):
            return self.pulls
        if method == "GET" and route.startswith("git/ref/heads/"):
            return self.refs.get(route.removeprefix("git/ref/heads/"))
        if method == "GET" and route.startswith("git/commits/"):
            return self.commits[route.removeprefix("git/commits/")]
        if method == "POST" and route == "git/trees":
            return {"sha": "prepared-tree"}
        if method == "POST" and route == "git/commits":
            commit = {"sha": "b" * 40, "tree": {"sha": data["tree"]},
                      "parents": [{"sha": sha} for sha in data["parents"]]}
            self.commits[commit["sha"]] = commit
            return commit
        if method == "POST" and route == "git/refs":
            self.refs[data["ref"].removeprefix("refs/heads/")] = {"object": {"sha": data["sha"]}}
            return {}
        if method == "POST" and route == "pulls":
            if self.fail_pull:
                self.fail_pull = False
                raise RuntimeError("Temporary pull request API failure")
            self.pulls.append({"html_url": f"https://github.com/{REPO}/pull/123", **data})
            return self.pulls[-1]
        raise AssertionError((method, path, data))


class PrepareReleaseTests(unittest.TestCase):
    def setUp(self):
        self.github = GitHub()
        self.source = (
            "[project]\nname = 'scenedetect'\nversion = '1.2.9' # Release version\n"
            "requires-python = '>=3.10'\n\n[tool.comfy]\nversion = 'unrelated'\n"
        )
        self.history = {"superseded": {"1.2.1": "1.2.2"}, "releases": {
            "1.2.8": {"sha": SHA, "body": "- Previous fixes."},
        }}
        self.summary = "- Fix scene detection for trimmed inputs.\n- Improve representative frames."

    def prepare(self, bump="patch", summary=None):
        return prepare.prepare_pull_request(
            self.github, REPO, SHA, self.source, json.dumps(self.history), bump,
            self.summary if summary is None else summary,
        )

    def test_each_bump_creates_one_pr_with_version_and_shared_summary(self):
        for bump, version in (("patch", "1.2.10"), ("minor", "1.3.0"), ("major", "2.0.0")):
            with self.subTest(bump=bump):
                self.github = GitHub()
                url = self.prepare(bump)
                tree = next(data for method, path, data in self.github.calls if path.endswith("git/trees"))
                files = {entry["path"]: entry["content"] for entry in tree["tree"]}
                self.assertEqual(tree["base_tree"], "base-tree")
                self.assertEqual(set(files), {"pyproject.toml", ".github/release-history.json"})
                config = tomllib.loads(files["pyproject.toml"])
                self.assertEqual(config["project"]["version"], version)
                self.assertEqual(config["project"]["requires-python"], ">=3.10")
                self.assertEqual(config["tool"]["comfy"]["version"], "unrelated")
                self.assertIn("# Release version", files["pyproject.toml"])
                history = json.loads(files[".github/release-history.json"])
                self.assertEqual(history["releases"][version], {"body": self.summary})
                self.assertEqual(history["releases"]["1.2.8"], self.history["releases"]["1.2.8"])
                self.assertEqual(history["superseded"], self.history["superseded"])
                pull = self.github.pulls[0]
                self.assertEqual(pull["title"], f"Release v{version}")
                self.assertEqual(pull["head"], f"release/{version}")
                self.assertEqual(pull["base"], "master")
                self.assertIn(self.summary, pull["body"])
                self.assertEqual(url, pull["html_url"])
                commit = self.github.commits["b" * 40]
                self.assertEqual(commit["parents"], [{"sha": SHA}])
                self.assertEqual(set(self.github.refs), {f"release/{version}"})

    def test_rerun_keeps_existing_pr_and_its_reviewed_notes(self):
        url = self.prepare()
        writes = len([call for call in self.github.calls if call[0] == "POST"])
        self.assertEqual(self.prepare(summary="- Different notes on a retry."), url)
        self.assertEqual(len([call for call in self.github.calls if call[0] == "POST"]), writes)
        self.assertIn(self.summary, self.github.pulls[0]["body"])

    def test_partial_failure_recovers_the_branch_without_recreating_its_commit(self):
        self.github.fail_pull = True
        with self.assertRaisesRegex(RuntimeError, "Temporary"):
            self.prepare()
        self.assertEqual(self.prepare(), self.github.pulls[0]["html_url"])
        for endpoint in ("git/commits", "git/refs"):
            writes = [call for call in self.github.calls if call[0] == "POST" and call[1].endswith(endpoint)]
            self.assertEqual(len(writes), 1)

    def test_existing_branch_with_other_changes_is_not_overwritten(self):
        self.github.refs["release/1.2.10"] = {"object": {"sha": "c" * 40}}
        self.github.commits["c" * 40] = {"tree": {"sha": "different-tree"}, "parents": [{"sha": SHA}]}
        with self.assertRaisesRegex(ValueError, "already has different changes"):
            self.prepare()
        self.assertFalse(any(call[0] == "POST" and not call[1].endswith("git/trees") for call in self.github.calls))

    def test_blank_summary_stops_before_creating_any_objects_or_pr(self):
        with self.assertRaisesRegex(ValueError, "summary is required"):
            self.prepare(summary=" \n ")
        self.assertEqual(self.github.calls, [])

    def test_cli_outputs_identify_the_prepared_pr_for_automatic_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            url = f"https://github.com/{REPO}/pull/123"
            def github(method, path, data=None):
                self.assertEqual((method, path), ("GET", f"/repos/{REPO}/pulls/123"))
                return {"number": 123, "head": {"sha": SHA}}
            with patch.object(prepare, "API", return_value=github), \
                 patch.object(prepare, "prepare_pull_request", return_value=url), \
                 patch.object(prepare, "git", return_value=SHA), \
                 patch.dict(prepare.os.environ, {"GITHUB_REPOSITORY": REPO, "GH_TOKEN": "test-token",
                                                "GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": "",
                                                "RELEASE_NOTES_INPUT": self.summary}), \
                 patch("sys.argv", ["prepare_release.py"]), contextlib.redirect_stdout(io.StringIO()):
                prepare.main()
            self.assertIn("123", output.read_text())
            self.assertIn(SHA, output.read_text())


if __name__ == "__main__":
    unittest.main()
