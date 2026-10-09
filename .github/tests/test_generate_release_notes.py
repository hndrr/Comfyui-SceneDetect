import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("generate_release_notes", SCRIPTS / "generate_release_notes.py")
summary = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def generate(self, output):
        with patch.object(summary.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output)) as run:
            body = summary.generate_summary("Full release diff")
        return body, run

    def test_single_tool_free_request_uses_auto_model_and_validates_english_bullets(self):
        expected = "- Fix scene detection for trimmed videos.\n- Improve representative frames."
        body, run = self.generate(expected)
        self.assertEqual(body, expected)
        run.assert_called_once()
        command = run.call_args.args[0]
        for flag in ("--model=auto", "--available-tools", "--deny-tool=shell", "--deny-tool=write",
                     "--disable-builtin-mcps", "--no-custom-instructions", "--no-remote-export"):
            self.assertIn(flag, command)
        self.assertNotIn("--allow-all", command)
        self.assertNotIn("--allow-all-tools", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 120)

    def test_invalid_or_attributed_notes_are_rejected(self):
        outputs = ("", "## Changes\n- Fix scene detection.", "- A\n- B\n- C\n- D",
                   "- Fix scenes by @hndrr.", "- Fix scenes in #16.", "- Fix [scenes](https://example.com).",
                   "- シーン検出を修正。", "- " + "x" * 601)
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.generate(output)

    def test_quota_failure_or_timeout_stops_without_retries_or_paid_fallback(self):
        for error in (subprocess.CalledProcessError(1, ["copilot"], stderr="Free allowance exhausted"),
                      subprocess.TimeoutExpired(["copilot"], 120)):
            with self.subTest(error=error), patch.object(summary.subprocess, "run", side_effect=error) as run:
                with self.assertRaisesRegex(RuntimeError, "No release was prepared"):
                    summary.generate_summary("Full release diff")
                run.assert_called_once()

    def test_manual_override_needs_neither_copilot_nor_a_git_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict(summary.os.environ, {"GITHUB_OUTPUT": str(output),
                                               "RELEASE_NOTES_INPUT": "  - Fix trimmed videos.  "}), \
                 patch.object(summary, "generate_summary") as generate, \
                 patch.object(summary, "release_prompt") as prompt, contextlib.redirect_stdout(io.StringIO()):
                summary.main()
            generate.assert_not_called()
            prompt.assert_not_called()
            self.assertIn("- Fix trimmed videos.", output.read_text())


class ReleaseDiffTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.git("init", "-q", "--initial-branch=master")
        self.change("pyproject.toml", '[project]\nversion = "1.2.3"\n')
        self.commit("Previous release")
        self.git("tag", "v1.2.3")

    def git(self, *arguments):
        return subprocess.check_output([
            "git", "-c", "user.name=Release test", "-c", "user.email=release@example.invalid",
            "-c", "commit.gpgsign=false", *arguments,
        ], cwd=self.root, text=True, stderr=subprocess.DEVNULL).strip()

    def change(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self, subject):
        self.git("add", ".")
        self.git("commit", "-qm", subject)

    def prompt(self):
        with contextlib.chdir(self.root):
            return summary.release_prompt()

    def test_summary_reads_entire_release_range_instead_of_pr_titles(self):
        self.change("nodes.py", "trimmed_video_fix = True\n")
        self.commit("First PR title by @author in #16")
        self.change("nodes.py", "trimmed_video_fix = True\nrepresentative_frame_fix = True\n")
        self.change("tests/test_nodes.py", "ignored_test_marker = True\n")
        self.change(".github/release-history.json", '{"old_notes": "ignored_history_marker"}\n')
        self.commit("Second PR title by @author in #20")
        prompt = self.prompt()
        diff = json.loads(prompt.split("\n\n", 1)[1])["git_diff"]
        self.assertIn("trimmed_video_fix", diff)
        self.assertIn("representative_frame_fix", diff)
        for ignored in ("First PR title", "Second PR title", "@author", "ignored_test_marker", "ignored_history_marker"):
            self.assertNotIn(ignored, diff)
        self.assertIn("untrusted source data", prompt)
        self.assertIn("no changes to scene detection behavior", prompt)

    def test_missing_current_release_tag_stops_before_generating_notes(self):
        self.git("tag", "-d", "v1.2.3")
        with self.assertRaisesRegex(ValueError, "Complete release v1.2.3"):
            self.prompt()

    def test_no_release_changes_stop_before_generating_notes(self):
        with self.assertRaisesRegex(ValueError, "no changes"):
            self.prompt()

    def test_large_json_encoded_diff_is_rejected_instead_of_truncated(self):
        self.change("nodes.py", '"' * 50000 + "\n")
        self.commit("Large release diff")
        with self.assertRaisesRegex(ValueError, "automatic summary limit"):
            self.prompt()


if __name__ == "__main__":
    unittest.main()
