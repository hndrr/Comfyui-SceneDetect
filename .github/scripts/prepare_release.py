"""Create a release preparation PR with an automatic version bump and curated notes."""

import argparse
import json
import os
from pathlib import Path
import re
import tomllib
from urllib.parse import quote

from release_notes import API, git, validate_release


def prepare_pull_request(github, repository, sha, source, history, bump, body):
    current = tomllib.loads(source)["project"]["version"]
    validate_release(current, sha)
    body = body.strip()
    if not body:
        raise ValueError("An English release summary is required")
    major, minor, patch = map(int, current.split("."))
    versions = {"patch": f"{major}.{minor}.{patch + 1}", "minor": f"{major}.{minor + 1}.0", "major": f"{major + 1}.0.0"}
    version = versions[bump]
    project = re.search(r"(?ms)^\[project\][^\n]*\n.*?(?=^\[|\Z)", source)
    if project is None:
        raise ValueError("Expected a [project] section")
    updated, count = re.subn(
        rf'''(?m)^([ \t]*version[ \t]*=[ \t]*)(["']){re.escape(current)}\2''',
        lambda match: f"{match[1]}{match[2]}{version}{match[2]}", project[0],
    )
    if count != 1:
        raise ValueError("Expected one project.version assignment")
    source = source[:project.start()] + updated + source[project.end():]
    history = json.loads(history)
    history["releases"][version] = {"body": body}
    history = json.dumps(history, ensure_ascii=False, indent=2) + "\n"
    prefix = f"/repos/{repository}"
    branch = f"codex/release-{version}"
    head = quote(f"{repository.split('/')[0]}:{branch}", safe="")
    existing = github("GET", f"{prefix}/pulls?state=open&base=master&head={head}")
    if existing:
        return existing[0]["html_url"]
    ref = github("GET", f"{prefix}/git/ref/heads/{branch}", missing_ok=True)
    parent = github("GET", f"{prefix}/git/commits/{sha}")
    tree = github("POST", f"{prefix}/git/trees", {
        "base_tree": parent["tree"]["sha"],
        "tree": [
            {"path": "pyproject.toml", "mode": "100644", "type": "blob", "content": source},
            {"path": ".github/release-history.json", "mode": "100644", "type": "blob", "content": history},
        ],
    })
    if ref is not None:
        commit = github("GET", f"{prefix}/git/commits/{ref['object']['sha']}")
        if commit["tree"]["sha"] != tree["sha"] or [parent["sha"] for parent in commit["parents"]] != [sha]:
            raise ValueError(f"Branch {branch} already has different changes; review it before retrying")
    else:
        commit = github("POST", f"{prefix}/git/commits", {
            "message": f"Prepare release v{version}", "tree": tree["sha"], "parents": [sha],
        })
        github("POST", f"{prefix}/git/refs", {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
    pull = github("POST", f"{prefix}/pulls", {
        "title": f"Release v{version}", "head": branch, "base": "master",
        "body": f"{body}\n\nMerging this PR publishes v{version} to Comfy Registry and GitHub Releases.",
    })
    return pull["html_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump", choices=("patch", "minor", "major"), default="patch")
    arguments = parser.parse_args()
    github = API(os.environ.get("GITHUB_API_URL", "https://api.github.com"), os.environ["GH_TOKEN"])
    url = prepare_pull_request(
        github, os.environ["GITHUB_REPOSITORY"], git("rev-parse", "HEAD"),
        Path("pyproject.toml").read_text(encoding="utf-8"),
        Path(".github/release-history.json").read_text(encoding="utf-8"),
        arguments.bump, os.environ["RELEASE_NOTES_INPUT"],
    )
    print(url)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
            output.write(f"Release preparation PR: {url}\n")


if __name__ == "__main__":
    main()
