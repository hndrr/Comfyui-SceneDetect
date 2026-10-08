import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/pyscenedetect.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "");

function loadPreview(extra = {}) {
    const context = vm.createContext({
        app: { registerExtension() {} },
        api: { apiURL: (path) => path },
        URLSearchParams,
        ...extra,
    });
    vm.runInContext(source, context);
    return context;
}

test("portrait and landscape clips retain their aspect ratio", () => {
    const preview = loadPreview();
    for (const aspect of [9 / 16, 16 / 9, 1]) {
        const size = preview.stageSizeForAspect(320, aspect);
        assert.ok(Math.abs(size.width / size.height - aspect) < 0.01);
        assert.ok(size.height <= 560);
    }
});

test("stream-copy playback stops at the detected scene boundary", () => {
    const preview = loadPreview();
    assert.equal(preview.playbackLimitSec({ duration: 3.0 }, 2.0), 2.0);
    assert.equal(preview.playbackLimitSec({ duration: 1.0 }, 2.0), 1.0);
    assert.equal(preview.playbackLimitSec({ duration: 3.0 }), 3.0);
    assert.equal(preview.clampSceneIndex(-1, 3), 2);
    assert.equal(preview.clampSceneIndex(3, 3), 0);
});

for (const phase of ["load", "reveal"]) {
    test(`failed clip ${phase} does not reject outside the preview handler`, async () => {
        const incoming = {
            style: {}, dataset: {}, duration: 2, isConnected: true,
            addEventListener() {}, pause() {},
            play() { return Promise.resolve(); },
        };
        const preview = loadPreview({
            document: { createElement: () => incoming },
        });
        const stage = { children: [], appendChild(child) { this.children.push(child); } };
        const node = {
            _psdPreviewEntries: [{ filename: "broken.mp4", type: "temp" }],
            _psdPreviewStage: stage,
        };
        if (phase === "load") {
            preview.waitForStartFrame = () => Promise.reject(new Error("broken clip"));
        } else {
            preview.waitForStartFrame = () => Promise.resolve();
            preview.waitForPresentedFrame = () => Promise.reject(new Error("frame unavailable"));
        }
        preview.showPreviewScene(node, 0);
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(stage.children.length, 1);
        assert.match(incoming.src, /broken.mp4/);
    });
}
