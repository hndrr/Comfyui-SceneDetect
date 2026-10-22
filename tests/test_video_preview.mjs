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
    for (const aspect of [9 / 16, 16 / 9, 3, 1]) {
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


test("rerunning the node cannot reveal a stale pending clip", async () => {
    class Element {
        constructor(tag) {
            this.tag = tag; this.style = {}; this.dataset = {};
            this.children = []; this.isConnected = true; this.duration = 2;
        }
        append(...children) { for (const child of children) this.appendChild(child); }
        appendChild(child) { this.children.push(child); child.parent = this; }
        replaceChildren() { this.children = []; }
        addEventListener() {}
        querySelector() { return null; }
        pause() {}
        load() {}
        play() { return Promise.resolve(); }
        removeAttribute(name) { delete this[name]; }
        getAttribute(name) { return this[name]; }
        remove() {
            if (this.parent) this.parent.children = this.parent.children.filter(x => x !== this);
            this.isConnected = false;
        }
    }
    const pending = new Map();
    const preview = loadPreview({document: {createElement: tag => new Element(tag)}});
    preview.waitForStartFrame = video => new Promise(resolve => pending.set(video, resolve));
    preview.waitForPresentedFrame = () => Promise.resolve();
    const node = {
        size: [320, 320], widgets: [],
        addDOMWidget() { return {}; },
        computeSize() { return this.size; },
        setSize(size) { this.size = size; },
    };
    preview.renderVideoPreviews(node, [{filename: "old.mp4"}]);
    const old = [...pending.keys()][0];
    preview.renderVideoPreviews(node, [{filename: "new.mp4"}]);
    const fresh = [...pending.keys()][1];
    pending.get(fresh)();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(node._psdPreviewVideo, fresh);
    pending.get(old)();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(node._psdPreviewVideo, fresh);
    assert.equal(old.isConnected, false);
    assert.equal(old.src, undefined);
});


test("a truncated stream-copy clip restarts without waiting for a paused frame callback", async () => {
    const events = new Map();
    let plays = 0;
    const video = {
        duration: 3, currentTime: 3, isConnected: true,
        addEventListener(name, handler) { events.set(name, handler); },
        requestVideoFrameCallback() {}, pause() {},
        play() { plays++; return Promise.resolve(); },
    };
    const preview = loadPreview();
    preview.coverFromVideo = () => {};
    preview.hideCover = () => {};
    preview.waitForEventOrTimeout = () => Promise.resolve();
    preview.waitForPresentedFrame = () => new Promise(() => {});
    preview.attachLoopFromStart(video, 2, {});
    assert.equal(typeof events.get("ended"), "function");
    events.get("ended")();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(video.currentTime, 0);
    assert.equal(plays, 1);
});


test("whole-file previews use native media looping for one-frame clips", () => {
    const preview = loadPreview();
    const video = {style: {}};
    preview.stylePreviewVideo(video);
    assert.equal(video.loop, true);
    assert.equal(video.style.objectFit, "contain");
});


test("a one-frame clip can reveal even when no further frame callback arrives", async () => {
    let expire;
    let cancelled;
    let cleared;
    const preview = loadPreview({
        setTimeout(callback) { expire = callback; return 9; },
        clearTimeout(id) { cleared = id; },
    });
    const video = {
        requestVideoFrameCallback() { return 7; },
        cancelVideoFrameCallback(id) { cancelled = id; },
    };
    const presented = preview.waitForPresentedFrame(video);
    expire();
    await presented;
    assert.equal(cancelled, 7);
    assert.equal(cleared, 9);
});
