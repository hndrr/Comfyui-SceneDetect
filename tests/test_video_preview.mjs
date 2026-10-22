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
    querySelectorAll(tag) {
        return this.children.flatMap(child => [
            ...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag),
        ]);
    }
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

test("rerunning the node cannot reveal a stale pending clip", async () => {
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


function graphWith(nodes = []) {
    const graph = { getNodeById(id) { return nodes.find(node => String(node.id) === String(id)); } };
    for (const node of nodes) node.graph = graph;
    return graph;
}

async function scopeHarness() {
    let extension;
    let queueHandler = async () => ({prompt_id: "accepted"});
    const app = {registerExtension(value) {extension = value;}};
    const listeners = new Set();
    const api = {
        apiURL: path => path,
        queuePrompt(...args) { return queueHandler.apply(this, args); },
        addEventListener(name, listener) { assert.equal(name, "executed"); listeners.add(listener); },
        removeEventListener(name, listener) { assert.equal(name, "executed"); listeners.delete(listener); },
        emitExecuted(detail) { for (const listener of listeners) listener({detail}); },
        listenerCount() { return listeners.size; },
    };
    const preview = loadPreview({app, api, document: {createElement: tag => new Element(tag)}});
    class PreviewNode {
        constructor(id) { this.id = id; this.type = "PySceneDetectPreviewVideos"; this.size = [320, 320]; this.widgets = []; }
        addDOMWidget() { return {}; }
        computeSize() { return this.size; }
        setSize(size) { this.size = size; }
    }
    await extension.beforeRegisterNodeDef(PreviewNode, {name: "PySceneDetectPreviewVideos"});
    extension.setup();
    preview.waitForStartFrame = () => Promise.resolve();
    preview.waitForPresentedFrame = () => Promise.resolve();
    return {app, api, preview, makeNode(id) {
        const node = new PreviewNode(id); node.onNodeCreated(); return node;
    }, setQueueHandler(handler) {queueHandler = handler;}};
}

async function display(node, filename = "current.mp4") {
    node.onExecuted({scene_previews: [{filename}]});
    await new Promise(resolve => setImmediate(resolve));
    return node._psdPreviewVideo;
}

const previewOutput = inputs => ({class_type: "PySceneDetectPreviewVideos", inputs: inputs || {}});

test("an accepted run releases current/pending videos and cached results restore the preview", async () => {
    const h = await scopeHarness(); const node = h.makeNode(28);
    h.app.rootGraph = graphWith([node]);
    const old = await display(node);
    const pending = new Element("video"); pending.src = "pending.mp4";
    node._psdPreviewStage.appendChild(pending);
    await h.api.queuePrompt(0, {output: {28: previewOutput()}});
    assert.equal(old.src, undefined);
    assert.equal(pending.src, undefined);
    assert.equal(node._psdVideoPreviewContainer.querySelectorAll("video").length, 0);
    assert.match(node._psdVideoPreviewContainer.children[0].textContent, /No videos/);
    const restored = await display(node);
    assert.match(restored.src, /current.mp4/);
    node.onRemoved();
    assert.equal(restored.src, undefined);
});

test("partial execution resolves nested IDs without clearing a sibling preview with the same leaf ID", async () => {
    const h = await scopeHarness(); const selected = h.makeNode(28); const unrelated = h.makeNode(28);
    const left = {id: 10, subgraph: graphWith([{id: 20, subgraph: graphWith([selected])}])};
    const right = {id: 11, subgraph: graphWith([{id: 20, subgraph: graphWith([unrelated])}])};
    h.app.rootGraph = graphWith([left, right]); h.app.graph = left.subgraph;
    const oldSelected = await display(selected, "selected.mp4");
    const oldUnrelated = await display(unrelated, "unrelated.mp4");
    const output = {"10:20:28": previewOutput(), "11:20:28": previewOutput()};
    await h.api.queuePrompt(0, {output}, {partialExecutionTargets: ["10:20:28"]});
    assert.equal(oldSelected.src, undefined);
    assert.equal(unrelated._psdPreviewVideo, oldUnrelated);
    assert.match(oldUnrelated.src, /unrelated.mp4/);
});

test("running another output leaves both downstream previews outside its dependency closure", async () => {
    const h = await scopeHarness(); const first = h.makeNode(28); const second = h.makeNode(29);
    h.app.rootGraph = graphWith([first, second]);
    const oldFirst = await display(first); const oldSecond = await display(second);
    const output = {20: {class_type: "LoadVideo", inputs: {}},
        27: {class_type: "PySceneDetectVideo", inputs: {video: ["20", 0]}},
        28: previewOutput({videos: ["27", 3]}), 29: previewOutput({videos: ["20", 0]}),
        30: {class_type: "SaveVideo", inputs: {video: ["27", 3]}}};
    await h.api.queuePrompt(0, {output}, {partialExecutionTargets: ["30"]});
    assert.equal(first._psdPreviewVideo, oldFirst);
    assert.equal(second._psdPreviewVideo, oldSecond);
    assert.match(oldFirst.src, /current.mp4/);
    assert.match(oldSecond.src, /current.mp4/);
});

test("a fast cached result is not erased when the queue response arrives later", async () => {
    const h = await scopeHarness(); const node = h.makeNode(28);
    h.app.rootGraph = graphWith([node]); await display(node, "old.mp4");
    h.setQueueHandler(async () => {
        await display(node, "cached-current.mp4");
        h.api.emitExecuted({prompt_id: "fast", node: "28"});
        return {prompt_id: "fast"};
    });
    await h.api.queuePrompt(0, {output: {28: previewOutput()}});
    assert.match(node._psdPreviewVideo.src, /cached-current.mp4/);
    assert.equal(h.api.listenerCount(), 0);
});

test("another queued job's result cannot suppress clearing the accepted run", async () => {
    const h = await scopeHarness(); const node = h.makeNode(28);
    h.app.rootGraph = graphWith([node]); await display(node, "old.mp4");
    h.setQueueHandler(async () => {
        await display(node, "previous-job.mp4");
        h.api.emitExecuted({prompt_id: "previous-job", node: "28"});
        return {prompt_id: "current-job"};
    });
    await h.api.queuePrompt(0, {output: {28: previewOutput()}});
    assert.equal(node._psdPreviewVideo, null);
    assert.equal(h.api.listenerCount(), 0);
});

test("queue rejection and invalid outputs retain their previous previews", async () => {
    const h = await scopeHarness(); const node = h.makeNode(28);
    h.app.rootGraph = graphWith([node]); const old = await display(node);
    h.setQueueHandler(async () => {throw new Error("validation failed");});
    await assert.rejects(h.api.queuePrompt(0, {output: {28: previewOutput()}}), /validation failed/);
    assert.equal(node._psdPreviewVideo, old);
    assert.equal(h.api.listenerCount(), 0);
    h.setQueueHandler(async () => ({node_errors: {}}));
    await h.api.queuePrompt(0, {output: {28: previewOutput()}});
    assert.equal(node._psdPreviewVideo, old);
    h.setQueueHandler(async () => ({prompt_id: "other-valid-output", node_errors: {20: {dependent_outputs: ["28"]}}}));
    await h.api.queuePrompt(0, {output: {28: previewOutput()}});
    assert.equal(node._psdPreviewVideo, old);
    assert.match(old.src, /current.mp4/);
});

test("workflow switches cannot clear another workflow with reused node IDs; queue arguments and return are preserved", async () => {
    const h = await scopeHarness(); const queued = h.makeNode(28); const other = h.makeNode(28);
    const queuedGraph = graphWith([queued]); const otherGraph = graphWith([other]);
    h.app.rootGraph = queuedGraph;
    const oldQueued = await display(queued); const oldOther = await display(other);
    const prompt = {output: {28: previewOutput()}}; const options = {partialExecutionTargets: ["28"], previewMethod: "none"};
    const response = {prompt_id: "queued", node_errors: {}};
    h.setQueueHandler(function(...args) {
        assert.equal(this, h.api); assert.deepEqual(args, [7, prompt, options, "extra"]);
        h.app.rootGraph = otherGraph; return Promise.resolve(response);
    });
    assert.equal(await h.api.queuePrompt(7, prompt, options, "extra"), response);
    assert.equal(oldQueued.src, undefined);
    assert.equal(other._psdPreviewVideo, oldOther);
    assert.match(oldOther.src, /current.mp4/);
});
