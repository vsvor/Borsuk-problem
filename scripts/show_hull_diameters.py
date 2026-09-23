#!/usr/bin/env python3
"""Inspect diameters of convex hulls in the supplied partition-file format.

Install:
    python -m pip install numpy scipy matplotlib plotly
Native interactive view (no WebGL):
    python show_hull_diameters.py r_dod_4_0966.txt --target-vertices 23 --matplotlib
Browser view:
    python show_hull_diameters.py r_dod_4_0966.txt --target-vertices 23 --open
Static image:
    python show_hull_diameters.py r_dod_4_0966.txt --hull 3 --png K3.png

By default the output is one self-contained, interactive HTML file. No server or network
connection is needed to view it. Drag to rotate; scroll to zoom. Select a hull,
show its diameter or every overlong pair, and click a pair to isolate it.

Input: hull count, reference diameter, point count (the first three lines), then
one Python-literal list of index lists and one Python-literal list of 3D points.
The lists may span multiple lines. Indexing is zero-based, exactly as in the file.
No coordinates, index lists, or distances are modified, snapped, or normalized.

Classification: a pair is flagged precisely when length > threshold + tolerance.
The tolerance is an absolute comparison margin, NOT an error bound on the data.
For rendering only, edges separating almost coplanar triangles are suppressed.

API references:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
https://plotly.com/python/3d-mesh/
https://plotly.com/python/interactive-html-export/
"""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
from pathlib import Path
import sys
import webbrowser

try:
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError
    from scipy.spatial.distance import pdist, squareform
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}\n"
        "Install with: python -m pip install numpy scipy"
    ) from exc


def read_partition(path: Path) -> tuple[float, np.ndarray, list[list[int]]]:
    """Read and validate a file without evaluating executable Python code."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if len(lines) < 5:
        raise ValueError("Expected three header lines followed by two list expressions.")
    try:
        count, stored, n = int(lines[0]), float(lines[1]), int(lines[2])
        expressions = ast.parse("\n".join(lines[3:]), mode="exec").body
        if len(expressions) != 2 or not all(isinstance(e, ast.Expr) for e in expressions):
            raise ValueError("Expected exactly two list expressions after the header.")
        groups = ast.literal_eval(expressions[0].value)
        coordinates = ast.literal_eval(expressions[1].value)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid partition-file format: {exc}") from exc
    if count < 1 or n < 4 or not math.isfinite(stored) or stored <= 0:
        raise ValueError("Hull/point counts and reference diameter must be positive.")
    points = np.asarray(coordinates, dtype=float)
    if points.shape != (n, 3) or not np.isfinite(points).all():
        raise ValueError(f"Expected {n} finite 3D points; got shape {points.shape}.")
    if not isinstance(groups, (list, tuple)) or len(groups) != count:
        raise ValueError(f"Expected {count} index lists.")
    for k, indices in enumerate(groups, 1):
        if not isinstance(indices, (list, tuple)) or len(indices) < 4:
            raise ValueError(f"K{k}: at least four point indices are required.")
        if any(type(i) is not int or not 0 <= i < n for i in indices):
            raise ValueError(f"K{k}: indices must be integers between 0 and {n - 1}.")
        if len(set(indices)) != len(indices):
            raise ValueError(f"K{k}: duplicate point indices in its input list.")
    return stored, points, [list(g) for g in groups]


def mesh_data(points: np.ndarray, name: str) -> dict:
    """Triangulate the true input hull; discard coplanar diagonals for display."""
    try:
        hull = ConvexHull(points)  # Deliberately no QJ / random coordinate joggling.
    except QhullError as exc:
        raise ValueError(f"{name} is not a numerically full-dimensional 3D hull: {exc}") from exc
    adjacent: dict[tuple[int, int], list[int]] = {}
    for face, triangle in enumerate(hull.simplices):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            adjacent.setdefault(edge, []).append(face)
    normals = hull.equations[:, :3]
    edges = [list(edge) for edge, faces in adjacent.items()
             if len(faces) != 2 or np.linalg.norm(normals[faces[0]] - normals[faces[1]]) > 1e-5]
    return {"xyz": points.tolist(), "triangles": hull.simplices.tolist(),
            "edges": edges, "vertices": hull.vertices.tolist()}


def measure_hulls(points: np.ndarray, groups: list[list[int]]) -> list[dict]:
    """Check every generating-point pair, not just pairs joined by hull edges."""
    result = []
    for k, indices in enumerate(groups, 1):
        xyz = points[indices]
        mesh = mesh_data(xyz, f"K{k}")
        distances = squareform(pdist(xyz, metric="euclidean"))
        pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = sorted((indices[i], indices[j]))
                pairs.append({"a": a, "b": b, "length": float(distances[i, j])})
        pairs.sort(key=lambda pair: (-pair["length"], pair["a"], pair["b"]))
        result.append({"number": k, "ids": indices, "mesh": mesh,
                       "diameter": pairs[0]["length"], "pairs": pairs})
    return result


HTML_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — convex hull diameters</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; font-size: 14px; line-height: 1.45; }
header { padding: 17px 24px 8px; border-bottom: 1px solid; }
h1 { margin: 0 0 3px; font-size: 23px; font-weight: 650; }
.subtitle { margin: 0 0 12px; opacity: .8; }
.controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px 17px; margin-bottom: 10px; }
.controls label > span { display: block; font-size: 12px; margin-bottom: 3px; }
input, select, button { font: inherit; }
input[type=number] { width: 180px; padding: 5px; }
input#tolerance { width: 108px; }
select, button { padding: 5px 9px; cursor: pointer; }
.toggles { display: flex; flex-wrap: wrap; gap: 9px 18px; margin-bottom: 10px; }
.workspace { display: grid; grid-template-columns: minmax(0, 1fr) 435px; }
#plot { width: 100%; height: calc(100vh - 205px); min-height: 550px; }
aside { padding: 18px 20px 24px; border-left: 1px solid; overflow: auto;
        max-height: calc(100vh - 205px); min-height: 550px; }
h2 { margin: 4px 0 8px; font-size: 17px; }
h3 { margin: 22px 0 6px; font-size: 15px; }
p { margin: 8px 0; }
.small { font-size: 12px; opacity: .85; }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 4px; text-align: right; white-space: nowrap; border-bottom: 1px solid; }
th:first-child, td:first-child { text-align: left; }
.table-wrap { overflow-x: auto; }
#summary button, #pairs button { padding: 2px 5px; font-size: 12px; }
#selection { padding: 10px 0; font-variant-numeric: tabular-nums; }
#coordinates { white-space: pre-wrap; word-break: break-word; font: 12px ui-monospace, monospace; }
#notice { padding-top: 12px; }
.sample { display: inline-block; width: 27px; border-top: 5px solid; margin-right: 8px; vertical-align: middle; }
#noscript { padding: 24px; }
@media (max-width: 1000px) {
  .workspace { grid-template-columns: 1fr; }
  #plot { height: 600px; min-height: 450px; }
  aside { border-left: 0; border-top: 1px solid; max-height: none; min-height: 0; }
}
</style>
<script>__PLOTLY__</script>
</head><body>
<header>
<h1>Convex hull diameter inspector</h1>
<p class="subtitle">__TITLE__ · __COUNTS__ · original coordinates and zero-based point indices</p>
<div class="controls">
<label><span>Hull</span><select id="hull"></select></label>
<label><span>Segments</span><select id="mode">
<option value="diameters">Maximum diameter of each hull</option>
<option value="violations">All overlong pairs</option>
<option value="selected">Selected pair only</option>
</select></label>
<label><span>Diameter threshold d</span><input id="threshold" type="number" min="0" step="any"></label>
<label><span>Absolute tolerance</span><input id="tolerance" type="number" min="0" step="any"></label>
<button id="reset">Reset camera</button><button id="download">Export flagged pairs (CSV)</button>
</div>
<div class="toggles">
<label><input id="surface" type="checkbox" checked> Transparent hull surfaces</label>
<label><input id="wireframe" type="checkbox" checked> Hull edges</label>
<label><input id="labels" type="checkbox"> All point numbers</label>
<label><input id="lengths" type="checkbox" checked> Segment lengths</label>
<label id="reference-label"><input id="reference" type="checkbox" checked> Reference polyhedron wireframe</label>
</div>
<p id="webgl-notice" hidden></p>
</header>
<main class="workspace">
<div id="plot" aria-label="Interactive 3D convex hull and diameter segments"></div>
<aside>
<h2>Measured diameters</h2>
<div class="table-wrap"><table id="summary"><thead><tr>
<th>Hull</th><th>Diameter</th><th>Pair</th><th>Flagged</th>
</tr></thead><tbody></tbody></table></div>
<p class="small">A pair is flagged when its length &gt; d + tolerance. The tolerance is a comparison margin, not a certified data-error bound.</p>
<div id="selection"></div>
<h3>Overlong pairs in the displayed hulls</h3>
<div class="table-wrap"><table id="pairs"><thead><tr>
<th>Inspect</th><th>Length</th><th>Excess over d</th>
</tr></thead><tbody></tbody></table></div>
<p id="pair-count" class="small"></p>
<div id="coordinates"></div>
<h3>Using the view</h3>
<p><span class="sample"></span>Thick segments are the measured pairs; their endpoints are labeled.</p>
<p class="small">Drag to rotate; scroll to zoom. Click a hull above to isolate it. Click a pair to show only that segment. Hover over points or segments for precise values. The camera is preserved when you switch hulls.</p>
<p class="small">Distances are computed over every pair in each input list. A diameter is generally a chord through the solid, not an edge of its surface. Equal axis scales preserve the geometry.</p>
<p id="notice" class="small"></p>
</aside></main>
<noscript><p id="noscript">Enable JavaScript in your browser to display the interactive 3D scene.</p></noscript>
<script>
"use strict";
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const graph = $("plot");
const probe = document.createElement("canvas");
window.viewerWebGLAvailable = !!(probe.getContext("webgl") || probe.getContext("experimental-webgl"));
if (!window.viewerWebGLAvailable) {
    $("webgl-notice").hidden = false;
    $("webgl-notice").textContent = "WebGL is unavailable in this browser. The tables remain usable. For a native 3D view run: python show_hull_diameters.py " + DATA.filename + " --matplotlib";
}
const defaultCamera = {eye: {x: 1.5, y: 1.4, z: 1.1}, projection: {type: "orthographic"}};
let selectedPair = null;
let currentThreshold = DATA.threshold;
let currentTolerance = DATA.tolerance;
let rendering = Promise.resolve();
let camera = structuredClone(defaultCamera);

function fixed(x) { return x.toFixed(12); }
function isBad(pair) { return pair.length > currentThreshold + currentTolerance; }
function badPairs(g) { return g.pairs.filter(isBad); }
function selectedGroups() {
    const choice = $("hull").value;
    if (choice === "all") return DATA.groups;
    if (choice === "bad") return DATA.groups.filter(g => isBad(g.pairs[0]));
    return DATA.groups.filter(g => String(g.number) === choice);
}
function addOption(value, text) {
    const opt = document.createElement("option"); opt.value = value; opt.textContent = text;
    $("hull").appendChild(opt);
}
addOption("bad", "All offending hulls"); addOption("all", "All hulls");
for (const g of DATA.groups) addOption(String(g.number), "K" + g.number);
$("hull").value = DATA.initialHull;
$("mode").value = DATA.initialMode;
$("threshold").value = DATA.threshold;
$("tolerance").value = DATA.tolerance;
$("reference-label").hidden = DATA.reference === null;
if (!DATA.reference) $("reference").checked = false;
$("notice").textContent = DATA.reference
    ? `Reference wireframe: convex hull of the first ${DATA.referenceCount} input points (p0–p${DATA.referenceCount-1}). It is for orientation only; the partition hulls are not clipped to it. The entire viewer runs locally, without a server.`
    : "The entire viewer runs locally, without a server. No target polyhedron has been assumed.";

function selectHull(number) {
    $("hull").value = String(number);
    if ($("mode").value === "selected") $("mode").value = "diameters";
    selectedPair = null; requestRender();
}
function selectPair(g, p) {
    selectedPair = {group: g.number, pair: p};
    $("hull").value = String(g.number); $("mode").value = "selected";
    requestRender();
}
function cell(row, text) {
    const element = document.createElement("td"); element.textContent = text;
    row.appendChild(element); return element;
}
function drawTables(groups) {
    const summary = $("summary").tBodies[0]; summary.replaceChildren();
    for (const g of DATA.groups) {
        const row = document.createElement("tr");
        const button = document.createElement("button"); button.textContent = "K" + g.number;
        button.onclick = () => selectHull(g.number);
        cell(row, "").appendChild(button);
        cell(row, fixed(g.diameter));
        cell(row, `${g.pairs[0].a}–${g.pairs[0].b}`);
        cell(row, String(badPairs(g).length)); summary.appendChild(row);
    }
    const table = $("pairs").tBodies[0]; table.replaceChildren(); let count = 0;
    for (const g of groups) for (const p of badPairs(g)) {
        count++;
        const row = document.createElement("tr");
        const button = document.createElement("button");
        button.textContent = `K${g.number}: ${p.a}–${p.b}`;
        button.onclick = () => selectPair(g, p); cell(row, "").appendChild(button);
        cell(row, fixed(p.length)); cell(row, "+" + fixed(p.length - currentThreshold));
        table.appendChild(row);
    }
    $("pair-count").textContent = `${count} flagged pair${count === 1 ? "" : "s"} in this view.`;
    $("selection").replaceChildren();
    for (const g of groups) {
        const text = document.createElement("p");
        text.textContent = `K${g.number}: ${g.ids.length} listed points, ${g.mesh.vertices.length} extreme vertices; diameter ${fixed(g.diameter)} ${isBad(g.pairs[0]) ? "exceeds" : "does not exceed"} the threshold plus tolerance.`;
        $("selection").appendChild(text);
    }
    if (!groups.length) $("selection").textContent = "No hull exceeds the threshold plus tolerance. Select All hulls to inspect the geometry.";
    if ($("mode").value === "selected" && selectedPair) {
        const p = selectedPair.pair;
        $("coordinates").textContent =
            `Selected K${selectedPair.group}: p${p.a}–p${p.b}\n` +
            `p${p.a} = (${DATA.points[p.a].map(x => x.toPrecision(16)).join(", ")})\n` +
            `p${p.b} = (${DATA.points[p.b].map(x => x.toPrecision(16)).join(", ")})\n` +
            `distance = ${p.length.toPrecision(16)}`;
    } else $("coordinates").textContent = "";
}
function lineCoordinates(mesh) {
    const x = [], y = [], z = [];
    for (const [a,b] of mesh.edges) {
        x.push(mesh.xyz[a][0], mesh.xyz[b][0], null);
        y.push(mesh.xyz[a][1], mesh.xyz[b][1], null);
        z.push(mesh.xyz[a][2], mesh.xyz[b][2], null);
    }
    return {x,y,z};
}
function edgeTrace(mesh, name, reference) {
    return {...lineCoordinates(mesh), type:"scatter3d", mode:"lines", name,
            line:{width:reference ? 1.5 : 2, dash:reference ? "dot" : "solid"},
            opacity:reference ? .35 : .7, hoverinfo:"skip", showlegend:false};
}
function pointTrace(ids, endpoints) {
    return {type:"scatter3d", x:ids.map(i=>DATA.points[i][0]),
        y:ids.map(i=>DATA.points[i][1]), z:ids.map(i=>DATA.points[i][2]),
        mode:(endpoints || $("labels").checked) ? "markers+text" : "markers",
        text:ids.map(i=>"p"+i), textposition:"top center", textfont:{size:endpoints ? 14 : 10},
        marker:{size:endpoints ? 6 : 2.5, symbol:endpoints ? "diamond" : "circle"},
        customdata:ids, showlegend:false,
        hovertemplate:"p%{customdata}<br>x=%{x:.15f}<br>y=%{y:.15f}<br>z=%{z:.15f}<extra></extra>"};
}
function buildTraces(groups) {
    const traces = [], pointIds = new Set(), endpointIds = new Set();
    if (DATA.reference && $("reference").checked)
        traces.push(edgeTrace(DATA.reference, "Reference polyhedron", true));
    for (const g of groups) {
        const mesh = g.mesh;
        if ($("surface").checked) traces.push({
            type:"mesh3d", x:mesh.xyz.map(p=>p[0]), y:mesh.xyz.map(p=>p[1]),
            z:mesh.xyz.map(p=>p[2]), i:mesh.triangles.map(t=>t[0]),
            j:mesh.triangles.map(t=>t[1]), k:mesh.triangles.map(t=>t[2]),
            opacity:.13, flatshading:true, name:"K"+g.number,
            hoverinfo:"skip", showlegend:false
        });
        if ($("wireframe").checked) traces.push(edgeTrace(mesh, "K"+g.number+" edges", false));
        for (const id of g.ids) pointIds.add(id);
        let pairs;
        if ($("mode").value === "violations") pairs = badPairs(g);
        else if ($("mode").value === "selected")
            pairs = selectedPair && selectedPair.group === g.number ? [selectedPair.pair] : [];
        else pairs = [g.pairs[0]];
        for (const p of pairs) {
            endpointIds.add(p.a); endpointIds.add(p.b);
            const a = DATA.points[p.a], b = DATA.points[p.b];
            const m = a.map((x,j)=>(x+b[j])/2);
            const label = `K${g.number}: p${p.a}–p${p.b}`;
            const info = `${label}<br>length=${p.length.toPrecision(16)}<br>` +
                         `excess over d=${(p.length-currentThreshold).toPrecision(12)}`;
            traces.push({type:"scatter3d", x:[a[0],m[0],b[0]], y:[a[1],m[1],b[1]],
                z:[a[2],m[2],b[2]], mode:"lines", name:label,
                line:{width:7, dash:p === g.pairs[0] ? "solid" : "dash"},
                text:[info,info,info], hovertemplate:"%{text}<extra></extra>", showlegend:false,
                meta:{kind:"measured-pair", hull:g.number, a:p.a, b:p.b, length:p.length}});
            if ($("lengths").checked) traces.push({type:"scatter3d", x:[m[0]], y:[m[1]], z:[m[2]],
                mode:"text", text:[`${p.a}–${p.b}<br>${p.length.toFixed(9)}`],
                textposition:"top center", textfont:{size:12}, hoverinfo:"skip", showlegend:false});
        }
    }
    const remaining = Array.from(pointIds).filter(i=>!endpointIds.has(i)).sort((a,b)=>a-b);
    if (remaining.length) traces.push(pointTrace(remaining, false));
    if (endpointIds.size) traces.push(pointTrace(Array.from(endpointIds).sort((a,b)=>a-b), true));
    return traces;
}
const ranges = [0,1,2].map(j => {
    const v = DATA.points.map(p=>p[j]); return [Math.min(...v),Math.max(...v)];
});
const extent = Math.max(...ranges.map(r=>r[1]-r[0])) * .6;
const axis = j => ({title:["x","y","z"][j],
    range:[(ranges[j][0]+ranges[j][1])/2-extent,(ranges[j][0]+ranges[j][1])/2+extent],
    nticks:5, showspikes:false});

async function render() {
    if (!$("threshold").checkValidity() || !$("tolerance").checkValidity() ||
        $("threshold").value === "" || $("tolerance").value === "") return;
    const d = Number($("threshold").value), tol = Number($("tolerance").value);
    if (!Number.isFinite(d) || d <= 0 || !Number.isFinite(tol) || tol < 0) return;
    currentThreshold = d; currentTolerance = tol;
    const groups = selectedGroups();
    if ($("mode").value === "selected" && !selectedPair && groups.length)
        selectedPair = {group:groups[0].number, pair:groups[0].pairs[0]};
    drawTables(groups);
    const title = groups.length ? groups.map(g=>"K"+g.number).join(" + ") : "No offending hulls";
    await Plotly.react(graph, buildTraces(groups), {
        title:{text:title, font:{size:19}, x:.04}, margin:{l:8,r:8,t:40,b:8},
        scene:{aspectmode:"cube", xaxis:axis(0), yaxis:axis(1), zaxis:axis(2), camera},
        uirevision:"preserve-camera", showlegend:false, autosize:true
    }, {responsive:true, displaylogo:false, scrollZoom:true,
        toImageButtonOptions:{format:"png",filename:"hull_diameters",scale:2}});
    window.viewerReady = true;
}
function requestRender() {
    rendering = rendering.then(render).catch(error => {
        $("selection").textContent = "Rendering error: " + error.message;
        console.error(error); throw error;
    });
    return rendering;
}
for (const id of ["threshold","tolerance","surface","wireframe","labels","lengths","reference"])
    $(id).addEventListener("change", requestRender);
$("hull").addEventListener("change", () => {
    selectedPair = null; if ($("mode").value === "selected") $("mode").value = "diameters";
    requestRender();
});
$("mode").addEventListener("change", requestRender);
$("reset").onclick = () => {
    camera = structuredClone(defaultCamera); return Plotly.relayout(graph, {"scene.camera":camera});
};
$("download").onclick = () => {
    const rows = ["hull,point_a,point_b,length,threshold,tolerance,excess"];
    for (const g of DATA.groups) for (const p of badPairs(g))
        rows.push([g.number,p.a,p.b,p.length,currentThreshold,currentTolerance,
                   p.length-currentThreshold].join(","));
    const url = URL.createObjectURL(new Blob([rows.join("\n")+"\n"], {type:"text/csv"}));
    const a = document.createElement("a"); a.href = url; a.download = "overlong_pairs.csv";
    a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
};
requestRender().then(() => graph.on("plotly_relayout", event => {
    if (event["scene.camera"]) camera = structuredClone(event["scene.camera"]);
}));
</script></body></html>
'''


def write_viewer(path: Path, payload: dict) -> None:
    """Embed the data and the installed Plotly JavaScript bundle, without a CDN."""
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise ValueError("HTML output requires Plotly: python -m pip install plotly; "
                         "or use --matplotlib for the native viewer.") from exc
    safe_data = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    page = (HTML_PAGE.replace("__TITLE__", html.escape(payload["filename"]))
            .replace("__COUNTS__", f'{len(payload["points"])} points · {len(payload["groups"])} hulls')
            .replace("__DATA__", safe_data)
            .replace("__PLOTLY__", get_plotlyjs()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def write_csv(path: Path, groups: list[dict], threshold: float, tolerance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hull", "point_a", "point_b", "length", "threshold", "tolerance", "excess"])
        for g in groups:
            for p in g["pairs"]:
                if p["length"] > threshold + tolerance:
                    writer.writerow([g["number"], p["a"], p["b"], repr(p["length"]),
                                     repr(threshold), repr(tolerance), repr(p["length"] - threshold)])


def matplotlib_view(payload: dict, save_path: Path | None = None, show: bool = True):
    """Open a native interactive view, or save one PNG without a GUI/WebGL.

    The return value is the Matplotlib Figure, so callers can further customize
    it. Its ``_diameter_controls`` attribute retains the radio/check widgets.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, CheckButtons
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    except ImportError as exc:
        raise ValueError("Native/PNG output requires: python -m pip install matplotlib") from exc

    points = np.asarray(payload["points"])
    groups = payload["groups"]
    threshold, tolerance = payload["threshold"], payload["tolerance"]
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, radius = (lo + hi) / 2, float(max(hi - lo)) * .59
    fig = plt.figure(figsize=(12.8, 8.5))
    # Explicit z-order makes diameter chords visible through transparent faces.
    ax = fig.add_axes([.25, .11, .74, .78], projection="3d", computed_zorder=False)
    fig.suptitle(f'{payload["filename"]} — convex hull diameters\n'
                 f'd = {threshold:.15f}; absolute flagging tolerance = {tolerance:g}',
                 fontsize=13, y=.97)
    choice_labels = [f'K{g["number"]}' for g in groups] + ["All offending", "All hulls"]
    choice_values = [str(g["number"]) for g in groups] + ["bad", "all"]
    rax = fig.add_axes([.03, .55, .19, .29])
    rax.set_title("Displayed hull", fontsize=11, loc="left")
    radios = RadioButtons(rax, choice_labels, active=choice_values.index(payload["initialHull"]))
    cax = fig.add_axes([.03, .285, .19, .215])
    check_labels = ["All overlong pairs", "Surface", "All point numbers", "Lengths", "Reference hull"]
    checks = CheckButtons(cax, check_labels,
                          [payload["initialMode"] == "violations", True, False, True,
                           payload["reference"] is not None])
    for label in checks.labels:
        label.set_fontsize(10)
    summary = "\n".join(f'K{g["number"]}: {g["diameter"]:.12f}\n'
                        f'    p{g["pairs"][0]["a"]} — p{g["pairs"][0]["b"]}' for g in groups)
    fig.text(.03, .25, summary, va="top", fontsize=9, family="monospace", linespacing=1.25)
    footer = fig.text(.28, .04, "", fontsize=10, va="bottom")

    def draw(_event=None):
        azim, elev = ax.azim, ax.elev
        ax.clear()
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        for j, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(center[j] - radius, center[j] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        all_pairs, surface, labels, lengths, reference = checks.get_status()
        choice = choice_values[choice_labels.index(radios.value_selected)]
        if choice == "all":
            chosen = groups
        elif choice == "bad":
            chosen = [g for g in groups if g["diameter"] > threshold + tolerance]
        else:
            chosen = [g for g in groups if str(g["number"]) == choice]
        if reference and payload["reference"] is not None:
            ref = payload["reference"]
            segments = np.asarray(ref["xyz"])[np.asarray(ref["edges"], dtype=int)]
            ax.add_collection3d(Line3DCollection(segments, linewidths=.8, linestyles=":",
                                               alpha=.4, zorder=1))
        endpoint_ids, all_ids = set(), set()
        displayed = []
        for g in chosen:
            mesh = g["mesh"]
            xyz = np.asarray(mesh["xyz"])
            all_ids.update(g["ids"])
            if surface:
                ax.add_collection3d(Poly3DCollection(xyz[np.asarray(mesh["triangles"])],
                                                     alpha=.12, linewidths=0, zorder=2))
            ax.add_collection3d(Line3DCollection(xyz[np.asarray(mesh["edges"])],
                                                linewidths=.8, alpha=.65, zorder=3))
            pairs = ([p for p in g["pairs"] if p["length"] > threshold + tolerance]
                     if all_pairs else [g["pairs"][0]])
            for p in pairs:
                a, b = p["a"], p["b"]
                endpoint_ids.update((a, b))
                segment = points[[a, b]]
                maximum = p is g["pairs"][0]
                ax.plot(*segment.T, linewidth=3.5 if maximum else 2.4,
                        linestyle="-" if maximum else "--", zorder=10)
                if lengths:
                    m = segment.mean(axis=0)
                    ax.text(*m, f'  {a}–{b}\n  {p["length"]:.9f}', fontsize=10, zorder=13)
                displayed.append((g["number"], a, b, p["length"]))
        if all_ids:
            ordinary = sorted(all_ids - endpoint_ids)
            if ordinary:
                ax.scatter(*points[ordinary].T, s=12, depthshade=False, zorder=4)
            if endpoint_ids:
                ids = sorted(endpoint_ids)
                ax.scatter(*points[ids].T, s=52, marker="D", depthshade=False, zorder=11)
            for i in sorted(all_ids if labels else endpoint_ids):
                ax.text(*points[i], f'  p{i}', fontsize=11 if i in endpoint_ids else 8,
                        fontweight="bold" if i in endpoint_ids else "normal", zorder=12)
        heading = " + ".join(f'K{g["number"]}' for g in chosen) or "No offending hulls"
        ax.set_title(heading + (" — all overlong pairs" if all_pairs else " — maximum diameter"), pad=12)
        nbad = sum(p["length"] > threshold + tolerance for g in chosen for p in g["pairs"])
        footer.set_text(f'{len(displayed)} segment(s) displayed; {nbad} overlong pair(s) in the selected hulls.\n'
                        'Drag the 3D view to rotate. Thick chords show the measured pairs. Indices are zero-based.')
        # Useful to callers/tests; these are the actual rendered measurement records.
        fig._diameter_displayed_pairs = displayed
        fig.canvas.draw_idle()

    radios.on_clicked(draw)
    checks.on_clicked(draw)
    fig._diameter_controls = (radios, checks, draw)
    ax.view_init(elev=23, azim=-62)
    draw()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Partition data file")
    parser.add_argument("--output", "-o", type=Path, help="HTML path (default: <input stem>_diameters.html)")
    parser.add_argument("--threshold", type=float, help="Diameter limit (default: value on line 2)")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute flagging margin (default: 1e-6)")
    parser.add_argument("--hull", default="auto", help="Initial hull: 1, 2, ..., all, bad, or auto (first bad hull)")
    parser.add_argument("--all-pairs", action="store_true", help="Initially show every overlong pair")
    parser.add_argument("--target-vertices", type=int, metavar="N",
                        help="Optional reference wireframe: convex hull of the first N points")
    parser.add_argument("--csv", type=Path, help="Also write the overlong pairs to CSV")
    parser.add_argument("--matplotlib", action="store_true", help="Open a native 3D window instead of HTML")
    parser.add_argument("--png", type=Path, help="Save the initial native view as a PNG (no GUI required)")
    parser.add_argument("--open", action="store_true", help="Open the HTML in your default browser")
    args = parser.parse_args(argv)
    try:
        stored, points, index_lists = read_partition(args.input)
        threshold = stored if args.threshold is None else args.threshold
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("--threshold must be finite and positive.")
        if not math.isfinite(args.tol) or args.tol < 0:
            raise ValueError("--tol must be finite and nonnegative.")
        groups = measure_hulls(points, index_lists)
        reference = None
        if args.target_vertices is not None:
            if not 4 <= args.target_vertices <= len(points):
                raise ValueError(f"--target-vertices must be between 4 and {len(points)}.")
            reference = mesh_data(points[:args.target_vertices], "Reference polyhedron")
        allowed = {"auto", "all", "bad"} | {str(g["number"]) for g in groups}
        if args.hull not in allowed:
            raise ValueError("--hull must be auto, all, bad, or a valid one-based hull number.")
        bad = [g for g in groups if g["diameter"] > threshold + args.tol]
        initial = args.hull if args.hull != "auto" else str((bad or groups)[0]["number"])
        wants_html = not (args.matplotlib or args.png) or args.output is not None or args.open
        output = ((args.output or args.input.with_name(args.input.stem + "_diameters.html"))
                  if wants_html else None)
        input_resolved = args.input.resolve()
        outputs = [p for p in (output, args.csv, args.png) if p is not None]
        if any(p.resolve() == input_resolved for p in outputs):
            raise ValueError("An output path would overwrite the input data file.")
        if len({p.resolve() for p in outputs}) != len(outputs):
            raise ValueError("HTML, CSV, and PNG outputs must have different paths.")
        payload = {"filename": args.input.name, "points": points.tolist(), "groups": groups,
                   "stored": stored, "threshold": threshold, "tolerance": args.tol,
                   "reference": reference, "referenceCount": args.target_vertices,
                   "initialHull": initial, "initialMode": "violations" if args.all_pairs else "diameters"}
        if output is not None:
            write_viewer(output, payload)
        if args.csv:
            write_csv(args.csv, groups, threshold, args.tol)
        if args.matplotlib or args.png:
            matplotlib_view(payload, save_path=args.png, show=args.matplotlib)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Stored reference: {stored:.16g}\nThreshold: {threshold:.16g}; tolerance: {args.tol:.6g}")
    print("Hull  Points          Diameter      Farthest pair   Flagged pairs")
    for g in groups:
        p = g["pairs"][0]
        count = sum(q["length"] > threshold + args.tol for q in g["pairs"])
        print(f'K{g["number"]:<3}  {len(g["ids"]):>6}  {g["diameter"]:>18.12f}'
              f'      {p["a"]:>3}--{p["b"]:<3}        {count:>4}')
    if output is not None:
        print(f"HTML: {output.resolve()}")
    if args.png is not None:
        print(f"PNG:  {args.png.resolve()}")
    if args.csv:
        print(f"CSV:  {args.csv.resolve()}")
    if args.open and output is not None and not webbrowser.open(output.resolve().as_uri()):
        print("The browser did not open automatically; open the HTML file manually.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
