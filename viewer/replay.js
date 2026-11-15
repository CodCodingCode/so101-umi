// In-browser sim replay of a UMI episode (used by studio.html -> Episodes).
// Draws the bench twin with three.js from data/episodes/<ts>/web/ (umi/web_export.py): no video, no encoding.
//   mountReplay(rootElement, { ep, side })  ->  { destroy() }
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CSS = `
.rp { --demo:#ff9f43; --robot:#3fd0ff; }
.rp .rp-grid { display:grid; grid-template-columns:1fr 260px; gap:12px; }
@media (max-width:1100px){ .rp .rp-grid{ grid-template-columns:1fr; } }
.rp .rp-stage { position:relative; background:#000; border-radius:8px; overflow:hidden; aspect-ratio:4/3; width:100%; }
.rp .rp-bg, .rp canvas { position:absolute; inset:0; width:100%; height:100%; }
.rp .rp-loading { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:var(--muted,#9a9ca3); background:#000c; z-index:2; }
.rp .rp-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:8px; }
.rp .rp-tl { flex:1; min-width:200px; }
.rp .rp-tl input { width:100%; }
.rp .rp-spans { position:relative; height:6px; background:#222; border-radius:3px; overflow:hidden; margin-top:2px; }
.rp .rp-spans div { position:absolute; top:0; height:100%; }
.rp .rp-time { font-variant-numeric:tabular-nums; min-width:105px; }
.rp .rp-banner { min-height:18px; color:var(--warn,#f9a825); margin-top:4px; font-size:13px; }
.rp .rp-card { background:var(--panel,#18191c); border:1px solid var(--border,#2e3036); border-radius:10px; padding:10px 12px; margin-bottom:10px; }
.rp .rp-card h4 { margin:0 0 6px; font-size:13px; }
.rp .rp-kv { display:grid; grid-template-columns:1fr auto; gap:3px 10px; font-variant-numeric:tabular-nums; font-size:13px; }
.rp .rp-kv .k { color:var(--muted,#9a9ca3); }
.rp .rp-big { font-size:20px; font-weight:600; }
.rp .rp-on { border-color:var(--accent,#4ea1ff)!important; color:var(--accent,#4ea1ff)!important; }
.rp .rp-legend span { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }
`;

export async function mountReplay(root, { ep, side = "right" }) {
  if (!document.getElementById("rp-css")) {
    const st = document.createElement("style");
    st.id = "rp-css";
    st.textContent = CSS;
    document.head.appendChild(st);
  }
  root.classList.add("rp");
  root.innerHTML = `
    <div class="rp-grid">
      <div>
        <div class="rp-stage"><img class="rp-bg" alt="" /><canvas></canvas><div class="rp-loading">Loading the sim…</div></div>
        <div class="rp-bar">
          <button class="small" data-r="play">▶</button>
          <div class="rp-tl"><input data-r="slider" type="range" min="0" max="0" value="0" step="1" /><div class="rp-spans"></div></div>
          <span class="rp-time" data-r="time">0.00 s</span>
          <select data-r="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
          <button class="small rp-on" data-r="v-overlay">On the real video</button>
          <button class="small" data-r="v-3d">3D</button>
          <label class="msg">Robot <input data-r="alpha" type="range" min="0" max="100" value="75" style="width:70px" /></label>
          <label class="msg"><input data-r="paths" type="checkbox" checked /> Paths</label>
        </div>
        <div class="rp-banner" data-r="banner"></div>
      </div>
      <div>
        <div class="rp-card"><h4>Right now</h4><div class="rp-kv">
          <span class="k">Robot vs your hand</span><span class="rp-big" data-r="now-miss">–</span>
          <span class="k">Gripper (0 closed – 100 open)</span><span data-r="now-grip">–</span>
          <span class="k">Handle tracked</span><span data-r="now-tracked">–</span></div></div>
        <div class="rp-card"><h4>Whole demo</h4><div class="rp-kv" data-r="summary"></div></div>
        <div class="rp-card rp-legend"><div><span style="background:var(--demo)"></span>your hand's path (jaw tip)</div>
          <div><span style="background:var(--robot)"></span>the robot's path</div>
          <div class="msg" style="margin-top:6px;font-size:12px">Space = play/pause · ← → = step · drag in 3D to rotate</div></div>
      </div>
    </div>`;
  const $ = (n) => root.querySelector(`[data-r="${n}"]`);
  const canvas = root.querySelector("canvas"),
    bg = root.querySelector(".rp-bg"),
    stage = root.querySelector(".rp-stage");
  const base = `/data/episodes/${ep}/web/`;
  let alive = true,
    raf = 0;

  const [scene, posesBuf, meshIdx, meshBuf] = await Promise.all([
    fetch(base + "scene.json").then((r) => {
      if (!r.ok) throw new Error("no sim data for this episode yet");
      return r.json();
    }),
    fetch(base + "poses.bin").then((r) => r.arrayBuffer()),
    fetch("/data/web/meshes.json").then((r) => r.json()),
    fetch("/data/web/meshes.bin").then((r) => r.arrayBuffer()),
  ]);
  const N = scene.frames,
    nd = scene.dynamic.length,
    poses = new Float32Array(posesBuf);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  const world = new THREE.Scene();
  world.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.3));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(0.3, -0.4, 1.2);
  world.add(sun);

  const geomCache = {};
  const meshGeometry = (id) => {
    if (geomCache[id]) return geomCache[id];
    const e = meshIdx.meshes[id],
      g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array(meshBuf, e.v_off, e.v_n * 3),
        3,
      ),
    );
    g.setIndex(
      new THREE.BufferAttribute(
        new Uint32Array(meshBuf, e.f_off, e.f_n * 3),
        1,
      ),
    );
    g.computeVertexNormals();
    return (geomCache[id] = g);
  };
  function makeObject(gi) {
    const [sx, sy, sz] = gi.size;
    let g;
    if (gi.kind === "mesh") g = meshGeometry(gi.mesh);
    else if (gi.kind === "box")
      g = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
    else if (gi.kind === "plane")
      g = new THREE.PlaneGeometry(2 * (sx || 1), 2 * (sy || 1));
    else if (gi.kind === "sphere") g = new THREE.SphereGeometry(sx, 24, 16);
    else if (gi.kind === "cylinder")
      g = new THREE.CylinderGeometry(sx, sx, 2 * sy, 32).rotateX(Math.PI / 2);
    else if (gi.kind === "capsule")
      g = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 16).rotateX(Math.PI / 2);
    else return null;
    const [r, gg, b, a] = gi.rgba;
    const o = new THREE.Mesh(
      g,
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, gg, b),
        roughness: 0.7,
        metalness: 0.05,
        transparent: true,
        opacity: a,
      }),
    );
    o.userData.baseOpacity = a;
    o.userData.scenery =
      gi.body === "world" ||
      /table|room|wall|offtable|floor/.test(gi.name || "");
    return o;
  }
  const setPose = (o, p) => {
    o.position.set(p[0], p[1], p[2]);
    o.quaternion.set(p[4], p[5], p[6], p[3]);
  };
  const staticObjs = [],
    dynObjs = [];
  for (const gi of scene.static) {
    const o = makeObject(gi);
    if (o) {
      setPose(o, gi.pose);
      world.add(o);
      staticObjs.push(o);
    }
  }
  for (const gi of scene.dynamic) {
    const o = makeObject(gi);
    if (o) world.add(o);
    dynObjs.push(o);
  }

  const lineOf = (pts, color) => {
    const l = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        pts.map((p) => new THREE.Vector3(...p)),
      ),
      new THREE.LineBasicMaterial({
        color,
        depthTest: false,
        transparent: true,
      }),
    );
    l.renderOrder = 10;
    return l;
  };
  const demoLine = lineOf(
    scene.demo_tcp.filter((_, k) => scene.tracked[k]),
    0xff9f43,
  );
  const robotLine = lineOf(scene.robot_tcp, 0x3fd0ff);
  world.add(demoLine, robotLine);
  const dot = (c) => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.006, 16, 12),
      new THREE.MeshBasicMaterial({ color: c, depthTest: false }),
    );
    m.renderOrder = 11;
    world.add(m);
    return m;
  };
  const demoDot = dot(0xff9f43),
    robotDot = dot(0x3fd0ff);

  // calibrated overhead camera (MuJoCo camera frame = three.js camera frame: -z forward, +y up)
  const cam = scene.camera,
    W = cam.width || 640,
    H = cam.height || 480;
  const overheadCam = new THREE.PerspectiveCamera(50, W / H, 0.01, 10);
  overheadCam.position.set(...cam.pos);
  const [qw, qx, qy, qz] = cam.quat_wxyz;
  overheadCam.quaternion.set(qx, qy, qz, qw);
  if (cam.K) {
    const [[fx, , cx], [, fy, cy]] = cam.K,
      n = 0.01,
      f = 10;
    overheadCam.projectionMatrix.set(
      (2 * fx) / W,
      0,
      1 - (2 * cx) / W,
      0,
      0,
      (2 * fy) / H,
      (2 * cy) / H - 1,
      0,
      0,
      0,
      -(f + n) / (f - n),
      (-2 * f * n) / (f - n),
      0,
      0,
      -1,
      0,
    );
    overheadCam.projectionMatrixInverse
      .copy(overheadCam.projectionMatrix)
      .invert();
    overheadCam.updateProjectionMatrix = () => {};
  }
  const freeCam = new THREE.PerspectiveCamera(45, W / H, 0.01, 10);
  freeCam.up.set(0, 0, 1);
  freeCam.position.set(0.55, -0.75, 0.5);
  const controls = new OrbitControls(freeCam, canvas);
  controls.target.set(0.05, 0.05, 0.05);
  controls.enabled = false;

  let pv = [];
  if (scene.preview) {
    const [bin, csv] = await Promise.all([
      fetch(base + scene.preview.bin).then((r) => r.arrayBuffer()),
      fetch(base + scene.preview.csv).then((r) => r.text()),
    ]);
    pv = csv
      .trim()
      .split("\n")
      .slice(1)
      .map((l) => {
        const [t, o, n] = l.split(",").map(Number);
        return {
          t,
          url: URL.createObjectURL(
            new Blob([new Uint8Array(bin, o, n)], { type: "image/jpeg" }),
          ),
        };
      });
  }
  const pvAt = (tc) => {
    let lo = 0,
      hi = pv.length - 1;
    if (hi < 0) return null;
    while (lo < hi) {
      const m = (lo + hi + 1) >> 1;
      if (pv[m].t <= tc) lo = m;
      else hi = m - 1;
    }
    return pv[lo];
  };

  const T = scene.t,
    dur = T[N - 1] || 1;
  $("slider").max = N - 1;
  const spans = scene.plan.clamped_spans_s || [];
  const spanColor = (w) =>
    /REACH|SAFE|CLOSEST|NO SAFE/i.test(w)
      ? "#f9a825"
      : /NOT SEEN/i.test(w)
        ? "#666"
        : /TABLE|FLOOR|TILT/i.test(w)
          ? "#4ea1ff"
          : "#e53935";
  for (const [a, b, why] of spans) {
    const d = document.createElement("div");
    d.style.left = `${(100 * a) / dur}%`;
    d.style.width = `${Math.max(0.4, (100 * (b - a)) / dur)}%`;
    d.style.background = spanColor(why);
    d.title = `${a.toFixed(1)}–${b.toFixed(1)} s: ${why}`;
    root.querySelector(".rp-spans").appendChild(d);
  }
  const spanAt = (t) => spans.find(([a, b]) => t >= a - 0.02 && t <= b + 0.02);

  const c = scene.plan.checks,
    md = c.miss_vs_demo_mm || c.ik_pos_err_mm;
  const row = (k, v) => `<span class="k">${k}</span><span>${v}</span>`;
  $("summary").innerHTML = [
    row("Miss vs your hand, typical", `${md.median.toFixed(1)} mm`),
    row("95% of the time under", `${md.p95.toFixed(1)} mm`),
    row("Worst", `${md.max.toFixed(1)} mm`),
    row(
      "Lowest hand point",
      `${c.hand_lowest_mm.toFixed(0)} mm (floor ${c.floor_mm})`,
    ),
    row(
      "Speed on the real arm",
      scene.plan.recommended_time_scale <= 1.05
        ? "full speed"
        : `${scene.plan.recommended_time_scale.toFixed(1)}× slower`,
    ),
    row("Length", `${dur.toFixed(1)} s`),
  ].join("");

  let k = 0,
    playing = false,
    last = performance.now(),
    tAcc = 0,
    mode = "overlay";
  const resize = () => {
    const r = stage.getBoundingClientRect();
    if (!r.width) return;
    renderer.setSize(r.width, r.height, false);
    freeCam.aspect = r.width / r.height;
    freeCam.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(stage);
  function show(kk) {
    k = Math.max(0, Math.min(N - 1, kk));
    for (let n = 0; n < nd; n++) {
      const o = dynObjs[n];
      if (o) setPose(o, poses.subarray((k * nd + n) * 7, (k * nd + n) * 7 + 7));
    }
    const rt = scene.robot_tcp[k],
      dt = scene.demo_tcp[k];
    robotDot.position.set(...rt);
    demoDot.position.set(...dt);
    demoDot.visible = scene.tracked[k];
    const miss = Math.hypot(rt[0] - dt[0], rt[1] - dt[1], rt[2] - dt[2]) * 1000;
    $("now-miss").textContent = scene.tracked[k]
      ? `${miss.toFixed(1)} mm`
      : "–";
    $("now-miss").style.color = !scene.tracked[k]
      ? ""
      : miss < 10
        ? "#43a047"
        : miss < 40
          ? "#f9a825"
          : "#e53935";
    $("now-grip").textContent = scene.grip[k].toFixed(0);
    $("now-tracked").textContent = scene.tracked[k] ? "yes" : "no (filled in)";
    $("slider").value = k;
    $("time").textContent = `${T[k].toFixed(2)} / ${dur.toFixed(2)} s`;
    const sp = spanAt(T[k]);
    $("banner").textContent = sp
      ? `Adjusted here: ${sp[2].toLowerCase()} (${sp[0].toFixed(1)}–${sp[1].toFixed(1)} s)`
      : "";
    if (mode === "overlay") {
      const f = pvAt(T[k] + scene.t_offset);
      if (f && bg.src !== f.url) bg.src = f.url;
    }
  }
  function applyAlpha() {
    const a = $("alpha").value / 100;
    for (const o of [...staticObjs, ...dynObjs])
      if (o) {
        o.material.opacity =
          (mode === "overlay" ? (o.userData.scenery ? 0 : a) : 1) *
          o.userData.baseOpacity;
        o.material.depthWrite = o.material.opacity > 0.98;
        o.visible = o.material.opacity > 0.001;
      }
  }
  function setMode(m) {
    mode = m;
    $("v-overlay").classList.toggle("rp-on", m === "overlay");
    $("v-3d").classList.toggle("rp-on", m === "3d");
    controls.enabled = m === "3d";
    bg.style.display = m === "overlay" && pv.length ? "" : "none";
    world.background = m === "3d" ? new THREE.Color(0x1d1f23) : null;
    applyAlpha();
    show(k);
  }
  function tick(now) {
    if (!alive) return;
    const dtw = (now - last) / 1000;
    last = now;
    if (playing) {
      tAcc += dtw * Number($("speed").value);
      let kk = k;
      while (kk < N - 1 && T[kk + 1] <= T[k] + tAcc) kk++;
      if (kk !== k) {
        tAcc -= T[kk] - T[k];
        show(kk);
      }
      if (k >= N - 1) {
        playing = false;
        $("play").textContent = "▶";
      }
    }
    demoLine.visible = robotLine.visible = $("paths").checked;
    if (mode === "3d") controls.update();
    renderer.render(world, mode === "3d" ? freeCam : overheadCam);
    raf = requestAnimationFrame(tick);
  }
  $("play").onclick = () => {
    if (k >= N - 1) show(0);
    playing = !playing;
    tAcc = 0;
    $("play").textContent = playing ? "❚❚" : "▶";
  };
  $("slider").oninput = () => {
    playing = false;
    $("play").textContent = "▶";
    show(+$("slider").value);
  };
  $("alpha").oninput = applyAlpha;
  $("v-overlay").onclick = () => setMode("overlay");
  $("v-3d").onclick = () => setMode("3d");
  const onKey = (e) => {
    if (
      !root.isConnected ||
      ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)
    )
      return;
    if (e.code === "Space") {
      e.preventDefault();
      $("play").click();
    }
    if (e.code === "ArrowRight") show(k + 1);
    if (e.code === "ArrowLeft") show(k - 1);
  };
  addEventListener("keydown", onKey);
  setMode(pv.length ? "overlay" : "3d");
  resize();
  show(0);
  root.querySelector(".rp-loading").remove();
  raf = requestAnimationFrame(tick);

  return {
    destroy() {
      alive = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
      removeEventListener("keydown", onKey);
      pv.forEach((p) => URL.revokeObjectURL(p.url));
      renderer.dispose();
      root.innerHTML = "";
    },
  };
}
