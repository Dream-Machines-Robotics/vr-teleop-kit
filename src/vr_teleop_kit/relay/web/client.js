(() => {
  const $ = (id) => document.getElementById(id);
  // Live tick rate of get_action() in the teleop process. Populated from
  // ik_state messages once the teleop publishes them; until then we fall
  // back to FALLBACK_LOOP_HZ for the joint-Δq-cap readout. 200 Hz matches
  // examples/teleop_bi_dk1.py's default; a lerobot-record loop ticks at the
  // dataset FPS, which the operator will see snap in shortly after starting.
  const FALLBACK_LOOP_HZ = 200;
  let measuredLoopHz = 0;
  const connPill = $("conn-pill"), calibPill = $("calib-pill");
  const wsStateEl = $("ws-state"), clutchEl = $("clutch-state");
  const sentEl = $("sent"), recvEl = $("recv"), rttEl = $("rtt");
  const log = $("log"), enterBtn = $("enter-vr");
  const calibBtn = $("calibrate"), clearCalibBtn = $("clear-calib");
  const calibStateEl = $("calib-state");
  const showAxesToggle = $("show-axes-toggle");

  // ---- Latency measurement mode (opt-in: ?latency=1, or ?latency=<hz>) ----
  // Continuously pings the relay — *including during a VR session* — and
  // aggregates the round-trip time, so you can compare transports (USB vs LAN)
  // under real teleop load. RTT is measured on this device's clock (no clock
  // sync needed); one-way transport ≈ RTT/2. Stats print to the debug rail and
  // are reported to the workstation terminal every 2 s for copy-paste. Off by
  // default — normal use keeps the 1 Hz idle ping below.
  const _latParam = new URL(location.href).searchParams.get("latency");
  const LAT_ON = _latParam !== null && _latParam !== "0" && _latParam !== "false";
  const LAT_HZ = Math.max(1, Math.min(50, parseFloat(_latParam) || 10));  // ?latency=20 → 20 Hz
  const LAT_WIN = 200;            // rolling sample window
  const _latBuf = [];            // recent RTT samples (ms)
  const _latPush = (rtt) => {
    if (!LAT_ON) return;
    _latBuf.push(rtt);
    if (_latBuf.length > LAT_WIN) _latBuf.shift();
  };
  const _latStats = () => {
    if (!_latBuf.length) return null;
    const s = _latBuf.slice().sort((a, b) => a - b);
    const pct = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
    return {
      n: s.length,
      mean: s.reduce((a, b) => a + b, 0) / s.length,
      p50: pct(0.5), p95: pct(0.95), min: s[0], max: s[s.length - 1],
    };
  };
  const _latFmt = (st) =>
    `latency[${location.host}] n=${st.n} mean=${st.mean.toFixed(1)} p50=${st.p50.toFixed(1)} `
    + `p95=${st.p95.toFixed(1)} min=${st.min.toFixed(1)} max=${st.max.toFixed(1)} ms `
    + `(one-way≈${(st.p50 / 2).toFixed(1)})`;

  const debugLink = $("debug-link"), debugClose = $("debug-close");
  const recalibLink = $("recalib-link"), recalibClose = $("recalib-close");
  const recalibModal = $("recalib-modal");
  const settingsLink = $("settings-link"), settingsClose = $("settings-close");
  const settingsModal = $("settings-modal"), resetSettingsBtn = $("reset-settings");
  const canvas = $("xr-canvas");
  const camOverlay = $("cam-overlay");

  // ---- Surface toggles ----
  // Debug rail is a session-persisting side panel (operators may keep it open
  // for the log). Calibration is a transient modal — open it, recalibrate,
  // close it. No persistence for the modal; debug-rail state is in
  // localStorage so it survives reloads during a debugging session.
  const DEBUG_OPEN_KEY = "vrteleop:debug_open_v1";
  const setDebugOpen = (open) => {
    document.body.classList.toggle("show-debug", open);
    debugLink.classList.toggle("active", open);
    try { localStorage.setItem(DEBUG_OPEN_KEY, open ? "1" : "0"); } catch (_) {}
  };
  const setRecalibOpen = (open) => {
    document.body.classList.toggle("show-recalib", open);
    recalibModal.setAttribute("aria-hidden", open ? "false" : "true");
  };
  const setSettingsOpen = (open) => {
    document.body.classList.toggle("show-settings", open);
    settingsModal.setAttribute("aria-hidden", open ? "false" : "true");
  };
  setDebugOpen(localStorage.getItem(DEBUG_OPEN_KEY) === "1");
  debugLink.addEventListener("click", () => setDebugOpen(!document.body.classList.contains("show-debug")));
  debugClose.addEventListener("click", () => setDebugOpen(false));
  recalibLink.addEventListener("click", () => setRecalibOpen(true));
  recalibClose.addEventListener("click", () => setRecalibOpen(false));
  settingsLink.addEventListener("click", () => setSettingsOpen(true));
  settingsClose.addEventListener("click", () => setSettingsOpen(false));
  // Close modals by clicking the backdrop (outside the card) or pressing Esc.
  recalibModal.addEventListener("click",  (e) => { if (e.target === recalibModal)  setRecalibOpen(false); });
  settingsModal.addEventListener("click", (e) => { if (e.target === settingsModal) setSettingsOpen(false); });
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    setRecalibOpen(false); setSettingsOpen(false);
  });

  // Aggregate UI state: the pill reflects whichever signal is worst.
  // wsState: "connecting" | "open" | "closed" | "error"
  // xrState: "checking" | "ready" | "in-session" | "calibrating" | "unsupported"
  const ui = { wsState: "connecting", xrState: "checking", engaged: false };
  const refreshPill = () => {
    let cls = "", text = "";
    if (ui.wsState === "closed" || ui.wsState === "error") {
      cls = "err"; text = ui.wsState === "error" ? "connection error" : "disconnected";
    } else if (ui.xrState === "unsupported") {
      cls = "warn"; text = "WebXR not supported";
    } else if (ui.xrState === "calibrating") {
      cls = "ok"; text = "calibrating — squeeze both grips";
    } else if (ui.xrState === "in-session") {
      cls = "ok"; text = ui.engaged ? "in VR · ENGAGED" : "in VR";
    } else if (ui.wsState === "open" && ui.xrState === "ready") {
      cls = "ok"; text = "ready";
    } else {
      cls = ""; text = "connecting…";
    }
    connPill.className = "pill" + (cls ? " " + cls : "");
    connPill.textContent = text;
    wsStateEl.textContent = ui.wsState;
  };
  refreshPill();

  // Per-hand controller-readout offset (3-vec in controller local frame).
  // WebXR gripSpace's origin is at the palm, but the operator's actual wrist
  // pivot is offset from the grip — typically backward and slightly off-axis.
  // Shifting the readout to coincide with the pivot makes pure wrist rotations
  // produce ~zero translation, so the IK doesn't see ghost translation on
  // wrist twists. Default value below is a reasonable hand-tuned starting
  // point until the user runs the in-VR pivot calibration ("Calibrate wrist
  // pivot" button). Calibration result lives in localStorage. Legacy URL
  // override: `?offset=0.07` → applies to Z on both hands.
  const STORAGE_KEY = "vrteleop:readout_offset_v1";
  const offsetParam = parseFloat(new URL(location.href).searchParams.get("offset"));
  const fallbackZ = Number.isFinite(offsetParam) ? offsetParam : 0.05;
  const readoutOffset = { left: [0, 0, fallbackZ], right: [0, 0, fallbackZ] };

  // Quaternion (xyzw) to 3x3 rotation matrix, row-major nested arrays.
  // Same convention as rotateVecByQuat below: R · v == rotateVecByQuat(v, q).
  const quatToMat3 = (q) => {
    const [x, y, z, w] = q;
    const xx = x*x, yy = y*y, zz = z*z;
    const xy = x*y, xz = x*z, yz = y*z;
    const wx = w*x, wy = w*y, wz = w*z;
    return [
      [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
      [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
      [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ];
  };

  // Solve M · x = v for 3-vector x via Cramer's rule. Returns {x, det}; x is
  // null if the system is too ill-conditioned to trust.
  const solve3x3 = (M, v) => {
    const [m00, m01, m02] = M[0];
    const [m10, m11, m12] = M[1];
    const [m20, m21, m22] = M[2];
    const det = m00*(m11*m22 - m12*m21)
              - m01*(m10*m22 - m12*m20)
              + m02*(m10*m21 - m11*m20);
    const tr = Math.abs(m00) + Math.abs(m11) + Math.abs(m22);
    if (!Number.isFinite(det) || Math.abs(det) < 1e-6 * Math.max(1, tr)**3) {
      return { x: null, det };
    }
    const dx = v[0]*(m11*m22 - m12*m21) - m01*(v[1]*m22 - m12*v[2]) + m02*(v[1]*m21 - m11*v[2]);
    const dy = m00*(v[1]*m22 - m12*v[2]) - v[0]*(m10*m22 - m12*m20) + m02*(m10*v[2] - v[1]*m20);
    const dz = m00*(m11*v[2] - v[1]*m21) - m01*(m10*v[2] - v[1]*m20) + v[0]*(m10*m21 - m11*m20);
    return { x: [dx/det, dy/det, dz/det], det };
  };

  // Pivot calibration: given samples of (p, R) where the wrist pivot was held
  // still, find the local-frame offset o so that p + R·o is constant. Solves
  //   o = -(Σ dRᵀ dR)⁻¹ (Σ dRᵀ dp)
  // with mean-centered dR, dp. Also returns RMS residual of the recovered
  // pivot path — small means the pivot was indeed held still.
  const solvePivot = (samples) => {
    const N = samples.length;
    if (N < 30) return { ok: false, reason: `too few samples (${N})` };
    const pbar = [0, 0, 0];
    const Rbar = [[0,0,0],[0,0,0],[0,0,0]];
    for (const s of samples) {
      for (let i = 0; i < 3; i++) {
        pbar[i] += s.p[i];
        for (let j = 0; j < 3; j++) Rbar[i][j] += s.R[i][j];
      }
    }
    for (let i = 0; i < 3; i++) {
      pbar[i] /= N;
      for (let j = 0; j < 3; j++) Rbar[i][j] /= N;
    }
    const A = [[0,0,0],[0,0,0],[0,0,0]];
    const b = [0, 0, 0];
    for (const s of samples) {
      const dR = [[0,0,0],[0,0,0],[0,0,0]];
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          dR[i][j] = s.R[i][j] - Rbar[i][j];
      const dp = [s.p[0]-pbar[0], s.p[1]-pbar[1], s.p[2]-pbar[2]];
      // A += dRᵀ · dR;  b += dRᵀ · dp
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          let aij = 0;
          for (let k = 0; k < 3; k++) aij += dR[k][i] * dR[k][j];
          A[i][j] += aij;
        }
        let bi = 0;
        for (let k = 0; k < 3; k++) bi += dR[k][i] * dp[k];
        b[i] += bi;
      }
    }
    const sol = solve3x3(A, [-b[0], -b[1], -b[2]]);
    if (!sol.x) return { ok: false, reason: `ill-conditioned (det=${sol.det.toExponential(2)})` };
    const o = sol.x;
    // Residual: variance of p + R·o around its mean.
    const pivots = samples.map(s => [
      s.p[0] + s.R[0][0]*o[0] + s.R[0][1]*o[1] + s.R[0][2]*o[2],
      s.p[1] + s.R[1][0]*o[0] + s.R[1][1]*o[1] + s.R[1][2]*o[2],
      s.p[2] + s.R[2][0]*o[0] + s.R[2][1]*o[1] + s.R[2][2]*o[2],
    ]);
    const cm = [0, 0, 0];
    for (const pv of pivots) for (let i = 0; i < 3; i++) cm[i] += pv[i] / N;
    let sumSq = 0;
    for (const pv of pivots) {
      const dx = pv[0]-cm[0], dy = pv[1]-cm[1], dz = pv[2]-cm[2];
      sumSq += dx*dx + dy*dy + dz*dz;
    }
    const rms = Math.sqrt(sumSq / N);
    return { ok: true, o, rms, n: N };
  };

  // Rotate vector v by quaternion q (xyzw form) — standard formula:
  //   v' = v + 2·qw·(q × v) + 2·(q × (q × v)),  q here = vector part.
  const rotateVecByQuat = (v, q) => {
    const [qx, qy, qz, qw] = q;
    const [vx, vy, vz] = v;
    const c1x = qy * vz - qz * vy;
    const c1y = qz * vx - qx * vz;
    const c1z = qx * vy - qy * vx;
    const c2x = qy * c1z - qz * c1y;
    const c2y = qz * c1x - qx * c1z;
    const c2z = qx * c1y - qy * c1x;
    return [
      vx + 2 * qw * c1x + 2 * c2x,
      vy + 2 * qw * c1y + 2 * c2y,
      vz + 2 * qw * c1z + 2 * c2z,
    ];
  };

  // Per-arm continuous-haptic intensity (0..1). Two sources, mixed per
  // tick into the actual controller pulse:
  //   haptic.*       — IK limit-pressure (singularity / joint-limit feel).
  //   forceHaptic.*  — gripper-torque feel (the orchestrator pushes this
  //                    via the teleop's `send_feedback`; teleop already
  //                    normalised to 0..1 via threshold + scaling).
  // The final pulse intensity is max(haptic, forceHaptic) so either
  // source can drive the buzz without one washing the other out.
  const haptic      = { left: 0.0, right: 0.0 };
  const forceHaptic = { left: 0.0, right: 0.0 };
  const HAPTIC_MIN_INTENSITY = 0.05;  // dead-zone: don't bother below this

  let sent = 0, recv = 0;
  const append = (msg) => {
    const line = document.createElement("div");
    line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
    log.prepend(line);
    while (log.childElementCount > 30) log.removeChild(log.lastChild);
  };

  const fmt3 = (v) => `(${v[0].toFixed(4)}, ${v[1].toFixed(4)}, ${v[2].toFixed(4)})`;
  const fmt3Short = (v) => `(${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)})`;
  const setCalibStatus = (ok, modalText, pillText) => {
    // Drive three surfaces in lockstep: the modal status line, the top-of-page
    // pill, and the launcher link label (which reads "Recalibrate" once the
    // operator has already calibrated, "Calibrate" otherwise).
    calibStateEl.textContent = modalText;
    calibStateEl.className = "calib-state" + (ok ? " ok" : "");
    calibPill.textContent = pillText;
    calibPill.className = "pill" + (ok ? " ok" : " warn");
    recalibLink.textContent = ok ? "Recalibrate wrist" : "Calibrate wrist";
  };

  // Pull any saved calibration. localStorage is per-origin/per-browser; clear
  // via the dev console with `localStorage.removeItem('vrteleop:readout_offset_v1')`.
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (stored && Array.isArray(stored.left) && Array.isArray(stored.right)) {
      readoutOffset.left = stored.left;
      readoutOffset.right = stored.right;
      append(`loaded calibration: L=${fmt3(stored.left)}  R=${fmt3(stored.right)}`);
      setCalibStatus(true,
        `calibrated · L=${fmt3Short(stored.left)} R=${fmt3Short(stored.right)}`,
        "wrist calibrated");
    } else {
      append(`no calibration stored; default ${fallbackZ.toFixed(3)}m on Z (legacy)`);
      setCalibStatus(false,
        `no calibration stored — using default ${(fallbackZ*100).toFixed(0)}cm Z offset`,
        "wrist not calibrated");
    }
  } catch (e) {
    append(`couldn't load calibration: ${e.message}`);
  }

  // ---- WebSocket ----
  const wsUrl = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
  append(`opening ${wsUrl}`);
  const ws = new WebSocket(wsUrl);

  ws.onopen    = () => {
    ui.wsState = "open"; refreshPill(); append("ws open");
    // Push the operator's persisted slider values to the teleop so it
    // matches the UI from tick 1 (instead of running on dataclass defaults
    // until the next slider drag).
    pushSettingsSnapshot();
  };
  ws.onclose   = (e) => { ui.wsState = "closed"; refreshPill(); append(`ws close ${e.code}`); };
  ws.onerror   = () => { ui.wsState = "error"; refreshPill(); append("ws error"); };

  ws.onmessage = (e) => {
    recv++;
    recvEl.textContent = recv;
    try {
      const data = JSON.parse(e.data);
      const t = data.echo && data.echo.t_client;
      if (typeof t === "number") {
        const rtt = performance.now() - t;
        rttEl.textContent = rtt.toFixed(1);
        _latPush(rtt);            // feed the latency-mode aggregator (no-op when off)
      }
      if (data.type === "ik_state") {
        const eng = !!data.engaged;
        if (eng !== ui.engaged) { ui.engaged = eng; refreshPill(); }
        clutchEl.textContent = eng ? "ENGAGED" : "idle";
        // Per-arm haptic intensity from the IK's limit-pressure metric.
        // Pulsed continuously in onXRFrame while the operator pushes
        // against a joint limit / unreachable orientation.
        if (typeof data.left_haptic === "number") haptic.left = data.left_haptic;
        if (typeof data.right_haptic === "number") haptic.right = data.right_haptic;
        if (typeof data.left_force_haptic === "number") forceHaptic.left = data.left_force_haptic;
        if (typeof data.right_force_haptic === "number") forceHaptic.right = data.right_force_haptic;
        // Measured teleop tick rate. Used by the joint-Δq-cap readout so
        // the slider shows honest rad/s for whatever loop is actually
        // driving get_action — examples/teleop_bi_dk1.py ticks at 200,
        // a lerobot-record loop at the dataset FPS.
        if (typeof data.loop_hz === "number" && data.loop_hz >= 1.0) {
          if (Math.abs(data.loop_hz - measuredLoopHz) > 0.5) {
            measuredLoopHz = data.loop_hz;
            // Refresh both rate-aware slider readouts so the new rate
            // shows immediately, not on the next user drag.
            for (const key of ["max_dq_per_joint_scalar_pos", "max_dq_per_joint_scalar_rot"]) {
              const p = params.find((p) => p.key === key);
              if (p) setSliderValue(p, parseFloat(p.input.value));
            }
          }
        }
      } else if (data.type === "request_settings") {
        // The teleop just (re)connected — send our current slider snapshot
        // so it doesn't run on stale dataclass defaults. Without this, any
        // changes made on the page *before* the teleop started would be
        // invisible to it.
        append("teleop requested settings snapshot — pushing");
        pushSettingsSnapshot();
      } else if (data.type === "camera_list") {
        cameras.onList(data.cameras || []);
      } else if (data.type === "webrtc_offer") {
        cameras.onOffer(data).catch((err) => append(`webrtc_offer failed: ${err.message}`));
      } else if (data.type === "haptic_calibrate_result") {
        onHapticCalibResult(data);
      }
    } catch {}
  };

  const wsSend = (obj) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
      sent++;
      sentEl.textContent = sent;
    }
  };

  // Heartbeat ping. Normal: 1 Hz, idle-only (keeps the RTT widget live on the
  // launcher). Latency mode (?latency=1): ping at LAT_HZ even during a VR
  // session so RTT is sampled under teleop load, with a rolling-stats report
  // to the debug rail + workstation terminal every 2 s.
  if (LAT_ON) {
    append(`latency mode: pinging ${LAT_HZ} Hz (incl. in VR), reporting every 2 s`);
    setInterval(() => wsSend({ type: "ping", t_client: performance.now() }), 1000 / LAT_HZ);
    setInterval(() => {
      const st = _latStats();
      if (!st) return;
      append(_latFmt(st));
      wsSend({ type: "latency_report", host: location.host, ...st });
    }, 2000);
  } else {
    setInterval(() => { if (!xrSession) wsSend({ type: "ping", t_client: performance.now() }); }, 1000);
  }

  // ---- Cameras (WebRTC) ----
  // Server publishes one VideoStreamTrack per cv2-opened camera and exchanges
  // SDP/ICE over the existing /ws. We map incoming tracks back to cameras
  // via the MediaStream id (the server stamps it with the camera id, e.g.
  // "top" / "left_wrist" / "right_wrist"). Per-camera toggle is a single WS
  // message; the server keeps the track in the SDP but pauses frame updates,
  // so the bandwidth drops to near zero without renegotiating.
  //
  // Master gate: streamingEnabled. Driven by the "Camera streaming" toggle
  // in the settings modal via cameras.setStreaming(v). Default is off — the
  // camera_list message still arrives so we know what's available, but no
  // webrtc_request goes out and the server's v4l2 devices stay closed.
  const CAMERAS_KEY = "vrteleop:cameras_v1";
  const cameras = (() => {
    let camList = [];              // [{id, label}, ...] from server
    let pc = null;                 // RTCPeerConnection
    const slots = {};              // camera id -> { wrap, video, toggle }
    let enabledState = {};         // camera id -> bool (persisted)
    let streamingEnabled = false;  // master toggle (settings modal)
    try { enabledState = JSON.parse(localStorage.getItem(CAMERAS_KEY) || "{}") || {}; } catch (_) {}

    // Hide the preview overlay until the master toggle turns streaming on.
    // Slots are built lazily inside _attach() so the v4l2 devices on the
    // server side are only opened when the operator actually asks for video.
    camOverlay.style.display = "none";

    const persist = () => {
      try { localStorage.setItem(CAMERAS_KEY, JSON.stringify(enabledState)); } catch (_) {}
    };

    const enabledFor = (id) => enabledState[id] !== false; // default on

    const buildSlot = (cam) => {
      const wrap = document.createElement("div");
      wrap.className = "cam-slot";
      wrap.dataset.id = cam.id;
      wrap.dataset.streaming = "false";
      wrap.dataset.enabled = enabledFor(cam.id) ? "true" : "false";
      const video = document.createElement("video");
      video.autoplay = true; video.playsInline = true; video.muted = true;
      wrap.appendChild(video);
      const meta = document.createElement("div");
      meta.className = "cam-meta";
      const label = document.createElement("span");
      label.className = "cam-label";
      label.textContent = cam.label || cam.id;
      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.checked = enabledFor(cam.id);
      toggle.title = "Stream this camera";
      toggle.addEventListener("change", () => {
        const v = !!toggle.checked;
        enabledState[cam.id] = v;
        persist();
        wrap.dataset.enabled = v ? "true" : "false";
        wsSend({ type: "camera_toggle", camera_id: cam.id, enabled: v });
      });
      meta.appendChild(label); meta.appendChild(toggle);
      wrap.appendChild(meta);
      camOverlay.appendChild(wrap);
      slots[cam.id] = { wrap, video, toggle };
    };

    const teardown = () => {
      if (pc) { try { pc.close(); } catch {} pc = null; }
      for (const cam of camList) {
        const s = slots[cam.id];
        if (s) { s.video.srcObject = null; s.wrap.dataset.streaming = "false"; }
      }
    };

    // Display order — top goes in the middle so left/right wrists flank
    // the main scene camera the way the operator's eyes flank the head.
    // Anything not in the explicit order falls to the end alphabetically.
    const DISPLAY_ORDER = ["left_wrist", "top", "right_wrist"];
    const orderRank = (id) => {
      const i = DISPLAY_ORDER.indexOf(id);
      return i < 0 ? DISPLAY_ORDER.length : i;
    };

    // Build the preview slots and negotiate WebRTC for the current camList.
    // Idempotent: tears down any existing peer connection first so toggling
    // off→on (or a server-side restart) can re-request cleanly.
    const _attach = () => {
      if (camList.length === 0) return;
      camOverlay.innerHTML = "";
      for (const k of Object.keys(slots)) delete slots[k];
      for (const cam of camList) buildSlot(cam);
      // Request the WebRTC offer using whichever cameras are currently
      // toggled on. Sending the explicit set saves the server the trouble
      // of starting tracks that will be muted on the first frame anyway.
      const enabled_cameras = camList.map(c => c.id).filter(enabledFor);
      teardown();
      wsSend({ type: "webrtc_request", enabled_cameras });
    };

    const onList = (list) => {
      camList = (list || []).slice().sort((a, b) => {
        const r = orderRank(a.id) - orderRank(b.id);
        return r !== 0 ? r : a.id.localeCompare(b.id);
      });
      if (camList.length === 0) {
        append("camera_list: no cameras advertised by server");
        return;
      }
      // Master toggle gates the actual handshake. When off, we just remember
      // what's available so flipping the toggle later can attach immediately.
      if (streamingEnabled) _attach();
    };

    // Master on/off from the settings modal. Idempotent. When turning off
    // mid-session: closes the PC, wipes the previews; the WebGL quad pass
    // already early-exits when slots[id] is undefined, so the in-VR view
    // empties on the next frame.
    const setStreaming = (v) => {
      v = !!v;
      if (v === streamingEnabled) return;
      streamingEnabled = v;
      if (v) {
        camOverlay.style.display = "";
        _attach();
      } else {
        teardown();
        camOverlay.style.display = "none";
        camOverlay.innerHTML = "";
        for (const k of Object.keys(slots)) delete slots[k];
      }
    };

    const onOffer = async (msg) => {
      // Server sets MediaStream.id = camera id, so the mapping is implicit
      // in the SDP — the incoming stream's id IS the camera id.

      // No STUN/TURN — LAN host candidates are enough for P2P media.
      // Cloudflared mode currently relies on the relay path (no TURN); for
      // off-LAN media we'd add a TURN server here. For now, P2P-on-LAN +
      // relay-for-signaling covers the primary use case.
      pc = new RTCPeerConnection({ iceServers: [] });
      pc.addEventListener("track", (e) => {
        const stream = e.streams[0];
        const camId = stream && stream.id;
        if (!camId || !slots[camId]) {
          append(`webrtc: orphan track sid=${stream && stream.id}`); return;
        }
        const slot = slots[camId];
        slot.video.srcObject = stream;
        slot.wrap.dataset.streaming = "true";
        // Re-apply enable state; the toggle's checkbox is the source of truth.
        slot.toggle.checked = enabledFor(camId);
        slot.wrap.dataset.enabled = enabledFor(camId) ? "true" : "false";
      });
      pc.addEventListener("icecandidate", (e) => {
        if (e.candidate) wsSend({ type: "ice_candidate", candidate: e.candidate });
      });
      pc.addEventListener("connectionstatechange", () => {
        append(`webrtc: ${pc.connectionState}`);
        if (pc.connectionState === "failed" || pc.connectionState === "closed") {
          for (const cam of camList) {
            const s = slots[cam.id]; if (s) s.wrap.dataset.streaming = "false";
          }
        }
      });

      await pc.setRemoteDescription({ type: msg.sdp_type, sdp: msg.sdp });
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      wsSend({ type: "webrtc_answer", sdp: pc.localDescription.sdp, sdp_type: pc.localDescription.type });
    };

    return {
      onList, onOffer, setStreaming,
      // Accessors for the in-VR WebGL renderer.
      list: () => camList,
      slot: (id) => slots[id],
      isEnabled: enabledFor,
    };
  })();

  // ---- Live-tunable settings (sliders → config_update WS message) ----
  // Sources of truth, in priority order:
  //   1. localStorage  (operator's last-tuned values, applied on load)
  //   2. data-default  on each .param row (matches the Python dataclass)
  // On every slider change we update the display, persist, and broadcast a
  // `config_update`; the teleop applies it on the next solve. We push the
  // initial values on WS open so a fresh process reflects whatever the
  // operator last tuned, rather than silently snapping back to the CLI
  // defaults.
  const SETTINGS_KEY = "vrteleop:settings_v1";
  // Slider params (continuous range inputs). Toggles are handled below.
  const params = Array.from(
    document.querySelectorAll(".param[data-key]:not(.param-toggle)")
  ).map((row) => {
    const key = row.dataset.key;
    const defaultVal = parseFloat(row.dataset.default);
    const input = row.querySelector("input[type=range]");
    const valEl = row.querySelector(".param-val");
    const infoBtn = row.querySelector(".info-btn");
    infoBtn.addEventListener("click", () => {
      const open = row.getAttribute("data-info-open") === "true";
      row.setAttribute("data-info-open", open ? "false" : "true");
      infoBtn.setAttribute("aria-expanded", open ? "false" : "true");
    });
    return { key, defaultVal, input, valEl, row };
  });
  // Toggle params (checkbox inputs). Same WS protocol — `config_update`
  // payload carries `{key: bool}`. The teleop's _LIVE_CONFIG_BOOLS handles
  // it the same way sliders are handled. data-scope="ui" rows are excluded
  // here and handled by uiToggles below — they affect only the browser
  // (camera streaming gate, debug visualisations, …) and shouldn't appear
  // in the snapshot sent to the teleop process.
  const wireInfoBtn = (row) => {
    const infoBtn = row.querySelector(".info-btn");
    if (!infoBtn) return;
    infoBtn.addEventListener("click", () => {
      const open = row.getAttribute("data-info-open") === "true";
      row.setAttribute("data-info-open", open ? "false" : "true");
      infoBtn.setAttribute("aria-expanded", open ? "false" : "true");
    });
  };
  const paramToggles = Array.from(
    document.querySelectorAll('.param-toggle[data-key]:not([data-scope="ui"])')
  ).map((row) => {
    const key = row.dataset.key;
    const defaultVal = row.dataset.default === "true";
    const input = row.querySelector("input[type=checkbox]");
    wireInfoBtn(row);
    return { key, defaultVal, input, row };
  });
  // Client-only toggles. Persisted in the same SETTINGS_KEY blob for
  // convenience but dispatched through UI_TOGGLE_HANDLERS instead of WS.
  const UI_TOGGLE_HANDLERS = {
    cameras_enabled: (v) => cameras.setStreaming(v),
  };
  const uiToggles = Array.from(
    document.querySelectorAll('.param-toggle[data-key][data-scope="ui"]')
  ).map((row) => {
    const key = row.dataset.key;
    const defaultVal = row.dataset.default === "true";
    const input = row.querySelector("input[type=checkbox]");
    wireInfoBtn(row);
    return { key, defaultVal, input, row };
  });
  const formatValue = (key, v) => {
    if (key === "pose_filter_alpha") return v.toFixed(2);
    if (key === "precision_factor") return `${v.toFixed(2)}×`;
    if (key.startsWith("force_haptic_threshold_nm")) return `${v.toFixed(2)} Nm`;
    // Joint Δq cap: dataclass is rad/tick but operators reason in rad/s.
    // Multiply by the measured teleop tick rate when available; if not
    // (no ik_state yet), fall back and tag the readout so the operator
    // knows the rate is assumed, not measured.
    if (key === "max_dq_per_joint_scalar_pos" || key === "max_dq_per_joint_scalar_rot") {
      if (measuredLoopHz >= 1.0) {
        return `${(v * measuredLoopHz).toFixed(1)} rad/s @ ${measuredLoopHz.toFixed(0)} Hz`;
      }
      return `${(v * FALLBACK_LOOP_HZ).toFixed(1)} rad/s (assumed)`;
    }
    return `${v.toFixed(1)}×`;
  };
  const setSliderValue = (p, v, { send = false } = {}) => {
    p.input.value = String(v);
    p.valEl.textContent = formatValue(p.key, v);
    // Mark non-default values so the operator sees at a glance which knobs
    // have been touched. (CSS could highlight; here we just keep the data.)
    p.row.dataset.modified = (v !== p.defaultVal) ? "1" : "0";
    if (send) wsSend({ type: "config_update", config: { [p.key]: v } });
  };
  // Hydrate from localStorage (fall back to default).
  let storedSettings = {};
  try { storedSettings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") || {}; } catch (_) {}
  for (const p of params) {
    const v = typeof storedSettings[p.key] === "number" ? storedSettings[p.key] : p.defaultVal;
    setSliderValue(p, v);
    p.input.addEventListener("input", () => {
      const v2 = parseFloat(p.input.value);
      setSliderValue(p, v2, { send: true });
      const next = { ...storedSettings, [p.key]: v2 };
      storedSettings = next;
      try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(next)); } catch (_) {}
    });
  }
  // Toggles that, beyond their config_update side, also flip a CSS hook on
  // <body> so dependent rows (`[data-haptic-only]`, etc.) can show/hide
  // without each one wiring its own listener. Add more entries here as
  // new dependent groups appear.
  const TOGGLE_BODY_CLASS = {
    force_haptic_enabled: "haptic-on",
  };
  const applyBodyToggleClass = (key, on) => {
    const cls = TOGGLE_BODY_CLASS[key];
    if (cls) document.body.classList.toggle(cls, !!on);
  };
  for (const t of paramToggles) {
    const v = typeof storedSettings[t.key] === "boolean" ? storedSettings[t.key] : t.defaultVal;
    t.input.checked = v;
    applyBodyToggleClass(t.key, v);
    t.input.addEventListener("change", () => {
      const v2 = t.input.checked;
      wsSend({ type: "config_update", config: { [t.key]: v2 } });
      applyBodyToggleClass(t.key, v2);
      const next = { ...storedSettings, [t.key]: v2 };
      storedSettings = next;
      try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(next)); } catch (_) {}
    });
  }
  for (const t of uiToggles) {
    const v = typeof storedSettings[t.key] === "boolean" ? storedSettings[t.key] : t.defaultVal;
    t.input.checked = v;
    const handler = UI_TOGGLE_HANDLERS[t.key];
    // Apply the persisted state once on load so reload preserves the choice.
    if (handler) handler(v);
    t.input.addEventListener("change", () => {
      const v2 = t.input.checked;
      if (handler) handler(v2);
      const next = { ...storedSettings, [t.key]: v2 };
      storedSettings = next;
      try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(next)); } catch (_) {}
    });
  }
  // Push the current snapshot to the teleop. Called on WS open and on Reset.
  const pushSettingsSnapshot = () => {
    const snap = {};
    for (const p of params) snap[p.key] = parseFloat(p.input.value);
    for (const t of paramToggles) snap[t.key] = !!t.input.checked;
    wsSend({ type: "config_update", config: snap });
  };
  resetSettingsBtn.addEventListener("click", () => {
    storedSettings = {};
    try { localStorage.removeItem(SETTINGS_KEY); } catch (_) {}
    for (const p of params) setSliderValue(p, p.defaultVal);
    for (const t of paramToggles) {
      t.input.checked = t.defaultVal;
      applyBodyToggleClass(t.key, t.defaultVal);
    }
    for (const t of uiToggles) {
      t.input.checked = t.defaultVal;
      const handler = UI_TOGGLE_HANDLERS[t.key];
      if (handler) handler(t.defaultVal);
    }
    pushSettingsSnapshot();
    // The calibrate button rides on the haptic toggle's checked state;
    // setting `.checked` programmatically doesn't fire `change`, so
    // refresh its disabled bit by hand.
    refreshHapticCalibBtn();
    append("settings reset to defaults");
  });

  // ---- Gripper-haptic threshold calibration ----
  // Operator presses "Calibrate" (inline with the force-haptic checkbox)
  // while the arms are still and grippers are empty. We send
  // `haptic_calibrate` to the teleop; it samples per-arm peak gripper
  // torque for `duration_s` seconds, writes new thresholds, and echoes a
  // `haptic_calibrate_result` we display. The button doubles as a status
  // surface ("Calibrating 4.2s") since we dropped the dedicated state row.
  const hapticCalibBtn = $("haptic-calibrate-btn");
  const hapticCalibLabelDefault = hapticCalibBtn ? hapticCalibBtn.textContent : "Calibrate";
  const hapticToggle = $("param-force-haptic-enabled");
  const HAPTIC_CALIB_DURATION_S = 5.0;
  let hapticCalibRunning = false;
  let hapticCalibTimer = null;
  // Enable the button iff the force-haptic toggle is on AND no calibration
  // is currently running. Called from the toggle change handler and on
  // initial hydrate (paramToggles loop above).
  const refreshHapticCalibBtn = () => {
    if (!hapticCalibBtn) return;
    hapticCalibBtn.disabled = hapticCalibRunning || !(hapticToggle && hapticToggle.checked);
  };
  // Register the calibrate button's enable/disable as a side-effect of
  // the haptic toggle, alongside the existing body-class side-effect.
  // Done by extending the TOGGLE_BODY_CLASS-driven path with a parallel
  // hook so the existing wiring keeps working untouched.
  if (hapticToggle) {
    hapticToggle.addEventListener("change", refreshHapticCalibBtn);
    refreshHapticCalibBtn();
  }
  if (hapticCalibBtn) {
    hapticCalibBtn.addEventListener("click", () => {
      if (hapticCalibRunning) return;
      if (ws.readyState !== WebSocket.OPEN) {
        hapticCalibBtn.textContent = "no WS";
        setTimeout(() => { hapticCalibBtn.textContent = hapticCalibLabelDefault; }, 1500);
        return;
      }
      hapticCalibRunning = true;
      refreshHapticCalibBtn();
      const t0 = performance.now();
      const tick = () => {
        const elapsed = (performance.now() - t0) / 1000;
        const remaining = Math.max(0, HAPTIC_CALIB_DURATION_S - elapsed);
        hapticCalibBtn.textContent = `Calibrating ${remaining.toFixed(1)}s`;
        if (remaining > 0) {
          hapticCalibTimer = setTimeout(tick, 100);
        }
      };
      tick();
      wsSend({ type: "haptic_calibrate", duration_s: HAPTIC_CALIB_DURATION_S });
      // The teleop will respond with `haptic_calibrate_result` via the WS
      // handler below. Belt-and-suspenders timeout in case the message is
      // dropped — restores the UI so the operator can retry.
      setTimeout(() => {
        if (!hapticCalibRunning) return;
        onHapticCalibResult(null);
      }, (HAPTIC_CALIB_DURATION_S + 2.0) * 1000);
    });
  }
  function onHapticCalibResult(data) {
    if (hapticCalibTimer) { clearTimeout(hapticCalibTimer); hapticCalibTimer = null; }
    hapticCalibRunning = false;
    if (hapticCalibBtn) {
      hapticCalibBtn.textContent = hapticCalibLabelDefault;
    }
    refreshHapticCalibBtn();
    if (!data) {
      append("haptic calib: timed out — is the teleop running?");
      return;
    }
    const fmt = (v) => (typeof v === "number" ? v.toFixed(3) : "?");
    let line =
      `haptic calib: peak L=${fmt(data.left_peak_nm)} R=${fmt(data.right_peak_nm)}` +
      ` → threshold L=${fmt(data.left_threshold_nm)} R=${fmt(data.right_threshold_nm)}`;
    if (typeof data.max_nm === "number") {
      line += `  (max ${fmt(data.max_nm)})`;
    }
    append(line);
    if (Array.isArray(data.suspicious) && data.suspicious.length > 0) {
      append(`haptic calib WARN — high idle torque: ${data.suspicious.join(", ")}. Hardware OK?`);
    }
    // Reflect the freshly-written thresholds in the two sliders so the
    // operator sees what was set and can nudge by hand. send:false because
    // the teleop already has these values — we don't want to round-trip.
    const updates = {
      force_haptic_threshold_nm_left: data.left_threshold_nm,
      force_haptic_threshold_nm_right: data.right_threshold_nm,
    };
    let changed = false;
    for (const [key, v] of Object.entries(updates)) {
      if (typeof v !== "number") continue;
      const p = params.find((p) => p.key === key);
      if (!p) continue;
      setSliderValue(p, v);
      storedSettings[key] = v;
      changed = true;
    }
    if (changed) {
      try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(storedSettings)); } catch (_) {}
    }
  }

  // ---- WebXR ----
  let xrSession = null, xrRefSpace = null, gl = null;
  let xrMode = null;  // "immersive-ar" (passthrough) or "immersive-vr"

  // Calibration state. The calibrate button starts its own dedicated XR
  // session (calibMode=true) — capture is triggered IN-VR by squeezing both
  // grips, since Quest browser empties inputSources whenever the user
  // interacts with a 2D HTML panel. `calibrating` gates per-frame sample push
  // (and `calibMode` gates suppression of normal xr_frame WS streaming).
  let calibMode = false;
  let calibrating = false;
  let calibStartTime = 0;
  const calibBuffers = { left: [], right: [] };
  let calibFrameCount = 0;
  let calibHandednessSeen = new Set();
  const CALIB_GRIP_THRESHOLD = 0.7;  // both grip values >= this triggers capture

  if (!navigator.xr) {
    ui.xrState = "unsupported"; refreshPill();
    enterBtn.textContent = "WebXR unavailable";
    append("WebXR API missing");
  } else {
    (async () => {
      // Prefer AR (passthrough) so the operator can see the real robot.
      for (const m of ["immersive-ar", "immersive-vr"]) {
        try { if (await navigator.xr.isSessionSupported(m)) { xrMode = m; break; } } catch {}
      }
      if (xrMode) {
        ui.xrState = "ready"; refreshPill();
        enterBtn.disabled = false;
        if (calibBtn) calibBtn.disabled = false;
      } else {
        ui.xrState = "unsupported"; refreshPill();
      }
    })();
  }

  enterBtn.addEventListener("click", async () => {
    if (xrSession) { await xrSession.end(); return; }
    try {
      gl = canvas.getContext("webgl2", { xrCompatible: true });
      if (!gl) throw new Error("WebGL2 unavailable");
      xrSession = await navigator.xr.requestSession(xrMode, { optionalFeatures: ["local-floor"] });
      xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
      xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
        .catch(() => xrSession.requestReferenceSpace("local"));
      xrSession.addEventListener("end", () => {
        xrSession = null; xrRefSpace = null;
        videoGL = null; cameraAnchor = null;
        ui.xrState = xrMode ? "ready" : "unsupported"; ui.engaged = false; refreshPill();
        enterBtn.textContent = "Start Teleop";
        if (calibBtn) calibBtn.disabled = !xrMode;
        append("xr session ended");
      });
      ui.xrState = "in-session"; refreshPill();
      enterBtn.textContent = "Stop Teleop";
      append("xr session started");
      xrSession.requestAnimationFrame(onXRFrame);
    } catch (err) {
      append(`xr error: ${err.message}`);
      console.error(err);
    }
  });

  // ---- Wrist-pivot calibration ----
  // Capture ~5s of raw controller poses while the operator freely rotates
  // their wrists (yaw + pitch + roll, mixed) without moving the arms, then
  // solve the linear pivot-calibration problem per hand and persist the
  // offsets. The trigger lives IN-VR (squeeze both grips) because Quest
  // browser empties inputSources while the user is pointing at a 2D HTML
  // panel — clicking the page button to start a recording, while in
  // immersive XR, doesn't work reliably on Meta browser today.
  const CALIB_DURATION_S   = 5.0;
  const CALIB_RESIDUAL_MAX = 0.015;   // 15mm RMS — anything larger probably means the arm moved
  const CALIB_OFFSET_MAX   = 0.20;    // sanity cap; real wrist offset is a few cm

  // Vibrate every input source attached to a session. Used as start/end
  // signal during calibration since the operator can't see the on-page log.
  const pulseAllControllers = (session, intensity, durationMs) => {
    if (!session) return;
    for (const src of session.inputSources) {
      const ha = src.gamepad && src.gamepad.hapticActuators;
      if (ha && ha[0]) {
        try { ha[0].pulse(intensity, durationMs); } catch (_) {}
      }
    }
  };

  // Run the pivot solve over current calibBuffers, log per-hand verdicts, and
  // persist the result to localStorage iff both hands passed all checks.
  const processCalibrationResults = () => {
    const handStr = [...calibHandednessSeen].sort().join(",") || "(none)";
    append(`capture done: L=${calibBuffers.left.length}  R=${calibBuffers.right.length}  frames=${calibFrameCount}  handedness seen={${handStr}}`);

    const results = {};
    const verdicts = {};
    for (const hand of ["left", "right"]) {
      const r = solvePivot(calibBuffers[hand]);
      if (!r.ok) {
        append(`${hand}: FAILED — ${r.reason}`);
        verdicts[hand] = false;
        continue;
      }
      if (r.rms > CALIB_RESIDUAL_MAX) {
        append(`${hand}: residual ${(r.rms*1000).toFixed(1)}mm RMS too high (>${(CALIB_RESIDUAL_MAX*1000).toFixed(0)}mm) — arm probably moved.`);
        verdicts[hand] = false;
        continue;
      }
      if (Math.abs(r.o[0]) > CALIB_OFFSET_MAX || Math.abs(r.o[1]) > CALIB_OFFSET_MAX || Math.abs(r.o[2]) > CALIB_OFFSET_MAX) {
        append(`${hand}: offset ${fmt3(r.o)} unreasonably large — retry with more rotation variety.`);
        verdicts[hand] = false;
        continue;
      }
      results[hand] = r;
      verdicts[hand] = true;
      append(`${hand}: o=${fmt3(r.o)}  residual=${(r.rms*1000).toFixed(1)}mm RMS  (n=${r.n})`);
    }
    if (verdicts.left && verdicts.right) {
      readoutOffset.left = results.left.o;
      readoutOffset.right = results.right.o;
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
          left: readoutOffset.left, right: readoutOffset.right,
        }));
        append("calibration applied and saved.");
      } catch (e) {
        append(`applied (couldn't persist: ${e.message})`);
      }
      setCalibStatus(true,
        `calibrated · L=${fmt3Short(readoutOffset.left)} R=${fmt3Short(readoutOffset.right)}`,
        "wrist calibrated");
    } else {
      append("calibration NOT applied — retry.");
      setCalibStatus(false,
        "calibration failed — try again with more wrist rotation",
        "wrist not calibrated");
    }
  };

  if (clearCalibBtn) {
    clearCalibBtn.addEventListener("click", () => {
      try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
      readoutOffset.left  = [0, 0, fallbackZ];
      readoutOffset.right = [0, 0, fallbackZ];
      append(`calibration cleared. Reverted to default offset ${fallbackZ.toFixed(3)}m on Z (legacy).`);
      setCalibStatus(false,
        `no calibration stored — using default ${(fallbackZ*100).toFixed(0)}cm Z offset`,
        "wrist not calibrated");
    });
  }

  if (calibBtn) {
    calibBtn.addEventListener("click", async () => {
      if (xrSession) {
        append("calibrate: please exit VR first, then click Calibrate.");
        return;
      }
      if (!xrMode) {
        append("calibrate: WebXR not supported on this browser.");
        return;
      }
      calibBtn.disabled = true;
      enterBtn.disabled = true;
      try {
        gl = canvas.getContext("webgl2", { xrCompatible: true });
        if (!gl) throw new Error("WebGL2 unavailable");
        xrSession = await navigator.xr.requestSession(xrMode, { optionalFeatures: ["local-floor"] });
        xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
        xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
          .catch(() => xrSession.requestReferenceSpace("local"));

        calibMode = true;
        calibrating = false;
        calibBuffers.left = [];
        calibBuffers.right = [];
        calibFrameCount = 0;
        calibHandednessSeen = new Set();

        xrSession.addEventListener("end", () => {
          const wasCalibMode = calibMode;
          calibMode = false;
          calibrating = false;
          xrSession = null; xrRefSpace = null;
          videoGL = null; cameraAnchor = null;
          ui.xrState = xrMode ? "ready" : "unsupported"; refreshPill();
          enterBtn.textContent = "Start Teleop";
          enterBtn.disabled = !xrMode;
          calibBtn.disabled = !xrMode;
          append("xr session ended");
          if (wasCalibMode) processCalibrationResults();
        });
        ui.xrState = "calibrating"; refreshPill();
        enterBtn.textContent = "Stop Teleop";
        enterBtn.disabled = false;   // allow operator to bail out via this button
        append("calibration session started — squeeze BOTH grips simultaneously to begin a 5s capture.");
        xrSession.requestAnimationFrame(onXRFrame);
      } catch (err) {
        append(`calibrate: xr error: ${err.message}`);
        calibMode = false;
        xrSession = null;
        calibBtn.disabled = !xrMode;
        enterBtn.disabled = !xrMode;
      }
    });
  }

  const xyz = (v) => [v.x, v.y, v.z];
  const xyzw = (v) => [v.x, v.y, v.z, v.w];

  // ---- Debug overlay toggle ----
  // Gates the per-controller readout-point balls drawn in onXRFrame (see
  // "Debug balls" below). Toggleable in the Debug section; persisted in
  // localStorage. URL param `?axes=1` is a one-shot override (handy for
  // shared links) — the "axes" naming is legacy from the old axis-triad
  // overlay, kept so stored prefs and shared links keep working. Default
  // OFF — operators shouldn't see the overlay during normal teleop.
  const AXES_KEY = "vrteleop:show_axes_v1";
  const axesParam = new URL(location.href).searchParams.get("axes");
  let showAxes = axesParam != null
    ? axesParam !== "0"
    : (localStorage.getItem(AXES_KEY) === "1");
  showAxesToggle.checked = showAxes;
  showAxesToggle.addEventListener("change", () => {
    showAxes = showAxesToggle.checked;
    try { localStorage.setItem(AXES_KEY, showAxes ? "1" : "0"); } catch (_) {}
  });

  // Multiply two 4×4 column-major matrices (Float32Array(16)).
  const mat4Mul = (a, b) => {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[r + k*4] * b[k + c*4];
        o[r + c*4] = s;
      }
    return o;
  };

  // 4×4 column-major from position [x,y,z] and quaternion [x,y,z,w].
  const mat4FromPosQuat = (p, q) => {
    const [x, y, z, w] = q;
    return new Float32Array([
      1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y),     0,
      2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x),     0,
      2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y), 0,
      p[0],              p[1],              p[2],              1,
    ]);
  };
  // ---- Debug balls (controller readout-point markers) ----
  // The "show axes" debug overlay draws two balls per controller instead of
  // coordinate frames: RED at the raw WebXR gripSpace origin (the palm, where
  // the Quest reports the controller) and BLUE at the shifted wrist-pivot
  // point we actually stream (gripSpace origin + readoutOffset). Only the
  // position of the readout point matters here, so balls read clearer than
  // axis triads.
  //
  // Rendered as sphere-impostors: a gl.POINTS sprite shaded per-pixel as a lit
  // 3D ball (round mask + hemisphere normal + Lambert), with screen size that
  // shrinks with distance. Looks like a sphere but is robust in stereo VR —
  // no mesh winding / depth-order pitfalls, always faces the camera, always
  // draws on top (the debug pass disables depth test).
  let ballsGL = null;
  const _initBallsGL = () => {
    const vs = `attribute vec3 aPos; uniform mat4 uMvp; uniform float uSize;
                void main(){
                  vec4 clip = uMvp * vec4(aPos, 1.0);
                  gl_Position = clip;
                  gl_PointSize = clamp(uSize / max(clip.w, 0.05), 4.0, 64.0);
                }`;
    const fs = `precision mediump float; uniform vec3 uCol;
                void main(){
                  vec2 c = gl_PointCoord * 2.0 - 1.0;     // -1..1 across the sprite
                  float r2 = dot(c, c);
                  if (r2 > 1.0) discard;                  // round silhouette
                  vec3 n = vec3(c.x, -c.y, sqrt(1.0 - r2)); // hemisphere normal
                  float lit = 0.40 + 0.60 * max(0.0, dot(n, normalize(vec3(0.3, 0.5, 0.8))));
                  gl_FragColor = vec4(uCol * lit, 1.0);
                }`;
    const compile = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 0]), gl.STATIC_DRAW);
    ballsGL = {
      prog, vbo,
      aPos: gl.getAttribLocation(prog, "aPos"),
      uMvp: gl.getUniformLocation(prog, "uMvp"),
      uCol: gl.getUniformLocation(prog, "uCol"),
      uSize: gl.getUniformLocation(prog, "uSize"),
    };
  };

  // Draw one shaded ball at the origin of `modelMat` (orientation ignored).
  // `sizeAt1m` is the on-screen diameter in pixels at 1 m; the shader shrinks
  // it with distance so the ball holds a consistent real-world size.
  const _drawBallAt = (VP, modelMat, rgb, sizeAt1m = 11.0) => {
    const { prog, vbo, aPos, uMvp, uCol, uSize } = ballsGL;
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uMvp, false, mat4Mul(VP, modelMat));
    gl.uniform3fv(uCol, rgb);
    gl.uniform1f(uSize, sizeAt1m);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, 1);
  };

  // ---- WebGL video quads (in-VR camera display) ----
  // dom-overlay isn't supported in immersive-vr and Quest's AR support for
  // it is unreliable, so we render the camera tracks as textured quads in
  // the WebXR scene. The anchor is **world-locked**: we capture the head's
  // pose on the first frame after session start and re-use it forever,
  // so the operator can turn their head to inspect a specific camera up
  // close instead of the videos chasing their gaze. To re-anchor after
  // physically repositioning, exit VR and re-enter.
  //
  // Layout (head-local at anchor time):
  //     ┌─────── top ───────┐
  //     │                   │
  //     └───────────────────┘
  //   ┌── left ──┐ ┌── right ──┐
  //   │          │ │           │
  //   └──────────┘ └───────────┘
  let videoGL = null;
  let cameraAnchor = null;  // 4x4 world matrix captured on first XR frame

  // Build a yaw-only rotation+translation matrix from the head pose. The
  // head's forward vector is projected onto the world XZ plane so any
  // pitch/roll at anchor time (e.g. the operator looking slightly down)
  // doesn't tilt the entire video bank. Position is captured as-is.
  const _yawOnlyAnchor = (position, qx, qy, qz, qw) => {
    // Rotate local forward (0, 0, -1) by the head quaternion.
    const fx = -2 * (qx * qz + qy * qw);
    const fz = -(1 - 2 * (qx * qx + qy * qy));
    const len = Math.hypot(fx, fz);
    if (len < 1e-4) {
      // Operator looking nearly straight up/down at anchor — degenerate
      // projection, fall back to identity yaw so the videos at least face
      // a consistent direction.
      return mat4FromPosQuat(position, [0, 0, 0, 1]);
    }
    const fxN = fx / len, fzN = fz / len;
    // Column-major 4x4: X axis = (-fzN, 0, fxN), Y axis = world up,
    // Z axis = anchor backward = -forward.
    return new Float32Array([
      -fzN, 0, fxN, 0,
      0,    1, 0,   0,
      -fxN, 0, -fzN, 0,
      position[0], position[1], position[2], 1,
    ]);
  };
  const _initVideoGL = () => {
    const vs = `attribute vec2 aPos; attribute vec2 aTex;
                uniform mat4 uMvp; varying vec2 vTex;
                void main(){
                  gl_Position = uMvp * vec4(aPos, 0.0, 1.0);
                  vTex = aTex;
                }`;
    const fs = `precision mediump float; varying vec2 vTex;
                uniform sampler2D uTex;
                void main(){ gl_FragColor = texture2D(uTex, vTex); }`;
    const compile = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    // Unit quad in XY plane (z=0): 4 vertices × (x, y, u, v).
    // Combined with UNPACK_FLIP_Y_WEBGL=true (set in _uploadVideoTexture),
    // top-of-image ends up at texture coord t=1, so top-of-quad samples
    // top-of-image and the video displays upright in headset.
    const verts = new Float32Array([
      -0.5, -0.5, 0, 0,
       0.5, -0.5, 1, 0,
       0.5,  0.5, 1, 1,
      -0.5,  0.5, 0, 1,
    ]);
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    videoGL = {
      prog, vbo, ibo,
      aPos: gl.getAttribLocation(prog, "aPos"),
      aTex: gl.getAttribLocation(prog, "aTex"),
      uMvp: gl.getUniformLocation(prog, "uMvp"),
      uTex: gl.getUniformLocation(prog, "uTex"),
      textures: {},   // camId -> WebGLTexture
    };
  };

  const _uploadVideoTexture = (camId, video) => {
    if (!videoGL) return false;
    if (video.readyState < 2 || !video.videoWidth) return false;
    let tex = videoGL.textures[camId];
    if (!tex) {
      tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      videoGL.textures[camId] = tex;
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    } catch (e) { return false; }
    return true;
  };

  const _drawVideoQuad = (VP, modelMat, camId) => {
    const tex = videoGL.textures[camId];
    if (!tex) return;
    const { prog, vbo, ibo, aPos, aTex, uMvp, uTex } = videoGL;
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uMvp, false, mat4Mul(VP, modelMat));
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(aTex);
    gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 16, 8);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(uTex, 0);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  };

  // Draw enabled cameras in a world-locked T-layout: main camera centered
  // on top, left/right wrists side-by-side below. Anchored on first frame
  // and reused, so head turns let the operator look at any camera up close
  // instead of the videos chasing their gaze.
  const _renderCameraQuads = (vpose, layer) => {
    if (!videoGL) _initVideoGL();
    if (cameraAnchor === null) {
      const q = xyzw(vpose.transform.orientation);
      cameraAnchor = _yawOnlyAnchor(
        [vpose.transform.position.x, vpose.transform.position.y, vpose.transform.position.z],
        q[0], q[1], q[2], q[3]
      );
    }

    // Upload each enabled video texture once per frame.
    const ready = {};
    for (const cam of cameras.list()) {
      const slot = cameras.slot(cam.id);
      if (!slot || !cameras.isEnabled(cam.id)) continue;
      if (!slot.video.srcObject) continue;
      if (_uploadVideoTexture(cam.id, slot.video)) ready[cam.id] = cam;
    }
    if (Object.keys(ready).length === 0) return;

    // Layout (anchor-local space — Y up, Z forward = -Z). Main camera up
    // top with the wrist cams paired directly below. The vertical centers
    // are chosen so the gap between the rows lands exactly at Y=0 (the
    // operator's eye-level horizon at anchor time): main entirely above,
    // wrists entirely below.
    const QUAD_W = 0.65, QUAD_H = 0.49;
    const H_GAP = 0.05;  // between left and right wrist
    const V_GAP = 0.05;  // between main and wrist row
    const Z = -1.0;
    const Y_MAIN  = +(QUAD_H + V_GAP) / 2;   // main row center, above horizon
    const Y_WRIST = -(QUAD_H + V_GAP) / 2;   // wrist row center, below horizon

    // (x, y) positions per camera id. Missing cameras are skipped.
    const placements = {
      top:         { x: 0,                              y: Y_MAIN  },
      left_wrist:  { x: -(QUAD_W / 2 + H_GAP / 2),      y: Y_WRIST },
      right_wrist: { x: +(QUAD_W / 2 + H_GAP / 2),      y: Y_WRIST },
    };

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    for (const view of vpose.views) {
      const vpRect = layer.getViewport(view);
      gl.viewport(vpRect.x, vpRect.y, vpRect.width, vpRect.height);
      const VP = mat4Mul(view.projectionMatrix, view.transform.inverse.matrix);
      for (const cam of Object.values(ready)) {
        const pos = placements[cam.id];
        if (!pos) continue;  // unknown camera id, no slot in layout
        const modelLocal = new Float32Array([
          QUAD_W, 0, 0, 0,
          0, QUAD_H, 0, 0,
          0, 0, 1, 0,
          pos.x, pos.y, Z, 1,
        ]);
        const modelWorld = mat4Mul(cameraAnchor, modelLocal);
        _drawVideoQuad(VP, modelWorld, cam.id);
      }
    }
    gl.disable(gl.BLEND);
  };

  function onXRFrame(_time, frame) {
    if (!xrSession) return;
    xrSession.requestAnimationFrame(onXRFrame);

    // Clear the framebuffer so the headset doesn't show flicker / undefined
    // image. In AR (passthrough) mode the alpha must be 0 so the camera feed
    // shows through; in VR mode use opaque dark grey.
    const layer = xrSession.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    if (xrMode === "immersive-ar") {
      gl.clearColor(0.0, 0.0, 0.0, 0.0);
    } else {
      gl.clearColor(0.08, 0.08, 0.10, 1.0);
    }
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Debug overlay — runs regardless of clutch/calib mode. Two balls per
    // controller: RED at the raw gripSpace origin (where the Quest reads the
    // controller), BLUE at the shifted wrist-pivot point we actually stream.
    if (showAxes) {
      if (!ballsGL) _initBallsGL();
      gl.disable(gl.DEPTH_TEST);
      const vpose = frame.getViewerPose(xrRefSpace);
      if (vpose) {
        for (const view of vpose.views) {
          const vpRect = layer.getViewport(view);
          gl.viewport(vpRect.x, vpRect.y, vpRect.width, vpRect.height);
          const VP = mat4Mul(view.projectionMatrix, view.transform.inverse.matrix);
          for (const src of xrSession.inputSources) {
            if (src.handedness !== "left" && src.handedness !== "right") continue;
            const space = src.gripSpace || src.targetRaySpace;
            const pose = frame.getPose(space, xrRefSpace);
            if (!pose) continue;
            // RED: original readout (raw gripSpace origin = palm).
            _drawBallAt(VP, pose.transform.matrix, [1.0, 0.15, 0.15]);
            // BLUE: shifted readout (gripSpace origin + readoutOffset = wrist pivot).
            const q = xyzw(pose.transform.orientation);
            const off = readoutOffset[src.handedness] || [0, 0, 0];
            const offW = rotateVecByQuat(off, q);
            const wristPos = [
              pose.transform.position.x + offW[0],
              pose.transform.position.y + offW[1],
              pose.transform.position.z + offW[2],
            ];
            _drawBallAt(VP, mat4FromPosQuat(wristPos, q), [0.2, 0.45, 1.0]);
          }
        }
      }
    }

    // Camera quad pass — world-locked T-layout video quads (anchored on the
    // first frame; see _renderCameraQuads).
    {
      const vpose = frame.getViewerPose(xrRefSpace);
      if (vpose) _renderCameraQuads(vpose, layer);
    }

    if (calibMode || calibrating) calibFrameCount++;

    const controllers = {};
    for (const src of xrSession.inputSources) {
      if (calibMode || calibrating) calibHandednessSeen.add(src.handedness || "(unset)");
      if (src.handedness !== "left" && src.handedness !== "right") continue;
      const space = src.gripSpace || src.targetRaySpace;
      const pose = frame.getPose(space, xrRefSpace);
      if (!pose) continue;
      const t = pose.transform;
      const gp = src.gamepad;
      const orient = xyzw(t.orientation);
      const rawPos = xyz(t.position);
      // Shift the reported readout point in the controller's local frame so
      // it lines up with the operator's wrist pivot — pure wrist twists then
      // produce ~zero translation delta. Per-hand offset is set by the
      // calibration flow (or the legacy ?offset URL param as a fallback).
      const off = readoutOffset[src.handedness] || [0, 0, 0];
      const offsetWorld = rotateVecByQuat(off, orient);
      const posOut = [
        rawPos[0] + offsetWorld[0],
        rawPos[1] + offsetWorld[1],
        rawPos[2] + offsetWorld[2],
      ];
      // Capture raw (un-shifted) pose samples for the pivot solver.
      if (calibrating && calibBuffers[src.handedness]) {
        calibBuffers[src.handedness].push({ p: rawPos, R: quatToMat3(orient) });
      }
      controllers[src.handedness] = {
        position: posOut,
        orientation: orient,
        buttons: gp ? gp.buttons.map((b) => ({ p: b.pressed, t: b.touched, v: b.value })) : [],
        axes:    gp ? Array.from(gp.axes) : [],
      };
    }

    // Per-frame controller haptic. Mix the two streams (IK limit pressure
    // + gripper-torque force feedback) via max(): either signal can drive
    // the buzz on its own. Pulses are short (30 ms) and re-issued each
    // frame so they overlap into continuous vibration. Suppressed during
    // calibMode so we don't fight the calibration start/end pulses.
    if (!calibMode) {
      for (const src of xrSession.inputSources) {
        if (src.handedness !== "left" && src.handedness !== "right") continue;
        const ikIntensity = haptic[src.handedness] || 0;
        const fIntensity  = forceHaptic[src.handedness] || 0;
        const intensity   = Math.max(ikIntensity, fIntensity);
        if (intensity > HAPTIC_MIN_INTENSITY) {
          const ha = src.gamepad && src.gamepad.hapticActuators;
          if (ha && ha[0]) {
            try { ha[0].pulse(intensity, 30); } catch (_) {}
          }
        }
      }
    }

    // ---- Calibration trigger / capture lifecycle (in-VR) ----
    if (calibMode) {
      // Quest controller buttons[1] is "grip" (analog 0..1 .value).
      const gripVal = (c) => (c && c.buttons && c.buttons[1]) ? (c.buttons[1].v || 0) : 0;
      const bothPressed =
        gripVal(controllers.left)  >= CALIB_GRIP_THRESHOLD &&
        gripVal(controllers.right) >= CALIB_GRIP_THRESHOLD;
      if (!calibrating && bothPressed) {
        calibBuffers.left = [];
        calibBuffers.right = [];
        calibFrameCount = 0;
        calibHandednessSeen = new Set();
        calibrating = true;
        calibStartTime = performance.now();
        pulseAllControllers(xrSession, 1.0, 200);  // start signal
        append(`CAPTURING for ${CALIB_DURATION_S}s — twist freely now.`);
      }
      if (calibrating) {
        const dt = (performance.now() - calibStartTime) / 1000;
        if (dt >= CALIB_DURATION_S) {
          calibrating = false;
          pulseAllControllers(xrSession, 1.0, 400);  // end signal — long pulse
          append(`capture window done after ${dt.toFixed(2)}s — ending session…`);
          xrSession.end();  // triggers `end` listener which calls processCalibrationResults()
          return;
        }
      }
      // While in calibMode, suppress the regular xr_frame stream so the
      // teleop server doesn't react to calibration motion (it'd otherwise
      // see grip-clutch and start commanding the simulated arm).
      return;
    }

    // Headset pose so the server can yaw-correct the engage frame to
    // wherever the operator is currently facing.
    let viewer = null;
    const vp = frame.getViewerPose(xrRefSpace);
    if (vp) {
      const vt = vp.transform;
      viewer = {
        position: xyz(vt.position),
        orientation: xyzw(vt.orientation),
      };
    }

    if (Object.keys(controllers).length > 0) {
      wsSend({ type: "xr_frame", t_client: performance.now(), controllers, viewer });
    }
  }
})();
