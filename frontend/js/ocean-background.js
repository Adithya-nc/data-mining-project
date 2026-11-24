/**
 * MarineMine Pro — Hyperrealistic Underwater WebGL Caustics & Ambient Marine Physics
 * Includes:
 * 1. WebGL Caustics, Volumetric God-Rays & Deep-Sea Swell
 * 2. 2D Hydrodynamic Bubbles, Marine Snow & Swimming Fish Silhouettes
 * 3. Interactive Sonar Ping Waves (Cursor click emits radial displacement impulse)
 * 4. Submersible ROV "Spotlight" Mode (Volumetric underwater headlight mask)
 * 5. Web Audio Underwater Ambience (Hydrophone low-pass swell & acoustic sonar ping)
 * 6. Dynamic Gerstner Surface Wave Header Simulation
 */

(function () {
  'use strict';

  // Global Audio Engine using Web Audio API
  window.OceanAudio = {
    ctx: null,
    isPlaying: false,
    noiseNode: null,
    lfo: null,
    lfoGain: null,
    gainNode: null,
    filterNode: null,

    init() {
      if (!this.ctx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioContext();
      }
    },

    toggle() {
      this.init();
      if (this.ctx && this.ctx.state === 'suspended') {
        this.ctx.resume();
      }
      if (this.isPlaying) {
        this.stop();
      } else {
        this.start();
      }
      return this.isPlaying;
    },

    start() {
      if (this.isPlaying) return;
      const ctx = this.ctx;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      // Generate brown/pink noise buffer for deep ocean swell
      const bufferSize = ctx.sampleRate * 2;
      const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const output = noiseBuffer.getChannelData(0);
      let lastOut = 0.0;
      for (let i = 0; i < bufferSize; i++) {
        const white = Math.random() * 2 - 1;
        output[i] = (lastOut + 0.02 * white) / 1.02;
        lastOut = output[i];
        output[i] *= 2.5; // Gain
      }

      this.noiseNode = ctx.createBufferSource();
      this.noiseNode.buffer = noiseBuffer;
      this.noiseNode.loop = true;

      // Deep low-pass resonant filter (oceanic bathymetry attenuation)
      this.filterNode = ctx.createBiquadFilter();
      this.filterNode.type = 'lowpass';
      this.filterNode.frequency.setValueAtTime(140, ctx.currentTime);
      this.filterNode.Q.setValueAtTime(2.8, ctx.currentTime);

      // Low-Frequency Oscillator (LFO) for undulating waves
      this.lfo = ctx.createOscillator();
      this.lfoGain = ctx.createGain();
      this.lfo.frequency.setValueAtTime(0.18, ctx.currentTime);
      this.lfoGain.gain.setValueAtTime(45, ctx.currentTime);
      this.lfo.connect(this.lfoGain);
      this.lfoGain.connect(this.filterNode.frequency);
      this.lfo.start();

      this.gainNode = ctx.createGain();
      this.gainNode.gain.setValueAtTime(0.001, ctx.currentTime);
      this.gainNode.gain.linearRampToValueAtTime(0.22, ctx.currentTime + 0.6);

      this.noiseNode.connect(this.filterNode);
      this.filterNode.connect(this.gainNode);
      this.gainNode.connect(ctx.destination);

      this.noiseNode.start(0);
      this.isPlaying = true;
    },

    stop() {
      this.isPlaying = false;
      const ctx = this.ctx;
      if (!ctx) return;

      try {
        if (this.gainNode) {
          this.gainNode.gain.cancelScheduledValues(ctx.currentTime);
          this.gainNode.gain.setValueAtTime(0, ctx.currentTime);
        }
        if (this.noiseNode) {
          this.noiseNode.stop();
          this.noiseNode.disconnect();
          this.noiseNode = null;
        }
        if (this.lfo) {
          this.lfo.stop();
          this.lfo.disconnect();
          this.lfo = null;
        }
        if (this.lfoGain) {
          this.lfoGain.disconnect();
          this.lfoGain = null;
        }
        if (this.filterNode) {
          this.filterNode.disconnect();
          this.filterNode = null;
        }
        if (this.gainNode) {
          this.gainNode.disconnect();
          this.gainNode = null;
        }
        ctx.suspend();
      } catch (e) {
        console.warn('Hydrophone audio teardown exception:', e);
      }
    },

    playSonarPing() {
      if (!this.ctx) return;
      if (this.ctx.state === 'suspended') this.ctx.resume();
      const ctx = this.ctx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(940, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(420, ctx.currentTime + 0.85);

      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(750, ctx.currentTime);
      filter.Q.setValueAtTime(4.5, ctx.currentTime);

      gain.gain.setValueAtTime(0.18, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.85);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.9);
    }
  };

  function initOceanBackground() {
    const glCanvas = document.getElementById('ocean-canvas');
    const bCanvas = document.getElementById('bubble-canvas');
    const rovCanvas = document.getElementById('rov-spotlight-canvas');
    if (!glCanvas || !bCanvas) return;

    let scrollDepthRatio = 0;
    let mouse = { x: window.innerWidth * 0.5, y: window.innerHeight * 0.3 };
    let mouseNorm = { x: 0.5, y: 0.3 };
    let rovActive = false;
    const sonarPings = []; // Active sonar ripple pulses

    window.addEventListener('mousemove', (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouseNorm.x = e.clientX / window.innerWidth;
      mouseNorm.y = 1.0 - (e.clientY / window.innerHeight);
    });

    // Sonar Ping on Click
    window.addEventListener('click', (e) => {
      // Don't trigger if clicked inside inputs, selects, or buttons
      const target = e.target;
      if (target.closest('input, select, textarea, button, a')) return;

      sonarPings.push({
        x: e.clientX,
        y: e.clientY,
        radius: 0,
        maxRadius: 260,
        speed: 4.5,
        opacity: 0.95
      });

      if (window.OceanAudio.isPlaying) {
        window.OceanAudio.playSonarPing();
      }
    });

    const updateScrollDepth = () => {
      const scrollTop = window.scrollY || document.documentElement.scrollTop || 0;
      const docHeight = (document.documentElement.scrollHeight || document.body.scrollHeight) - window.innerHeight;
      scrollDepthRatio = docHeight > 0 ? Math.min(1, Math.max(0, scrollTop / docHeight)) : 0;
    };
    window.addEventListener('scroll', updateScrollDepth, { passive: true });

    function resizeOceanCanvases() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      glCanvas.width = window.innerWidth * dpr;
      glCanvas.height = window.innerHeight * dpr;
      bCanvas.width = window.innerWidth;
      bCanvas.height = window.innerHeight;
      if (rovCanvas) {
        rovCanvas.width = window.innerWidth;
        rovCanvas.height = window.innerHeight;
      }
    }
    window.addEventListener('resize', resizeOceanCanvases);
    resizeOceanCanvases();

    /* =========================================================================
       1. WEBGL SHADER: CAUSTICS, SUNBEAMS, WATER ABSORPTION & SWELL
       ========================================================================= */
    const gl = glCanvas.getContext('webgl') || glCanvas.getContext('experimental-webgl');
    if (gl) {
      gl.viewport(0, 0, glCanvas.width, glCanvas.height);

      const vsSource = `
        attribute vec2 a_pos;
        void main() {
          gl_Position = vec4(a_pos, 0.0, 1.0);
        }
      `;

      const fsSource = `
        precision highp float;
        uniform vec2 u_res;
        uniform float u_time;
        uniform float u_depth;
        uniform vec2 u_mouse;
        uniform float u_rov;

        mat2 rot(float a) {
          float s = sin(a), c = cos(a);
          return mat2(c, -s, s, c);
        }

        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
            mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
            u.y
          );
        }

        float causticPattern(vec2 uv, float t) {
          vec2 p = mod(uv * 6.2831853, 6.2831853) - 250.0;
          vec2 i = p;
          float c = 1.0;
          float inten = 0.005;
          for (int n = 0; n < 4; n++) {
            float time = t * (1.0 - (3.5 / float(n + 1)));
            i = p + vec2(
              cos(time - i.x) + sin(time + i.y),
              sin(time - i.y) + cos(time + i.x)
            );
            c += 1.0 / length(vec2(
              p.x / (sin(i.x + time) / inten),
              p.y / (cos(i.y + time) / inten)
            ));
          }
          c /= 4.0;
          c = 1.17 - pow(c, 1.4);
          return pow(abs(c), 8.0);
        }

        float godRays(vec2 uv, float t) {
          vec2 sunPos = vec2(0.45 + (u_mouse.x - 0.5) * 0.18, 1.28);
          vec2 dir = uv - sunPos;
          float angle = atan(dir.y, dir.x);
          float dist = length(dir);

          float ray1 = sin(angle * 11.0 + t * 0.45 + noise(vec2(angle * 3.0, t * 0.2)) * 2.5);
          float ray2 = cos(angle * 19.0 - t * 0.32 + sin(uv.x * 4.0 + t * 0.3));
          float rays = smoothstep(0.15, 0.95, ray1 * 0.55 + ray2 * 0.45);

          float attenuation = smoothstep(1.65, 0.1, dist) * smoothstep(0.05, 0.95, uv.y);
          return rays * attenuation;
        }

        void main() {
          vec2 uv = gl_FragCoord.xy / u_res.xy;
          float aspect = u_res.x / u_res.y;
          vec2 p = vec2(uv.x * aspect, uv.y);

          vec3 surfaceTop = vec3(0.03, 0.34, 0.48);
          vec3 midPelagic = vec3(0.012, 0.13, 0.24);
          vec3 deepAbyss  = vec3(0.004, 0.025, 0.065);

          float depthFactor = clamp(u_depth * 0.85, 0.0, 1.0);
          if (u_rov > 0.5) depthFactor = clamp(depthFactor + 0.35, 0.0, 1.0);

          vec3 topCol = mix(surfaceTop, midPelagic, depthFactor);
          vec3 botCol = mix(midPelagic, deepAbyss, clamp(depthFactor + 0.25, 0.0, 1.0));

          vec3 col = mix(botCol, topCol, pow(uv.y, 1.35));

          vec2 waveUV = p * 0.55;
          waveUV.x += sin(uv.y * 5.0 + u_time * 0.4) * 0.04;
          waveUV.y += cos(uv.x * 4.0 + u_time * 0.35) * 0.03;

          float c1 = causticPattern(waveUV + vec2(u_time * 0.015, -u_time * 0.01), u_time * 0.55);
          float c2 = causticPattern(waveUV * 1.45 * rot(0.4) - vec2(u_time * 0.012), u_time * 0.42);
          float caustics = (c1 * 0.65 + c2 * 0.45) * smoothstep(0.15, 1.0, uv.y) * (1.0 - depthFactor * 0.72);

          vec3 causticTint = mix(vec3(0.05, 0.78, 0.92), vec3(0.0, 0.96, 0.83), 0.45);
          col += causticTint * caustics * 0.42;

          float rays = godRays(uv, u_time) * (1.0 - depthFactor * 0.68);
          col += vec3(0.08, 0.65, 0.82) * rays * 0.26;

          float surfaceShimmer = smoothstep(0.88, 1.0, uv.y) * (1.0 - depthFactor * 1.6);
          if (surfaceShimmer > 0.0) {
            float ripple = noise(vec2(p.x * 14.0 + u_time * 1.2, uv.y * 35.0 - u_time * 0.8));
            col += vec3(0.35, 0.92, 1.0) * surfaceShimmer * ripple * 0.28;
          }

          float bioGlow = noise(p * 3.2 + vec2(0.0, -u_time * 0.15));
          col += vec3(0.0, 0.95, 0.83) * smoothstep(0.76, 0.96, bioGlow) * (0.04 + depthFactor * 0.09);

          gl_FragColor = vec4(col, 1.0);
        }
      `;

      function compileShader(type, src) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        return s;
      }

      try {
        const prog = gl.createProgram();
        gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, vsSource));
        gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, fsSource));
        gl.linkProgram(prog);
        gl.useProgram(prog);

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);

        const aPos = gl.getAttribLocation(prog, 'a_pos');
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

        const uRes = gl.getUniformLocation(prog, 'u_res');
        const uTime = gl.getUniformLocation(prog, 'u_time');
        const uDepth = gl.getUniformLocation(prog, 'u_depth');
        const uMouse = gl.getUniformLocation(prog, 'u_mouse');
        const uRov = gl.getUniformLocation(prog, 'u_rov');

        function renderWebGL(now) {
          gl.viewport(0, 0, glCanvas.width, glCanvas.height);
          gl.uniform2f(uRes, glCanvas.width, glCanvas.height);
          gl.uniform1f(uTime, now * 0.001);
          gl.uniform1f(uDepth, scrollDepthRatio);
          gl.uniform2f(uMouse, mouseNorm.x, mouseNorm.y);
          gl.uniform1f(uRov, rovActive ? 1.0 : 0.0);
          gl.drawArrays(gl.TRIANGLES, 0, 6);
          requestAnimationFrame(renderWebGL);
        }
        requestAnimationFrame(renderWebGL);
      } catch (err) {
        console.warn('[Ocean Background] WebGL shader fallback:', err);
      }
    }

    /* =========================================================================
       2. 2D CANVAS: BUBBLES, MARINE SNOW, FISH & SONAR PING DISPLACEMENT
       ========================================================================= */
    const bCtx = bCanvas.getContext('2d');
    if (!bCtx) return;

    class RealisticBubble {
      constructor() {
        this.reset(true);
      }
      reset(initial = false) {
        this.x = Math.random() * bCanvas.width;
        this.y = initial ? Math.random() * bCanvas.height : bCanvas.height + 20;
        this.r = Math.pow(Math.random(), 2.4) * 5.2 + 1.1;
        this.vy = -(0.35 + this.r * 0.24);
        this.vx = 0;
        this.wobbleAmp = 0.25 + this.r * 0.14;
        this.wobbleFreq = 0.02 + Math.random() * 0.025;
        this.phase = Math.random() * Math.PI * 2;
        this.opacity = Math.min(0.65, 0.18 + this.r * 0.07);
      }
      applyImpulse(ix, iy) {
        this.vx += ix;
        this.vy += iy;
      }
      update() {
        this.phase += this.wobbleFreq;
        this.y += this.vy;
        this.x += Math.sin(this.phase) * this.wobbleAmp + this.vx;
        this.vx *= 0.94; // Velocity damping
        if (this.vy < -(0.35 + this.r * 0.24)) {
          this.vy += 0.08;
        }
        const expansion = 1.0 + (1.0 - this.y / Math.max(bCanvas.height, 1)) * 0.18;
        this.drawR = this.r * expansion;
        if (this.y < -25 || this.x < -30 || this.x > bCanvas.width + 30) this.reset(false);
      }
      draw() {
        const r = this.drawR;
        bCtx.save();
        const grad = bCtx.createRadialGradient(
          this.x - r * 0.32, this.y - r * 0.32, r * 0.08,
          this.x, this.y, r
        );
        grad.addColorStop(0, `rgba(220, 252, 255, ${this.opacity * 0.9})`);
        grad.addColorStop(0.45, `rgba(56, 210, 240, ${this.opacity * 0.18})`);
        grad.addColorStop(0.85, `rgba(0, 245, 212, ${this.opacity * 0.42})`);
        grad.addColorStop(1, `rgba(180, 245, 255, ${this.opacity * 0.75})`);

        bCtx.beginPath();
        bCtx.arc(this.x, this.y, r, 0, Math.PI * 2);
        bCtx.fillStyle = grad;
        bCtx.fill();

        if (r > 1.8) {
          bCtx.beginPath();
          bCtx.arc(this.x - r * 0.34, this.y - r * 0.34, r * 0.24, 0, Math.PI * 2);
          bCtx.fillStyle = `rgba(255, 255, 255, ${Math.min(0.9, this.opacity * 1.4)})`;
          bCtx.fill();
        }
        bCtx.restore();
      }
    }

    class MarineSnow {
      constructor() {
        this.x = Math.random() * bCanvas.width;
        this.y = Math.random() * bCanvas.height;
        this.size = Math.random() * 1.6 + 0.4;
        this.vx = (Math.random() - 0.35) * 0.22;
        this.vy = (Math.random() - 0.5) * 0.14;
        this.alpha = Math.random() * 0.35 + 0.08;
      }
      applyImpulse(ix, iy) {
        this.vx += ix * 1.5;
        this.vy += iy * 1.5;
      }
      update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= 0.95;
        if (this.x > bCanvas.width) this.x = 0;
        if (this.x < 0) this.x = bCanvas.width;
        if (this.y > bCanvas.height) this.y = 0;
        if (this.y < 0) this.y = bCanvas.height;
      }
      draw() {
        bCtx.fillStyle = `rgba(186, 236, 255, ${this.alpha})`;
        bCtx.beginPath();
        bCtx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        bCtx.fill();
      }
    }

    class DistantFish {
      constructor() {
        this.reset(true);
      }
      reset(initial = false) {
        this.dir = Math.random() > 0.5 ? 1 : -1;
        this.x = initial ? Math.random() * bCanvas.width : (this.dir === 1 ? -120 : bCanvas.width + 120);
        this.y = (0.15 + Math.random() * 0.65) * bCanvas.height;
        this.scale = 0.45 + Math.random() * 0.65;
        this.speed = (0.28 + Math.random() * 0.35) * this.dir;
        this.tailPhase = Math.random() * Math.PI * 2;
      }
      update() {
        this.x += this.speed;
        this.tailPhase += 0.09;
        this.y += Math.sin(this.tailPhase * 0.25) * 0.15;
        if ((this.dir === 1 && this.x > bCanvas.width + 140) || (this.dir === -1 && this.x < -140)) {
          this.reset(false);
        }
      }
      draw() {
        const wag = Math.sin(this.tailPhase) * 5;
        bCtx.save();
        bCtx.translate(this.x, this.y);
        bCtx.scale(this.dir * this.scale, this.scale);
        bCtx.fillStyle = 'rgba(2, 16, 32, 0.32)';
        bCtx.beginPath();
        bCtx.moveTo(26, 0);
        bCtx.quadraticCurveTo(8, -9, -16, -3);
        bCtx.lineTo(-30, -9 + wag);
        bCtx.lineTo(-26, wag * 0.5);
        bCtx.lineTo(-30, 9 + wag);
        bCtx.lineTo(-16, 3);
        bCtx.quadraticCurveTo(8, 9, 26, 0);
        bCtx.closePath();
        bCtx.fill();
        bCtx.restore();
      }
    }

    const bubbles = Array.from({ length: 55 }, () => new RealisticBubble());
    const snow = Array.from({ length: 60 }, () => new MarineSnow());
    const fishSchool = Array.from({ length: 5 }, () => new DistantFish());

    function animateMarineLife() {
      bCtx.clearRect(0, 0, bCanvas.width, bCanvas.height);

      // 1. Update and Render Sonar Pings & Apply Radial Force Field
      for (let i = sonarPings.length - 1; i >= 0; i--) {
        const ping = sonarPings[i];
        ping.radius += ping.speed;
        ping.opacity = Math.max(0, 1.0 - (ping.radius / ping.maxRadius));

        // Sonar Glowing Ring
        bCtx.save();
        bCtx.beginPath();
        bCtx.arc(ping.x, ping.y, ping.radius, 0, Math.PI * 2);
        bCtx.strokeStyle = `rgba(0, 245, 212, ${ping.opacity * 0.8})`;
        bCtx.lineWidth = 2.5;
        bCtx.shadowColor = '#00f5d4';
        bCtx.shadowBlur = 14;
        bCtx.stroke();

        // Inner secondary pulse
        if (ping.radius > 25) {
          bCtx.beginPath();
          bCtx.arc(ping.x, ping.y, ping.radius - 20, 0, Math.PI * 2);
          bCtx.strokeStyle = `rgba(0, 187, 249, ${ping.opacity * 0.45})`;
          bCtx.lineWidth = 1.2;
          bCtx.stroke();
        }
        bCtx.restore();

        // Push nearby bubbles & marine snow outwards
        const pushRadius = ping.radius;
        const pushThickness = 45;

        bubbles.forEach(b => {
          const dx = b.x - ping.x;
          const dy = b.y - ping.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (Math.abs(dist - pushRadius) < pushThickness) {
            const force = (1.0 - dist / ping.maxRadius) * 2.8;
            b.applyImpulse((dx / dist) * force, (dy / dist) * force);
          }
        });

        snow.forEach(s => {
          const dx = s.x - ping.x;
          const dy = s.y - ping.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (Math.abs(dist - pushRadius) < pushThickness) {
            const force = (1.0 - dist / ping.maxRadius) * 1.8;
            s.applyImpulse((dx / dist) * force, (dy / dist) * force);
          }
        });

        if (ping.radius >= ping.maxRadius) {
          sonarPings.splice(i, 1);
        }
      }

      snow.forEach(s => { s.update(); s.draw(); });
      fishSchool.forEach(f => { f.update(); f.draw(); });
      bubbles.forEach(b => { b.update(); b.draw(); });

      // 2. Render ROV Submersible Headlight if Active
      if (rovCanvas && rovActive) {
        const rCtx = rovCanvas.getContext('2d');
        rCtx.clearRect(0, 0, rovCanvas.width, rovCanvas.height);

        // Dark water mask
        rCtx.save();
        rCtx.fillStyle = 'rgba(1, 6, 14, 0.76)';
        rCtx.fillRect(0, 0, rovCanvas.width, rovCanvas.height);

        // Volumetric Headlight Cone cutout & radial illumination
        rCtx.globalCompositeOperation = 'destination-out';
        const radGrad = rCtx.createRadialGradient(
          mouse.x, mouse.y, 10,
          mouse.x, mouse.y, 280
        );
        radGrad.addColorStop(0, 'rgba(0, 0, 0, 1)');
        radGrad.addColorStop(0.5, 'rgba(0, 0, 0, 0.75)');
        radGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');

        rCtx.beginPath();
        rCtx.arc(mouse.x, mouse.y, 280, 0, Math.PI * 2);
        rCtx.fillStyle = radGrad;
        rCtx.fill();
        rCtx.restore();

        // Glowing ROV Headlight Ring
        rCtx.save();
        rCtx.beginPath();
        rCtx.arc(mouse.x, mouse.y, 14, 0, Math.PI * 2);
        rCtx.fillStyle = 'rgba(0, 245, 212, 0.35)';
        rCtx.strokeStyle = '#00f5d4';
        rCtx.lineWidth = 2;
        rCtx.shadowColor = '#00f5d4';
        rCtx.shadowBlur = 18;
        rCtx.stroke();
        rCtx.fill();
        rCtx.restore();
      }

      requestAnimationFrame(animateMarineLife);
    }
    animateMarineLife();

    /* =========================================================================
       3. TOP NAVBAR BUTTON BINDINGS (ROV MODE & HYDROPHONE AUDIO)
       ========================================================================= */
    const btnRov = document.getElementById('btn-rov-mode');
    if (btnRov) {
      btnRov.addEventListener('click', () => {
        rovActive = !rovActive;
        btnRov.classList.toggle('active', rovActive);
        if (rovCanvas) {
          rovCanvas.classList.toggle('active', rovActive);
          if (!rovActive) {
            const rCtx = rovCanvas.getContext('2d');
            rCtx.clearRect(0, 0, rovCanvas.width, rovCanvas.height);
          }
        }
      });
    }

    const btnAudio = document.getElementById('btn-hydrophone');
    if (btnAudio) {
      btnAudio.addEventListener('click', () => {
        const isPlaying = window.OceanAudio.toggle();
        btnAudio.classList.toggle('active', isPlaying);
        const statusSpan = document.getElementById('hydrophone-status');
        if (statusSpan) {
          statusSpan.textContent = isPlaying ? 'Hydrophone: ON' : 'Hydrophone';
        }
      });
    }

    /* =========================================================================
       4. DYNAMIC GERSTNER SURFACE WAVE HEADER SIMULATION
       ========================================================================= */
    initGerstnerWaveHeader();
  }

  function initGerstnerWaveHeader() {
    const waveCanvas = document.getElementById('surface-wave-canvas');
    if (!waveCanvas) return;
    const ctx = waveCanvas.getContext('2d');
    const windSlider = document.getElementById('wave-wind-slider');
    const windValDisplay = document.getElementById('wave-wind-val');

    let windSpeed = 16; // knots
    if (windSlider) {
      windSlider.addEventListener('input', (e) => {
        windSpeed = parseFloat(e.target.value);
        if (windValDisplay) windValDisplay.textContent = `${windSpeed} kts`;
      });
    }

    function resizeWave() {
      waveCanvas.width = waveCanvas.parentElement.clientWidth || window.innerWidth;
      waveCanvas.height = waveCanvas.parentElement.clientHeight || 85;
    }
    window.addEventListener('resize', resizeWave);
    resizeWave();

    let waveTime = 0;

    function renderWaves() {
      ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
      const w = waveCanvas.width;
      const h = waveCanvas.height;
      waveTime += 0.02 + (windSpeed / 100) * 0.03;

      const seaSeverity = windSpeed / 45; // 0 to 1
      const amp1 = 8 + seaSeverity * 16;
      const amp2 = 6 + seaSeverity * 10;
      const baseHeight = h * 0.52;

      // Layer 1: Deep Back Swell (Deep Cyan/Teal)
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let x = 0; x <= w; x += 6) {
        const y = baseHeight - 6 +
          Math.sin(x * 0.008 + waveTime * 1.1) * amp1 +
          Math.cos(x * 0.018 - waveTime * 0.8) * amp2 * 0.6;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 187, 249, 0.16)';
      ctx.fill();
      ctx.restore();

      // Layer 2: Main Forefront Swell with Foam Crest
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let x = 0; x <= w; x += 5) {
        // Approximate Gerstner Cycloidal steepening: peak sharpening
        const sinVal = Math.sin(x * 0.012 + waveTime * 1.4);
        const peaked = Math.sign(sinVal) * Math.pow(Math.abs(sinVal), 0.85);
        const y = baseHeight +
          peaked * amp1 * 0.85 +
          Math.sin(x * 0.025 - waveTime * 1.6) * (amp2 * 0.4);
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();

      const waveGrad = ctx.createLinearGradient(0, baseHeight - 20, 0, h);
      waveGrad.addColorStop(0, 'rgba(0, 245, 212, 0.32)');
      waveGrad.addColorStop(0.5, 'rgba(4, 28, 54, 0.65)');
      waveGrad.addColorStop(1, 'rgba(2, 11, 22, 0.9)');
      ctx.fillStyle = waveGrad;
      ctx.fill();

      // Crisp Luminescent Wave Crest Rim
      ctx.strokeStyle = 'rgba(0, 245, 212, 0.65)';
      ctx.lineWidth = 1.8;
      ctx.shadowColor = '#00f5d4';
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.restore();

      requestAnimationFrame(renderWaves);
    }
    renderWaves();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initOceanBackground);
  } else {
    initOceanBackground();
  }
})();
