(() => {
  'use strict';

  const MODEL_URL = new URL('models/fin_yolo11n_best.onnx', document.baseURI).href;
  const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
  const INPUT_SIZE = 640;
  const PAD_VALUE = 114;
  const NMS_IOU = 0.45;
  const MAX_CANVAS_SIDE = 1600;
  const CLASS_NAMES = ['fin'];

  const state = {
    session: null,
    provider: null,
    image: null,
    imageWidth: 0,
    imageHeight: 0,
    displayScale: 1,
    objectUrl: null,
    imageName: '',
    letterbox: null,
    lastOutput: null,
    durations: { preprocess: 0, inference: 0 },
    detections: [],
    busy: false,
    confidence: 0.5,
    activeSample: null,
    loadId: 0,
  };

  const el = {
    modelStatus: document.getElementById('modelStatus'),
    modelStatusText: document.getElementById('modelStatusText'),
    fileInput: document.getElementById('fileInput'),
    dropZone: document.getElementById('dropZone'),
    emptyState: document.getElementById('emptyState'),
    canvasWrap: document.getElementById('canvasWrap'),
    canvas: document.getElementById('previewCanvas'),
    fileName: document.getElementById('fileName'),
    imageMeta: document.getElementById('imageMeta'),
    detectBtn: document.getElementById('detectBtn'),
    buttonLabel: document.querySelector('#detectBtn .button-label'),
    resetBtn: document.getElementById('resetBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    confidenceRange: document.getElementById('confidenceRange'),
    confidenceValue: document.getElementById('confidenceValue'),
    resultSummary: document.getElementById('resultSummary'),
    targetCount: document.getElementById('targetCount'),
    topConfidence: document.getElementById('topConfidence'),
    inferenceTime: document.getElementById('inferenceTime'),
    resultsList: document.getElementById('resultsList'),
    resultBadge: document.getElementById('resultBadge'),
    pipeline: document.getElementById('pipeline'),
    logLines: document.getElementById('logLines'),
    toast: document.getElementById('toast'),
  };

  const context = el.canvas.getContext('2d', { alpha: false });
  let toastTimer = null;
  let confidenceTimer = null;

  function formatClock(date = new Date()) {
    return date.toLocaleTimeString('zh-CN', { hour12: false });
  }

  function log(message, tone = '') {
    const row = document.createElement('p');
    if (tone) row.className = tone;
    const time = document.createElement('time');
    time.textContent = formatClock();
    row.append(time, document.createTextNode(message));
    el.logLines.appendChild(row);
    while (el.logLines.children.length > 9) el.logLines.firstElementChild.remove();
    el.logLines.scrollTop = el.logLines.scrollHeight;
  }

  function setModelStatus(kind, text) {
    el.modelStatus.className = `model-status ${kind}`;
    el.modelStatusText.textContent = text;
  }

  function showToast(message, isError = false) {
    clearTimeout(toastTimer);
    el.toast.textContent = message;
    el.toast.className = `toast show${isError ? ' error' : ''}`;
    toastTimer = setTimeout(() => { el.toast.className = 'toast'; }, 3600);
  }

  function setStep(active) {
    el.pipeline.querySelectorAll('li').forEach((item) => {
      const step = Number(item.dataset.step);
      item.classList.toggle('active', step === active);
      item.classList.toggle('done', step < active);
    });
  }

  function setBusy(isBusy) {
    state.busy = isBusy;
    el.detectBtn.classList.toggle('busy', isBusy);
    el.buttonLabel.textContent = isBusy ? '正在检测' : '开始检测';
    el.canvasWrap.classList.toggle('detecting', isBusy);
    updateDetectButton();
  }

  function updateDetectButton() {
    el.detectBtn.disabled = !(state.session && state.image && !state.busy);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function frame() {
    return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 20)));
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  async function loadModel() {
    if (!window.ort) {
      setModelStatus('error', '推理引擎加载失败');
      log('无法加载 ONNX Runtime Web，请检查网络或 CDN 是否可访问。', 'error');
      return;
    }

    window.ort.env.wasm.wasmPaths = ORT_CDN;
    window.ort.env.wasm.numThreads = 1;
    window.ort.env.logLevel = 'error';
    setModelStatus('loading', '正在加载模型');
    log('开始加载 fin_yolo11n_best.onnx（约 10.2 MB）…');

    try {
      if (navigator.gpu) {
        try {
          state.session = await window.ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
          });
          state.provider = 'WebGPU';
        } catch (error) {
          console.warn('WebGPU session failed, falling back to WASM.', error);
          log('WebGPU 不可用，自动切换到 WASM。', 'warn');
        }
      }

      if (!state.session) {
        state.session = await window.ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        state.provider = 'WASM';
      }

      setModelStatus('ready', `${state.provider} · 模型已就绪`);
      log(`模型加载完成，推理后端：${state.provider}。`, 'success');
      updateDetectButton();
      maybeRunDemo();
    } catch (error) {
      console.error(error);
      setModelStatus('error', '模型加载失败');
      log(`模型加载失败：${error.message || error}`, 'error');
      if (location.protocol === 'file:') {
        log('不能直接双击 index.html，请用本地 HTTP 服务打开。', 'warn');
      }
      showToast('模型加载失败，请查看智能体日志。', true);
    }
  }

  function drawCanvas(detections = []) {
    const image = state.image;
    if (!image) return;

    const longest = Math.max(image.naturalWidth, image.naturalHeight);
    const scale = Math.min(1, MAX_CANVAS_SIDE / longest);
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));

    if (el.canvas.width !== width || el.canvas.height !== height) {
      el.canvas.width = width;
      el.canvas.height = height;
    }

    state.displayScale = width / image.naturalWidth;
    context.fillStyle = '#02070a';
    context.fillRect(0, 0, width, height);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(image, 0, 0, width, height);

    for (const detection of detections) {
      drawDetection(detection);
    }
  }

  function drawDetection(detection) {
    const scale = state.displayScale;
    const width = el.canvas.width;
    const height = el.canvas.height;
    const x1 = clamp(detection.x1 * scale, 1, width - 1);
    const y1 = clamp(detection.y1 * scale, 1, height - 1);
    const x2 = clamp(detection.x2 * scale, 1, width - 1);
    const y2 = clamp(detection.y2 * scale, 1, height - 1);
    const boxWidth = Math.max(1, x2 - x1);
    const boxHeight = Math.max(1, y2 - y1);
    const lineWidth = clamp(width * 0.005, 3, 6.5);
    const corner = clamp(Math.min(boxWidth, boxHeight) * 0.16, 10, 38);

    context.save();
    context.strokeStyle = 'rgba(103, 245, 210, 0.98)';
    context.fillStyle = 'rgba(103, 245, 210, 0.09)';
    context.lineWidth = lineWidth;
    context.shadowColor = 'rgba(103, 245, 210, 0.52)';
    context.shadowBlur = 18;
    context.fillRect(x1, y1, boxWidth, boxHeight);
    context.strokeRect(x1, y1, boxWidth, boxHeight);
    context.shadowBlur = 0;

    context.lineWidth = lineWidth * 2.1;
    context.beginPath();
    context.moveTo(x1, y1 + corner);
    context.lineTo(x1, y1);
    context.lineTo(x1 + corner, y1);
    context.moveTo(x2 - corner, y1);
    context.lineTo(x2, y1);
    context.lineTo(x2, y1 + corner);
    context.moveTo(x2, y2 - corner);
    context.lineTo(x2, y2);
    context.lineTo(x2 - corner, y2);
    context.moveTo(x1 + corner, y2);
    context.lineTo(x1, y2);
    context.lineTo(x1, y2 - corner);
    context.stroke();

    const fontSize = clamp(width * 0.026, 18, 38);
    const label = `${detection.label} ${(detection.score * 100).toFixed(1)}%`;
    context.font = `700 ${fontSize}px "Segoe UI", "Microsoft YaHei", sans-serif`;
    const textWidth = context.measureText(label).width;
    const labelWidth = textWidth + fontSize * 1.15;
    const labelHeight = fontSize + 10;
    let labelX = x1;
    let labelY = y1 - labelHeight - 4;
    if (labelY < 2) labelY = y1 + 4;
    if (labelX + labelWidth > width - 2) labelX = width - labelWidth - 2;

    context.fillStyle = '#67f5d2';
    roundedRect(context, labelX, labelY, labelWidth, labelHeight, 5);
    context.fill();
    context.fillStyle = '#04130f';
    context.textBaseline = 'middle';
    context.fillText(label, labelX + fontSize * 0.55, labelY + labelHeight / 2 + 0.5);
    context.restore();
  }

  function setImage(image, name, objectUrl = null, sampleButton = null) {
    if (state.objectUrl && state.objectUrl !== objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
    }

    state.image = image;
    state.imageWidth = image.naturalWidth;
    state.imageHeight = image.naturalHeight;
    state.imageName = name;
    state.objectUrl = objectUrl;
    state.loadId += 1;
    state.lastOutput = null;
    state.detections = [];
    state.letterbox = null;
    state.durations = { preprocess: 0, inference: 0 };

    el.emptyState.hidden = true;
    el.canvasWrap.hidden = false;
    el.dropZone.classList.add('has-image');
    el.fileName.textContent = name;
    el.imageMeta.textContent = `${state.imageWidth} × ${state.imageHeight} px`;
    el.resultSummary.hidden = true;
    el.resultBadge.hidden = true;
    el.downloadBtn.disabled = true;
    el.resetBtn.disabled = false;
    el.resultsList.innerHTML = '<p class="placeholder">图片已就绪，点击“开始检测”。</p>';
    state.activeSample = sampleButton;
    document.querySelectorAll('.sample-button').forEach((button) => {
      button.classList.toggle('active', button === sampleButton);
    });

    drawCanvas();
    setStep(1);
    log(`已载入图片：${name}（${state.imageWidth} × ${state.imageHeight}）。`);
    updateDetectButton();
  }

  function loadFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showToast('请选择 JPG、PNG 或 WebP 图片。', true);
      return;
    }
    if (file.size > 40 * 1024 * 1024) {
      showToast('图片超过 40 MB，请先压缩后再试。', true);
      return;
    }

    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => setImage(image, file.name, url, null);
    image.onerror = () => {
      URL.revokeObjectURL(url);
      showToast('图片解码失败，请换一张图片。', true);
    };
    image.src = url;
  }

  function loadSample(button, autoDetect = false) {
    const image = new Image();
    image.onload = () => {
      setImage(image, `示例 · ${button.dataset.name}`, null, button);
      if (autoDetect && state.session) detect();
    };
    image.onerror = () => showToast('示例图片加载失败。', true);
    image.src = button.dataset.sample;
  }

  function preprocess(image) {
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const scale = Math.min(INPUT_SIZE / image.naturalWidth, INPUT_SIZE / image.naturalHeight);
    const newWidth = Math.round(image.naturalWidth * scale);
    const newHeight = Math.round(image.naturalHeight * scale);
    const dx = Math.floor((INPUT_SIZE - newWidth) / 2);
    const dy = Math.floor((INPUT_SIZE - newHeight) / 2);

    ctx.fillStyle = `rgb(${PAD_VALUE}, ${PAD_VALUE}, ${PAD_VALUE})`;
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(image, dx, dy, newWidth, newHeight);

    const rgba = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const pixels = INPUT_SIZE * INPUT_SIZE;
    const data = new Float32Array(pixels * 3);
    for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
      data[p] = rgba[i] / 255;
      data[pixels + p] = rgba[i + 1] / 255;
      data[pixels * 2 + p] = rgba[i + 2] / 255;
    }

    return {
      tensor: new window.ort.Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      letterbox: { scale, dx, dy, newWidth, newHeight },
    };
  }

  function decodeAndNms(output, threshold, letterbox) {
    const { data, dims } = output;
    if (!dims || dims.length !== 3) throw new Error(`不支持的输出形状：${dims}`);

    let channels;
    let anchors;
    let read;
    if (dims[1] < dims[2]) {
      channels = dims[1];
      anchors = dims[2];
      read = (channel, index) => data[channel * anchors + index];
    } else {
      anchors = dims[1];
      channels = dims[2];
      read = (channel, index) => data[index * channels + channel];
    }

    const classCount = channels - 4;
    if (classCount < 1) throw new Error(`输出通道数异常：${channels}`);

    const candidates = [];
    for (let i = 0; i < anchors; i += 1) {
      let bestScore = 0;
      let bestClass = 0;
      for (let classId = 0; classId < classCount; classId += 1) {
        const score = read(4 + classId, i);
        if (score > bestScore) {
          bestScore = score;
          bestClass = classId;
        }
      }
      if (bestScore < threshold) continue;

      const cx = read(0, i);
      const cy = read(1, i);
      const width = read(2, i);
      const height = read(3, i);
      if (![cx, cy, width, height, bestScore].every(Number.isFinite) || width <= 0 || height <= 0) {
        continue;
      }

      const x1 = clamp((cx - width / 2 - letterbox.dx) / letterbox.scale, 0, state.imageWidth);
      const y1 = clamp((cy - height / 2 - letterbox.dy) / letterbox.scale, 0, state.imageHeight);
      const x2 = clamp((cx + width / 2 - letterbox.dx) / letterbox.scale, 0, state.imageWidth);
      const y2 = clamp((cy + height / 2 - letterbox.dy) / letterbox.scale, 0, state.imageHeight);
      if (x2 - x1 < 1 || y2 - y1 < 1) continue;

      candidates.push({
        x1,
        y1,
        x2,
        y2,
        score: bestScore,
        classId: bestClass,
        label: CLASS_NAMES[bestClass] || `class_${bestClass}`,
      });
    }

    candidates.sort((a, b) => b.score - a.score);
    const kept = [];
    while (candidates.length) {
      const best = candidates.shift();
      kept.push(best);
      for (let i = candidates.length - 1; i >= 0; i -= 1) {
        if (intersectionOverUnion(best, candidates[i]) > NMS_IOU) candidates.splice(i, 1);
      }
    }
    return kept;
  }

  function intersectionOverUnion(a, b) {
    const overlapWidth = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
    const overlapHeight = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    const overlap = overlapWidth * overlapHeight;
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    return overlap / (areaA + areaB - overlap || 1);
  }

  function updateResults(detections) {
    state.detections = detections;
    drawCanvas(detections);

    if (!detections.length) {
      el.resultBadge.hidden = true;
      el.downloadBtn.disabled = true;
      el.resultsList.innerHTML = `<div class="empty-result">未检测到置信度高于 ${state.confidence.toFixed(2)} 的 fin 目标。可尝试降低阈值，或换一张背鳍更清晰的照片。</div>`;
      return;
    }

    el.resultBadge.hidden = false;
    el.downloadBtn.disabled = false;
    el.resultsList.innerHTML = detections.map((item, index) => {
      const confidence = (item.score * 100).toFixed(1);
      const x = Math.round(item.x1);
      const y = Math.round(item.y1);
      const width = Math.round(item.x2 - item.x1);
      const height = Math.round(item.y2 - item.y1);
      return `
        <article class="result-item">
          <div class="result-item-head"><b>${item.label} · #${String(index + 1).padStart(2, '0')}</b><span>${confidence}%</span></div>
          <div class="confidence-track"><div class="confidence-fill" style="width:${confidence}%"></div></div>
          <div class="result-coords">x ${x} · y ${y} · w ${width} · h ${height}</div>
        </article>`;
    }).join('');
  }

  async function detect() {
    if (!state.session || !state.image || state.busy) return;

    setBusy(true);
    setStep(2);
    log('开始预处理：等比缩放、灰边补全、RGB → NCHW。');
    await frame();

    const preStart = performance.now();
    try {
      const runId = state.loadId;
      const sourceImage = state.image;
      const { tensor, letterbox } = preprocess(sourceImage);
      state.letterbox = letterbox;
      const preEnd = performance.now();

      setStep(3);
      log(`输入张量已就绪 [1, 3, ${INPUT_SIZE}, ${INPUT_SIZE}]，执行 ${state.provider} 推理…`);
      await frame();

      const inputName = state.session.inputNames[0];
      const outputName = state.session.outputNames[0];
      const feeds = { [inputName]: tensor };
      const outputs = await state.session.run(feeds);
      const inferEnd = performance.now();
      const output = outputs[outputName];

      if (state.loadId !== runId || state.image !== sourceImage) {
        log('检测期间图片已更换，本次结果已丢弃。', 'warn');
        return;
      }

      state.lastOutput = { data: output.data, dims: Array.from(output.dims) };
      state.durations = { preprocess: preEnd - preStart, inference: inferEnd - preEnd };

      applyConfidence();
      setStep(4);
      updateSummary();

      const count = state.detections.length;
      const top = count ? Math.max(...state.detections.map((item) => item.score)) : 0;
      log(
        count
          ? `检测完成：${count} 个 fin，最高置信度 ${(top * 100).toFixed(1)}%，推理 ${state.durations.inference.toFixed(0)} ms。`
          : `检测完成：当前阈值下未找到 fin，推理 ${state.durations.inference.toFixed(0)} ms。`,
        count ? 'success' : 'warn'
      );
    } catch (error) {
      console.error(error);
      setStep(1);
      log(`检测失败：${error.message || error}`, 'error');
      showToast('检测失败，请查看智能体日志。', true);
    } finally {
      setBusy(false);
    }
  }

  function applyConfidence() {
    if (!state.lastOutput || !state.letterbox) return;
    try {
      const detections = decodeAndNms(state.lastOutput, state.confidence, state.letterbox);
      updateResults(detections);
      if (!state.busy) updateSummary();
    } catch (error) {
      console.error(error);
      log(`后处理失败：${error.message || error}`, 'error');
    }
  }

  function updateSummary() {
    const detections = state.detections;
    el.resultSummary.hidden = false;
    el.targetCount.textContent = String(detections.length);
    el.topConfidence.textContent = detections.length
      ? `${(Math.max(...detections.map((item) => item.score)) * 100).toFixed(1)}%`
      : '—';
    el.inferenceTime.textContent = `${state.durations.inference.toFixed(0)} ms`;
  }

  function downloadResult() {
    if (!state.image || !state.detections.length) return;
    drawCanvas(state.detections);
    el.canvas.toBlob((blob) => {
      if (!blob) return;
      const baseName = state.imageName.replace(/\.[^.]+$/, '').replace(/[^\w\u4e00-\u9fa5-]+/g, '_') || 'fin_detection';
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `${baseName}_fin_result.png`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      log('已导出带检测框的结果图片。', 'success');
    }, 'image/png');
  }

  function reset() {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.image = null;
    state.loadId += 1;
    state.objectUrl = null;
    state.imageName = '';
    state.lastOutput = null;
    state.letterbox = null;
    state.detections = [];
    state.durations = { preprocess: 0, inference: 0 };
    state.activeSample = null;
    el.fileInput.value = '';
    el.emptyState.hidden = false;
    el.canvasWrap.hidden = true;
    el.dropZone.classList.remove('has-image');
    el.resultSummary.hidden = true;
    el.resultBadge.hidden = true;
    el.downloadBtn.disabled = true;
    el.resetBtn.disabled = true;
    el.resultsList.innerHTML = '<p class="placeholder">完成检测后，这里会显示每个目标的位置和置信度。</p>';
    document.querySelectorAll('.sample-button').forEach((button) => button.classList.remove('active'));
    setStep(1);
    log('工作区已清空。');
    updateDetectButton();
  }

  function maybeRunDemo() {
    if (new URLSearchParams(location.search).get('demo') !== '1') return;
    const firstSample = document.querySelector('.sample-button');
    if (firstSample) loadSample(firstSample, true);
  }

  el.fileInput.addEventListener('change', () => {
    loadFile(el.fileInput.files[0]);
    el.fileInput.value = '';
  });

  el.dropZone.addEventListener('click', () => el.fileInput.click());
  el.dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      el.fileInput.click();
    }
  });

  ['dragenter', 'dragover'].forEach((type) => {
    el.dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      el.dropZone.classList.add('is-dragging');
    });
  });

  ['dragleave', 'drop'].forEach((type) => {
    el.dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (type === 'drop' || !el.dropZone.contains(event.relatedTarget)) {
        el.dropZone.classList.remove('is-dragging');
      }
    });
  });

  el.dropZone.addEventListener('drop', (event) => {
    loadFile(event.dataTransfer.files[0]);
  });

  document.addEventListener('paste', (event) => {
    const imageItem = Array.from(event.clipboardData?.items || []).find((item) => item.type.startsWith('image/'));
    if (imageItem) loadFile(imageItem.getAsFile());
  });

  document.querySelectorAll('.sample-button').forEach((button) => {
    button.addEventListener('click', () => loadSample(button, false));
  });

  el.detectBtn.addEventListener('click', detect);
  el.downloadBtn.addEventListener('click', downloadResult);
  el.resetBtn.addEventListener('click', reset);

  el.confidenceRange.addEventListener('input', () => {
    state.confidence = Number(el.confidenceRange.value);
    el.confidenceValue.textContent = state.confidence.toFixed(2);
    clearTimeout(confidenceTimer);
    confidenceTimer = setTimeout(applyConfidence, 80);
  });

  // Keep the upload overlay from opening a file picker when clicking a sample.
  document.querySelectorAll('.sample-button').forEach((button) => {
    button.addEventListener('click', (event) => event.stopPropagation());
  });

  setStep(1);
  updateDetectButton();
  loadModel();

  window.__finDetector = {
    state,
    detect,
    loadSample: (index = 0) => {
      const buttons = document.querySelectorAll('.sample-button');
      if (buttons[index]) loadSample(buttons[index], false);
    },
  };
})();
