const state = { strategy: "function_calling", sessionId: null };

const statusBadge = document.getElementById("status-badge");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const dropzoneLabel = document.getElementById("dropzone-label");
const uploadStatus = document.getElementById("upload-status");
const strategyPicker = document.getElementById("strategy-picker");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const traceLog = document.getElementById("trace-log");
const traceMeta = document.getElementById("trace-meta");
const finalAnswerBox = document.getElementById("final-answer");
const finalAnswerBody = document.getElementById("final-answer-body");
const sessionIdLabel = document.getElementById("session-id");
const newSessionBtn = document.getElementById("new-session-btn");

const STEP_LABELS = {
  thought: "Thought",
  tool_call: "Action",
  observation: "Observation",
  final_answer: "Final answer",
};

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const mode = data.llm_mode === "real" ? "● real API mode" : "● mock offline mode";
    statusBadge.textContent = `${mode} · vectors: ${data.vector_backend} · history: ${data.chat_history_backend}`;
    statusBadge.classList.add(data.llm_mode === "real" ? "mode-real" : "mode-mock");
  } catch (e) {
    statusBadge.textContent = "offline";
  }
}

// --------------------------------------------------------------- session ---
// A session is created once (page load) so chat turns get attached to it and
// persisted server-side (Postgres when CHAT_HISTORY_BACKEND=postgres, an
// in-process dict otherwise -- either way the API doesn't need the frontend
// to know which). "New session" just starts a fresh one; it does not delete
// the old one, history for it is still in the database if you want it back
// via GET /api/sessions/{id}/messages.
async function createSession() {
  try {
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    state.sessionId = data.id;
    sessionIdLabel.textContent = data.id.slice(0, 8) + "…";
    sessionIdLabel.title = data.id;
  } catch (e) {
    sessionIdLabel.textContent = "session unavailable";
  }
}

newSessionBtn.addEventListener("click", () => {
  traceLog.innerHTML = "";
  finalAnswerBox.hidden = true;
  createSession();
});

strategyPicker.addEventListener("click", (e) => {
  const btn = e.target.closest(".strategy-btn");
  if (!btn) return;
  document.querySelectorAll(".strategy-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  state.strategy = btn.dataset.strategy;
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => e.preventDefault());
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    handleUpload(e.dataTransfer.files[0]);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleUpload(fileInput.files[0]);
});

async function handleUpload(file) {
  dropzoneLabel.textContent = file.name;
  uploadStatus.textContent = "Indexing…";
  uploadStatus.classList.remove("error");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
    const data = await res.json();
    uploadStatus.textContent = `Indexed "${data.filename}" ✓`;
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.classList.add("error");
  }
}

function appendStep(step) {
  document.querySelector(".empty-state")?.remove();
  const div = document.createElement("div");
  div.className = `trace-step ${step.step_type}`;
  const label = document.createElement("span");
  label.className = "step-label";
  label.textContent = STEP_LABELS[step.step_type] || step.step_type;
  div.appendChild(label);
  div.appendChild(document.createTextNode(step.content));
  traceLog.appendChild(div);
  traceLog.scrollTop = traceLog.scrollHeight;
}

sendBtn.addEventListener("click", runAgent);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runAgent();
});

// ----------------------------------------------------------- SSE parsing ---
// True Server-Sent Events framing from /api/chat/stream:
//   event: <step_type>\n
//   data: {"step_type": "...", "content": "..."}\n
//   \n                                            <- blank line ends the frame
//
// EventSource can't be used here since it's GET-only and this endpoint takes
// a JSON body (query/strategy/session_id) via POST -- so we read the raw
// response stream with fetch() and parse the SSE framing by hand instead.
// Frames are separated by a blank line ("\n\n"), never split across reads by
// assumption -- the buffer below re-joins partial reads before splitting, so
// a frame arriving split across two network chunks still parses correctly.
function parseSSEFrame(frameText) {
  let data = null;
  for (const line of frameText.split("\n")) {
    if (line.startsWith("data:")) {
      data = line.slice(5).trim();
    }
  }
  return data ? JSON.parse(data) : null;
}

async function runAgent() {
  const query = queryInput.value.trim();
  if (!query) return;
  if (!state.sessionId) await createSession();

  traceLog.innerHTML = "";
  finalAnswerBox.hidden = true;
  sendBtn.disabled = true;
  sendBtn.textContent = "Running…";
  const startedAt = performance.now();

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, strategy: state.strategy, session_id: state.sessionId }),
    });
    if (!res.ok || !res.body) throw new Error(`Server returned ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop(); // last chunk may be an incomplete frame -- keep it for the next read

      for (const frameText of frames) {
        if (!frameText.trim()) continue;
        const frame = parseSSEFrame(frameText);
        if (!frame) continue;
        if (frame.step_type === "done") {
          finalAnswerBody.textContent = frame.content;
          finalAnswerBox.hidden = false;
        } else {
          appendStep(frame);
        }
      }
    }
  } catch (err) {
    appendStep({ step_type: "observation", content: `Error: ${err.message}` });
  } finally {
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(2);
    traceMeta.textContent = `${state.strategy} · ${elapsed}s (client-observed) · session ${state.sessionId ? state.sessionId.slice(0, 8) : "n/a"}`;
    sendBtn.disabled = false;
    sendBtn.textContent = "Run agent";
  }
}

loadHealth();
createSession();
