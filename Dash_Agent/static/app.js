// ── 状态 ──────────────────────────────────────────────────────────────────────
// chats: { [id]: { title: string, messages: [{role, content}] } }
const chats = {};
let activeId = null;

// ── DOM 引用 ──────────────────────────────────────────────────────────────────
const chatBox     = document.getElementById("chatBox");
const input       = document.getElementById("messageInput");
const sendBtn     = document.getElementById("sendBtn");
const newChatBtn  = document.getElementById("newChatBtn");
const chatList    = document.getElementById("chatList");

// ── 侧边栏渲染 ────────────────────────────────────────────────────────────────
function renderSidebar() {
  chatList.innerHTML = "";
  // 最新的排在最上面
  const ids = Object.keys(chats).reverse();
  for (const id of ids) {
    const item = document.createElement("div");
    item.className = "chat-item" + (id === activeId ? " active" : "");
    item.textContent = chats[id].title || "新对话";
    item.dataset.id = id;
    item.addEventListener("click", () => switchChat(id));
    chatList.appendChild(item);
  }
}

// ── 切换会话 ──────────────────────────────────────────────────────────────────
function switchChat(id) {
  activeId = id;
  chatBox.innerHTML = "";
  // 从本地缓存恢复消息（只渲染 user / assistant 文本气泡）
  for (const msg of chats[id].messages) {
    if (msg.role === "user" || msg.role === "assistant") {
      const content = typeof msg.content === "string" ? msg.content : null;
      if (content) appendMessage(msg.role, content);
    }
  }
  renderSidebar();
}

// ── 新建会话 ──────────────────────────────────────────────────────────────────
function createNewChat() {
  // 如果当前已有空的新会话（还没发过消息），直接切到它即可
  if (activeId && chats[activeId] && chats[activeId].messages.length === 0) {
    return;
  }
  // 生成临时 id，后端首次回复后会返回真实 id（两者一致，因为我们用同一个 uuid）
  const id = crypto.randomUUID();
  chats[id] = { title: "新对话", messages: [] };
  activeId = id;
  chatBox.innerHTML = "";
  renderSidebar();
}

// ── 消息气泡 ──────────────────────────────────────────────────────────────────
function appendMessage(role, content) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  row.appendChild(bubble);
  chatBox.appendChild(row);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// ── 发送消息 ──────────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  // 若还没有任何会话，先创建一个
  if (!activeId) createNewChat();

  appendMessage("user", text);
  input.value = "";

  // 禁用输入，防止重复发送
  sendBtn.disabled = true;
  input.disabled = true;

  const loadingRow = document.createElement("div");
  loadingRow.className = "message-row assistant";
  loadingRow.id = "loadingRow";
  loadingRow.innerHTML = `<div class="bubble loading">思考中...</div>`;
  chatBox.appendChild(loadingRow);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: activeId, message: text }),
    });

    const data = await resp.json();

    // 用后端返回的 id 更新（两者应一致，但防止前后端不同步）
    if (data.conversation_id !== activeId) {
      chats[data.conversation_id] = chats[activeId];
      delete chats[activeId];
      activeId = data.conversation_id;
    }

    // 同步本地消息缓存
    chats[activeId].messages = data.messages;

    // 首条消息后更新标题（取用户消息前 30 字）
    if (chats[activeId].title === "新对话") {
      const firstUser = data.messages.find(m => m.role === "user");
      if (firstUser) {
        const t = firstUser.content;
        chats[activeId].title = t.length > 30 ? t.slice(0, 30) + "…" : t;
      }
    }

    document.getElementById("loadingRow")?.remove();
    appendMessage("assistant", data.answer || "无返回内容");
    renderSidebar();
  } catch (err) {
    document.getElementById("loadingRow")?.remove();
    appendMessage("assistant", "请求失败：" + err.message);
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

// ── 事件绑定 ──────────────────────────────────────────────────────────────────
newChatBtn.addEventListener("click", createNewChat);

sendBtn.addEventListener("click", sendMessage);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── 启动时从后端加载已有会话列表 ─────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch("/api/conversations");
    const list = await res.json();
    for (const { conversation_id, title } of list) {
      chats[conversation_id] = { title, messages: [] };
    }
    if (list.length > 0) {
      // 加载最新一条会话的消息
      const latest = list[list.length - 1].conversation_id;
      const detail = await fetch(`/api/conversations/${latest}`);
      const detailData = await detail.json();
      chats[latest].messages = detailData.messages;
      switchChat(latest);
    } else {
      renderSidebar();
    }
  } catch {
    renderSidebar();
  }
})();
