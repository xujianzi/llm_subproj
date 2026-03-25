# ACS Data Explorer — Web App Design Spec

**Date:** 2026-03-25
**Status:** Approved
**Scope:** 在现有 Dash_Agent 项目基础上，新建 `web/` 目录构建 Web 应用，不修改任何现有文件。

---

## 1. 目标

为已有的 ACS（美国社区调查）LLM 查询 Agent 提供 Web 界面，实现：

1. **地图可视化** — Choropleth 地图，多粒度（State / County / City / Zipcode）展示 ACS 指标
2. **Chat Agent** — 自然语言问话 → 自动生成 SQL → 结果联动地图和数据面板

运行环境：本地开发，不考虑生产部署配置。

---

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| 前端框架 | React + Vite + TypeScript |
| 样式 | Tailwind CSS + shadcn/ui |
| 地图 | Mapbox GL JS |
| 表格 | AG Grid |
| 图表 | Recharts |
| 状态管理 | Zustand |
| 后端 | FastAPI (Python) |
| 流式通信 | SSE (Server-Sent Events) |
| 数据库 | PostgreSQL（现有，通过现有 db_utils / query_db 访问）|

---

## 3. 目录结构

```
Dash_Agent/
├── (现有文件，不修改)
│   ├── agent.py
│   ├── query_db.py
│   ├── db_utils.py
│   ├── db_models.py
│   ├── Settings.py
│   └── model_registry.py
│
└── web/
    ├── backend/
    │   ├── main.py               # FastAPI 入口，挂载路由，配置 CORS
    │   ├── routers/
    │   │   ├── map_router.py     # 地图数据 API
    │   │   └── chat_router.py    # Chat Agent SSE API
    │   └── services/
    │       ├── map_service.py    # 查询 ACS 数据 + 合并 GeoJSON
    │       └── chat_service.py   # 封装 agent_loop，生成 SSE 事件流
    │
    └── frontend/
        ├── src/
        │   ├── components/
        │   │   ├── Map/          # Mapbox 地图 + 粒度切换 + Choropleth
        │   │   ├── Chat/         # 悬浮对话框 + 流式打字
        │   │   ├── ConfigPanel/  # 顶部配置栏（Year/Variable/State/County）
        │   │   └── DataPanel/    # Variable Statistics 卡片 + Data Table
        │   ├── api/              # fetch 封装（map API + SSE chat）
        │   ├── store/            # Zustand 全局状态
        │   └── App.tsx
        ├── package.json
        └── vite.config.ts
```

---

## 4. 后端 API 设计

### 4.1 地图数据

```
GET /api/map/data
    ?level=state|county|city|zipcode
    &region=...          # 区域名或编码，支持多值（逗号分隔）
    &variables=...       # ACS 指标列表，逗号分隔
    &year=2020
→ GeoJSON FeatureCollection
  每个 Feature.properties 含所选 ACS 指标值及统计摘要
```

### 4.2 字段列表

```
GET /api/map/variables
→ { "columns": ["population", "median_income", "pct_bachelor", ...] }
  直接调用现有 get_column_names()
```

### 4.3 Chat Agent（流式）

```
POST /api/chat/stream
Body: { "message": "纽约州的中位收入", "history": [...] }
→ SSE 事件流：
  event: text   data: "正在查询..."     # 文字片段，流式打字
  event: data   data: { rows, columns } # 结构化查询结果，触发地图/表格更新
  event: done   data: ""               # 结束标记
```

**chat_service 机制：**
- 复用现有 `agent_loop`（`agent.py`），通过 sys.path 引用父目录模块
- 当 agent 调用 `QueryACSData` 工具获得数据后，额外 yield 一条 `data` 事件
- 对话历史由前端维护，随每次请求传入（后端无状态）

### 4.4 GeoJSON 边界数据

- 使用 US Census TIGER GeoJSON 静态文件，按粒度分文件存储于 `web/backend/geodata/`
- `map_service` 加载对应粒度的 GeoJSON，按 GEOID/zipcode/state FIPS 与 ACS 查询结果 join

---

## 5. 前端布局

### 5.1 整体结构

```
┌──────────────────────────────────────────────────────────┐
│  Tab: [Map Visualization★] [Data Table]                  │
├──────────────────────────────────────────────────────────┤
│  ┌────────────────────── Configure Panel ─────────────┐  │
│  │ Year: [2023▼]  Variable: [▼]  State: [▼]  County:[▼]│ │
│  │              [Update Map & Stats ━━━━━━━━━━━]       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│                  Mapbox 全屏地图                         │
│                  Choropleth 着色                         │
│                  Hover tooltip                           │
│                                                          │
│  ┌──────────────────┐        ╔═══════════════════════╗  │
│  │ Variable Stats   │        ║ Chat Agent     [─] [×] ║  │
│  │ Min / Max / Mean │        ║ > 纽约州的收入...      ║  │
│  │ (左下悬浮卡片)   │        ║ AI: 正在查询...▌       ║  │
│  └──────────────────┘        ║ [输入框]        [发送] ║  │
│                              ╚═══════════════════════╝  │
└──────────────────────────────────────────────────────────┘
```

### 5.2 组件职责

| 组件 | 职责 |
|------|------|
| `<MapView>` | Mapbox GL JS，choropleth layer，粒度切换重载 source，点击/hover 交互 |
| `<ConfigPanel>` | Year / Variable / State / County 下拉选择，触发地图更新 |
| `<ChatPanel>` | 右下角悬浮，可最小化为气泡，SSE 流式接收，`data` 事件触发地图联动 |
| `<StatsCard>` | 左下角悬浮，展示当前变量统计摘要（Min/Max/Mean/Median） |
| `<DataTable>` | Tab 切换后显示，AG Grid 完整数据表格，支持排序筛选 |

### 5.3 视觉风格

- **主题：** 深色
- **底图：** Mapbox `dark-v11`
- **背景：** `#0f1117`，面板 `#1a1d27`，边框 `#2a2d3e`
- **主色：** 蓝青渐变 `#4f7cff → #00d4aa`（按钮、Tab 高亮、Update 按钮）
- **文字：** 主 `#e2e8f0`，次 `#94a3b8`
- **Choropleth 色阶：** 深蓝 → 青绿 → 黄橙（viridis 风格）
- **Chat 悬浮窗：** 毛玻璃效果 `backdrop-blur`

---

## 6. 状态管理（Zustand）

```typescript
{
  // 地图配置
  level: 'state' | 'county' | 'city' | 'zipcode'
  selectedRegions: string[]
  selectedVariable: string
  selectedYear: number
  availableVariables: string[]

  // 地图数据
  geojsonData: GeoJSON.FeatureCollection | null
  stats: { min, max, mean, median } | null

  // Chat
  chatHistory: Message[]
  chatOpen: boolean

  // 联动：chat data 事件触发地图更新
  updateFromChatData: (rows, columns) => void
}
```

---

## 7. 数据流

### 手动查询流
```
用户配置 Year/Variable/Region
  → ConfigPanel 触发 GET /api/map/data
  → map_service: query_db → join GeoJSON
  → 返回 GeoJSON FeatureCollection
  → MapView 更新 choropleth 着色
  → StatsCard 更新统计摘要
```

### Chat 查询流
```
用户输入问题
  → POST /api/chat/stream (SSE)
  → chat_service 驱动 agent_loop
  → text 事件 → ChatPanel 打字效果
  → data 事件 → Zustand updateFromChatData
               → MapView 重绘 + StatsCard 更新
  → done 事件 → 流结束
```

---

## 8. 错误处理

- 后端：FastAPI 统一异常处理，返回 `{ "error": "..." }` JSON
- SSE 错误：发送 `event: error` 事件，前端 Chat 面板显示错误提示
- 地图数据为空：显示"暂无数据"overlay，不崩溃
- 数据库连接失败：复用现有 db_utils 的异常，后端捕获后返回 500

---

## 9. 约束

- 不修改 `Dash_Agent/` 根目录下任何现有文件
- 所有新文件在 `web/` 目录内
- 后端通过 `sys.path` 引用父目录的现有模块（agent.py、query_db.py 等）
