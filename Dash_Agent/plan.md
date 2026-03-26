# utils 暴露的接口
![alt text](image.png)



# plan
我现在的目录实现了基于 LLM的智能数据查询助手，让用户能够用中文自然语言查询美国社区调查（ACS）数据库。Agent 自动完成语言映射、字段识别和 SQL 构建，无需用户了解数据库结构。但是目前的操作测试还是停留在代码执行模式，
我计划稍后实现一个简单的 Web 界面，让用户能够更方便地与 Agent 交互。
具体实现下面功能
- 根据我的数据库创建前后端，后端我希望使用fastapi，前段的技术栈我不清楚有哪些，请你根据我的需求选择
- 我希望实现下面的功能，首先前端页面的展示风格相对要审美上好看高级一些 
- 我希望通过地图的方式展示我的acs数据，用户可以手动选择地图上的区域，然后展示该区域的acs数据
- 同时前段还应该有一个chat agent功能，当用户不想手动查询的时候，可以通过向agent问话， eg:向我展示纽约州的median house hold income；这个时候agent就能根据用户的问话，自动生成sql查询语句，然后展示查询结果
- 不要修当前目录的文件，创建新的文件或者目录，做到工作区规范整洁
# ACS Data Explorer — Web App Design Spec

**Date:** 2026-03-25
**Status:** Approved
**Scope:** 在现有 Dash_Agent 项目基础上，新建 `web/` 目录构建 Web 应用，不修改任何现有文件。

---
# ACS Data Explorer — Web App Design Spec

**Date:** 2026-03-25
**Status:** Approved
**Scope:** 在现有 Dash_Agent 项目基础上，新建 `web/` 目录构建 Web 应用，不修改任何现有文件。

---

## 1. 目标

为已有的 ACS（美国社区调查）LLM 查询 Agent 提供 Web 界面，实现：

1. **地图可视化** — Choropleth 地图，多粒度（State / County / City / Zipcode）展示 ACS 指标
2. **Chat Agent** — 自然语言问话 → 自动调用工具 → 结果联动地图和数据面板

运行环境：本地开发，uvicorn 单进程单 worker，不考虑生产部署配置。

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
| 后端 | FastAPI (Python)，uvicorn 单进程 |
| 流式通信 | SSE (Server-Sent Events)，使用 `sse-starlette` |
| 数据库 | PostgreSQL（现有，通过现有 db_utils / query_db 访问）|

---

## 3. 目录结构

```
Dash_Agent/
├── (现有文件，不修改)
│   ├── agent.py
│   ├── query_db.py
│   ├── db_utils.py
│   ├── db_models.py          ← Pydantic tool schemas (GetColumnNames, QueryACSData)
│   ├── model_registry.py
│   └── Settings.py
│
└── web/
    ├── backend/
    │   ├── main.py               # FastAPI 入口，CORS 配置，路由注册
    │   ├── requirements.txt      # FastAPI, uvicorn, sse-starlette, psycopg2-binary 等
    │   ├── routers/
    │   │   ├── map_router.py     # 地图数据 API
    │   │   └── chat_router.py    # Chat Agent SSE API
    │   ├── services/
    │   │   ├── map_service.py    # 查询 ACS 数据 + 合并 GeoJSON
    │   │   └── chat_service.py   # 封装 agent_loop，生成 SSE 事件流
    │   ├── geodata/              # 静态 GeoJSON 边界文件（经 mapshaper 简化）
    │   │   ├── states.geojson    # 56 features，~1MB 简化版
    │   │   ├── counties.geojson  # ~3200 features，~5MB 简化版
    │   │   └── zcta.geojson      # ~33000 features，按需分州切片，见第 4.4 节
    │   └── data/
    │       ├── state_fips.json   # 州缩写 → FIPS 映射：{"NY": "36", "CA": "06", ...}
    │       └── county_names.json # state → county 列表（供前端下拉使用）
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
        ├── .env.local            # VITE_MAPBOX_TOKEN=...
        ├── package.json
        └── vite.config.ts        # proxy: /api → http://localhost:8000
```

---

## 4. 后端 API 设计

### 4.1 地图数据

```
GET /api/map/data
    ?level=state|county|zipcode          # city 粒度通过 county 近似
    &state=NY                            # 州过滤（两字母缩写）
    &county=Kings                        # county 过滤（配合 level=zipcode 使用）
    &variables=median_income,population  # 逗号分隔的 ACS 字段名
    &year=2020

→ {
    "geojson": GeoJSON.FeatureCollection,  # Feature.properties 含 ACS 指标值
    "stats": {                              # 当前变量在返回集合内的统计摘要
      "variable": "median_income",
      "min": 30000,
      "max": 120000,
      "mean": 65000,
      "median": 62000
    }
  }
```

**参数说明：**
- `level=state`：返回全美各州数据，忽略 state/county 过滤
- `level=county`：必须提供 `state`，返回该州所有县数据
- `level=zipcode`：必须提供 `state`，可选 `county`，返回对应范围内 ZCTA 数据

### 4.2 可用字段列表

```
GET /api/map/variables
→ { "columns": ["population", "median_income", "pct_bachelor", ...] }
  直接调用现有 get_column_names()
```

### 4.3 地理过滤项（联动下拉数据源）

```
GET /api/map/regions?level=county&state=NY
→ { "regions": ["Kings", "Queens", "New York", "Bronx", "Richmond", ...] }
  从 county_names.json 静态文件读取，无需查库
```

### 4.4 Chat Agent（SSE 流式）

```
POST /api/chat/stream
Content-Type: application/json
Body: {
  "message": "纽约州的中位收入",
  "history": [                        # 仅含 user/assistant 文本轮次
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

→ SSE 事件流（text/event-stream）：
  event: text   data: "正在查询..."     # 文字片段，流式打字
  event: data   data: <JSON>            # 结构化查询结果，见下方格式
  event: error  data: "错误信息"        # 错误后立即跟 done
  event: done   data: ""               # 流结束标记（所有路径均发送）
```

**`data` 事件的 JSON 格式：**
```json
{
  "rows": [{"state": "NY", "county": "Kings", "median_income": 65000}, ...],
  "columns": ["state", "county", "median_income"],
  "geojson": { /* GeoJSON FeatureCollection，可直接用于地图渲染 */ },
  "stats": { "variable": "median_income", "min": ..., "max": ..., "mean": ..., "median": ... }
}
```

**`history` 格式说明：**
- 前端仅维护纯文本对话（user/assistant content 字段），不包含 tool_calls 结构
- `chat_service` 将 history 转为 `agent_loop` 所需的 OpenAI message 格式（只保留文本轮次，工具调用记录不传回前端）
- agent 每次调用从完整历史+新 message 开始，无需客户端感知工具调用细节

**chat_service 的 SSE 实现策略（解决 agent_loop 同步问题）：**

`agent_loop` 是同步阻塞函数（返回 `None`，最终文字存于 `input_messages[-1]["content"]`），无法直接 yield。采用以下方案：

1. 每个请求构造独立的 `local_handlers` 字典（不修改全局 `TOOL_HANDLERS`），将 `QueryACSData` 替换为带副作用的包装函数；`agent_loop` 在调用时使用 `local_handlers` 派发（实现时在 `chat_service` 内部构建局部工具派发逻辑，绕开全局状态，避免并发请求互相污染）
2. 包装函数执行原始查询后，将结果（rows + 推断的 level）写入请求专属的 `asyncio.Queue`
3. `chat_service` 用 `asyncio.to_thread` 在线程池中运行 agent 逻辑
4. 同时从 Queue 异步读取工具结果，每次读到后：构建 GeoJSON（见 level 推断规则）→ yield `data` 事件
5. agent 逻辑结束后，从 `input_messages[-1]["content"]` 读取最终文字 → yield `text` 事件 → yield `done`

**level 推断规则（chat_service 构建 GeoJSON 时使用）：**

agent 调用 `QueryACSData` 时不带显式 level 参数，`chat_service` 根据返回行的字段组合推断：
- 行中含 `zipcode` 且非空 → `level=zipcode`
- 行中含 `county` 且非空，无 zipcode → `level=county`
- 否则 → `level=state`

推断出 level 后，加载对应边界 GeoJSON 并 join，结果写入 `data` 事件。

> 注意：此方案中文字输出不是逐 token 流式的（LLM 调用为同步），而是 agent 完成后一次性发送。对本地使用场景可接受。若需 token 级流式，需修改 agent.py 使用 streaming=True，超出本期范围。

### 4.5 GeoJSON 边界数据策略

| 粒度 | 数据来源 | 简化方式 | 文件大小目标 | join 键 |
|------|----------|----------|-------------|---------|
| State | TIGER State | mapshaper 简化至 0.5% | ~0.5MB | `STUSPS`（州缩写） = ACS `state` |
| County | TIGER County | mapshaper 简化至 0.5% | ~3MB | `STUSPS`+`NAME` = ACS `state`+`county` |
| Zipcode | TIGER ZCTA | 按州切片 + 简化至 1% | 每州 ~2-5MB | `ZCTA5CE20` = ACS `zipcode` |

**join 键映射规则：**
- State：GeoJSON `STUSPS`（如 "NY"）直接匹配 ACS `state` 字段
- County：`STUSPS` + `NAME`（如 "NY" + "Kings"）匹配 ACS `state` + `county`，大小写不敏感
- Zipcode：`ZCTA5CE20` 直接匹配 ACS `zipcode`（均为五位字符串）

**State → FIPS 映射：**
`web/backend/data/state_fips.json` 存储 `{"NY": "36", "CA": "06", ...}`，供 map_service 按需转换。

**Zipcode 文件按州切片：**
`geodata/zcta/` 下按州缩写命名（`zcta_NY.geojson`、`zcta_CA.geojson` 等），`map_service` 根据请求参数动态加载对应文件，避免一次性加载全量 500MB。

---

## 5. 前端布局

### 5.1 整体结构

```
┌──────────────────────────────────────────────────────────┐
│  Tab: [Map Visualization★] [Data Table]                  │
├──────────────────────────────────────────────────────────┤
│  ┌────────────── Configure Panel（可折叠）──────────────┐  │
│  │ Year: [2023▼]  Variable: [▼]  State: [▼]  County:[▼] │ │
│  │              [Update Map & Stats ━━━━━━━━━━━]         │ │
│  └──────────────────────────────────────────────────────┘  │
│                                                          │
│                  Mapbox 全屏地图                         │
│                  Choropleth 着色                         │
│                  Hover tooltip（显示区域名+指标值）       │
│                                                          │
│  ┌──────────────────┐        ╔══════════════════════╗   │
│  │ Variable Stats   │        ║ Chat Agent    [─][×] ║   │
│  │ Min / Max / Mean │        ║ > 纽约州的收入...    ║   │
│  │ Median           │        ║ AI: 查询完成 ✓       ║   │
│  │ (左下悬浮卡片)   │        ║ [数据表格]           ║   │
│  └──────────────────┘        ║ [输入框]      [发送] ║   │
│                              ╚══════════════════════╝   │
└──────────────────────────────────────────────────────────┘
```

最小化后 Chat 面板收缩为右下角圆形气泡按钮。

### 5.2 组件职责

| 组件 | 职责 |
|------|------|
| `<MapView>` | Mapbox GL JS，choropleth layer，粒度切换重载 GeoJSON source，hover/click tooltip |
| `<ConfigPanel>` | Year / Variable / State / County 联动下拉，County 数据来自 `/api/map/regions`，触发地图更新 |
| `<ChatPanel>` | 右下角悬浮，可最小化，SSE 流式接收，`data` 事件触发地图和 StatsCard 更新 |
| `<StatsCard>` | 左下角悬浮，展示当前变量 Min/Max/Mean/Median，数据来自 API 响应 `stats` 字段 |
| `<DataTable>` | Tab 切换后显示，AG Grid 完整数据表格，支持排序筛选 |

### 5.3 视觉风格

- **主题：** 深色
- **底图：** Mapbox `dark-v11`
- **背景：** `#0f1117`，面板 `#1a1d27`，边框 `#2a2d3e`
- **主色：** 蓝青渐变 `#4f7cff → #00d4aa`（按钮、Tab 高亮、Update 按钮）
- **文字：** 主 `#e2e8f0`，次 `#94a3b8`
- **Choropleth 色阶：** 深蓝 → 青绿 → 黄橙（viridis 风格）
- **Chat 悬浮窗：** 毛玻璃效果 `backdrop-blur + bg-opacity-80`

### 5.4 Mapbox Token 配置

前端通过 `frontend/.env.local` 注入（不提交到 git）：
```
VITE_MAPBOX_TOKEN=pk.eyJ1...
```

`vite.config.ts` 配置 API 代理，避免跨域：
```ts
server: { proxy: { '/api': 'http://localhost:8000' } }
```

---

## 6. 状态管理（Zustand）

```typescript
{
  // 地图配置
  level: 'state' | 'county' | 'zipcode'
  selectedState: string | null
  selectedCounty: string | null
  selectedVariable: string
  selectedYear: number
  availableVariables: string[]
  availableCounties: string[]       // 随 selectedState 变化动态加载

  // 地图数据
  geojsonData: GeoJSON.FeatureCollection | null
  stats: { variable: string; min: number; max: number; mean: number; median: number } | null

  // Chat
  chatHistory: { role: 'user' | 'assistant'; content: string }[]  // 仅文本
  chatOpen: boolean

  // 联动：chat data 事件直接更新地图（data 事件含 geojson + stats）
  updateFromChatData: (payload: { geojson: GeoJSON.FeatureCollection; stats: Stats }) => void
}
```

**联动说明：** `data` 事件包含完整 GeoJSON，前端直接更新 `geojsonData` 和 `stats`，无需重新请求 `/api/map/data`。

---

## 7. 数据流

### 手动查询流
```
用户配置 Year/Variable/State/County → 点击 Update
  → GET /api/map/data?level=...&state=...&variables=...&year=...
  → map_service:
      1. query_acs_data(variables, state, county, year)
      2. 加载对应粒度 GeoJSON（按州切片）
      3. join ACS 数据到 Feature.properties
      4. 计算 stats（min/max/mean/median）
  → 返回 { geojson, stats }
  → MapView 更新 choropleth + StatsCard 更新
```

### Chat 查询流
```
用户输入问题 → POST /api/chat/stream (SSE)
  → chat_service:
      1. 构建 input_messages（history + new message）
      2. 注入带副作用的 QueryACSData 包装函数到 TOOL_HANDLERS
      3. asyncio.to_thread 运行 agent_loop
      4. 工具执行后：构建 GeoJSON → yield data 事件（含 geojson + stats）
      5. agent_loop 返回后：yield text 事件（最终文字回复）
      6. yield done 事件
  → 前端：
      text 事件 → ChatPanel 显示 AI 文字
      data 事件 → updateFromChatData → MapView + StatsCard 更新
      done 事件 → 流结束，解除 loading 状态
      error 事件 → ChatPanel 显示错误，紧接 done 事件关闭流
```

---

## 8. 错误处理

- 后端：FastAPI 统一异常处理，HTTP 错误返回 `{ "error": "..." }` JSON
- SSE：所有路径（包括错误路径）均发送 `event: done`，前端以 done 为流结束信号
- 地图数据为空（ACS 无对应记录）：返回空 FeatureCollection，前端显示"暂无数据" overlay
- 数据库连接失败：db_utils 异常由后端捕获，返回 HTTP 500
- Mapbox Token 缺失：前端启动时检测，显示配置提示页

---

## 9. 依赖管理

**`web/backend/requirements.txt`**（独立于现有项目根目录依赖）：
```
fastapi
uvicorn[standard]
sse-starlette
psycopg2-binary
pydantic-settings
python-dotenv
openai
```

共享同一个 venv（在 `Dash_Agent/` 根目录激活），`sys.path` 插入父目录引用现有模块。

**`web/frontend/package.json`** 关键依赖：
```json
{
  "dependencies": {
    "mapbox-gl": "^3",
    "react": "^18",
    "zustand": "^4",
    "ag-grid-react": "^31",
    "recharts": "^2",
    "@shadcn/ui": "latest"
  }
}
```

---

## 10. 约束

- 不修改 `Dash_Agent/` 根目录下任何现有文件
- 所有新文件在 `web/` 目录内（含 `docs/`）
- 后端通过以下方式引用现有模块（防止重复插入路径）：
  ```python
  _root = str(Path(__file__).parent.parent.parent)
  if _root not in sys.path:
      sys.path.insert(0, _root)
  ```
- uvicorn 本地运行单进程（`uvicorn main:app --reload`），与现有连接池（maxconn=10）兼容
- CORS 配置允许 `http://localhost:5173`（Vite 开发服务器默认端口）

# Terminal 1 — backend                                                                                                                                                                                                                              cd I:/LLM_proj/llm_subproj/Dash_Agent/web                                                                                                                                  