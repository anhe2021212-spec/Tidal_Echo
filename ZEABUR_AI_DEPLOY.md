# 给 Zeabur 部署助手的执行指令

你现在负责把这个仓库部署到 Zeabur，并让用户可以通过网页使用第三方 OpenAI 兼容 API，同时保留 MCP 工具接入入口。

请严格按照下面的目标、约束和验证步骤执行。不要把本项目改造成 Claude Code 专用部署，也不要要求用户安装 nginx、systemd 或本地 Claude Code。

## 一、部署目标

这是一个单用户私密聊天应用，包含：

* FastAPI Relay 后端

* API Loop：调用用户填写的第三方 OpenAI 兼容 API

* `web/` 静态前端

* MCP HTTP JSON-RPC 工具接入

* SQLite 聊天记录、API 配置、MCP 配置和上传文件

采用单容器、单 Zeabur 服务部署：

* Relay、API Loop 和前端运行在同一个容器中

* Relay 监听 Zeabur 提供的 `$PORT`

* API Loop 监听容器内部的 `3020` 端口

* 前端由 Relay 直接托管

* 所有持久化数据写入 `/data`

* 不需要 nginx 反向代理

## 二、部署前检查

先检查仓库根目录是否存在以下文件：

* `Dockerfile`

* `zeabur-start.sh`

* `.dockerignore`

* `backend/app.py`

* `backend/requirements.txt`

* `examples/api_loop.py`

* `web/index.html`

优先使用仓库已有的 `Dockerfile` 和 `zeabur-start.sh`，不要重新设计启动架构。

检查以下内容：

1. `Dockerfile` 必须复制 `backend/`、`examples/` 和 `web/`。
2. 容器启动命令必须是 `/app/zeabur-start.sh`。
3. Relay 必须监听 `0.0.0.0`，端口使用 `$PORT`。
4. API Loop 必须能访问 `http://127.0.0.1:3020/loop/ingest`。
5. Relay 必须能直接托管 `web/`，并兼容 `/relay/...` 路径。
6. 不要提交、打印或回显任何真实 API Key、Token 或连接密钥。

如果以上文件缺失，先报告缺失文件，不要擅自创建与现有架构冲突的替代服务。

## 三、创建 Zeabur 服务

在 Zeabur 中从当前 Git 仓库创建一个服务：

* 构建方式：Dockerfile

* Dockerfile 路径：`/Dockerfile`

* 服务根目录：仓库根目录 `/`

* 不要使用 `backend/` 作为构建根目录，否则 Dockerfile 无法复制 `web/` 和 `examples/`

* 对外端口使用 Zeabur 自动提供的 `$PORT`

* 协议使用 HTTP

为该服务创建一个 Volume：

* 挂载路径：`/data`

这是必须配置的。没有 `/data` 持久化卷，服务重启或重新部署后可能丢失聊天记录、上传文件、API 配置和 MCP 配置。

## 四、设置环境变量

请在 Zeabur 的服务环境变量中设置以下变量。

必填变量：

```env
RELAY_SECRET=<由用户自己生成的随机长字符串>
RELAY_AI_NAME=<用户希望显示的 AI 名称>
RELAY_HUMAN_NAME=<用户希望使用的称呼>

LLM_API_BASE=<用户的第三方 OpenAI 兼容 API 基址>
LLM_API_KEY=<用户的第三方 API Key>
LLM_MODEL=<用户的模型名称>
```

默认运行变量：

```env
RELAY_PORT=$PORT
LOOP_PORT=3020
RELAY_DB=/data/relay.db
RELAY_UPLOAD_DIR=/data/uploads
RELAY_BRAIN_FILE=/data/brain_target
LOOP_CONFIG=/data/api_loop.config.json
RELAY_WEB_DIR=/app/web
RELAY_PUBLIC_PREFIX=/relay
RELAY_APP_PATH=/
RELAY_LOOP_INGEST_URL=http://127.0.0.1:3020/loop/ingest
```

允许前端访问的来源：

```env
RELAY_ALLOW_ORIGINS=*
```

如果 Zeabur 为服务绑定了固定公网域名，也可以把它改成该域名，例如：

```env
RELAY_ALLOW_ORIGINS=https://example.zeabur.app
```

可选变量：

```env
HISTORY_N=24
LLM_MAX_TOKENS=2000
LLM_TEMPERATURE=0.7
LOOP_STREAM=1
```

MCP 工具模式变量（一般不填，有自动探测；仅在第三方中转端工具行为异常时使用）：

```env
# 逗号分隔的模型名列表：这些模型强制用提示词工具协议（<tool_call> 文本模式），
# 适用于不支持原生 tools 参数的中转端。免去每次先失败一次才自动切换。
LLM_PROMPT_TOOLS=<模型名,另一模型名>

# 设为 1 时强制所有模型走原生 tools 参数，禁用自动切换提示词模式（不建议）。
FORCE_NATIVE_TOOLS=0
```

不要把第三方 API 地址误填成完整的 `/chat/completions` 路径。程序会自动在 `LLM_API_BASE` 后面追加 `/chat/completions`。

示例：

```env
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

## 五、第三方 API 要求

用户使用的 AI 不是 Claude Code，而是通过 URL、Key 和模型名称连接的第三方 API。

因此必须使用 OpenAI Chat Completions 兼容格式，至少支持：

* `POST {LLM_API_BASE}/chat/completions`

* 请求字段：`model`、`messages`

* 请求头：`Authorization: Bearer <LLM_API_KEY>`

* 响应字段：`choices[0].message.content`

如果要启用流式回复，服务还应支持 SSE 格式的 `data:` 响应；不支持流式时，可以把 `LOOP_STREAM=0`。

不要要求用户配置 Claude API、Anthropic SDK 或 Claude Code。

## 六、MCP 接入要求

MCP 不通过 Claude Code 接入，而是由 `examples/api_loop.py` 直接调用 HTTP MCP Server。

用户部署完成后，在网页设置页面填写：

* MCP 服务名称

* MCP Server URL

* 可选鉴权 Token

* 是否启用 MCP 工具

MCP Server 必须支持 HTTP JSON-RPC，并提供：

```text
POST <MCP_SERVER_URL>
method: tools/list
method: tools/call
```

当前接入方式使用：

```http
Content-Type: application/json
Authorization: Bearer <MCP_TOKEN>
```

如果 MCP 服务只支持本地 stdio，不能直接填入网页 URL；不要把 stdio 地址伪装成 HTTP 地址。

## 七、构建和启动验证

部署后按以下顺序验证，不要只看构建是否成功。

### 1. 健康检查

访问：

```text
https://<Zeabur域名>/healthz
```

期望返回 JSON，并且 `ok` 为 `true`。

### 2. 前端检查

访问：

```text
https://<Zeabur域名>/
```

应看到聊天登录页面，而不是 404 或默认欢迎页。

### 3. 密钥鉴权检查

使用用户设置的 `RELAY_SECRET` 调用：

```bash
curl -s https://<Zeabur域名>/relay/healthz \\
  -H "Authorization: Bearer <RELAY_SECRET>"
```

未携带密钥访问受保护接口时必须返回 401。

### 4. API 配置入口检查

登录网页后打开设置页面，确认可以看到：

* 第三方 API 配置区域

* API URL 输入框

* API Key 输入框

* 模型名称输入框

* MCP 工具接入区域

* MCP URL 和 Token 输入框

保存 API 配置后刷新页面，URL 和模型应保留，Key 只能以脱敏形式显示。

### 5. API Loop 检查

确认服务日志中没有以下问题：

* `RELAY_SECRET missing`

* 无法连接 `127.0.0.1:3020`

* 无法连接第三方模型 API

* `main_chain` 为空

保存 API 配置后，在网页中把联系对象切换为 `API`，发送一条测试消息，确认能收到模型回复。

### 6. MCP 检查

填写 MCP 配置并保存后：

1. 确认 `/data/api_loop.config.json` 已生成或更新。
2. 确认 API Loop 能从 MCP Server 获取 `tools/list`。
3. 发送一条需要使用工具的消息。
4. 确认模型收到工具定义，并且 API Loop 能执行 `tools/call`。
5. MCP 服务不可用时，聊天主流程不能因此导致容器退出。

## 八、部署故障处理

### 页面打开但 API 请求 404

确认：

* Relay 已监听 `$PORT`。

* 访问路径使用 `/relay/...`。

* `backend/app.py` 已启用 `/relay` 前缀兼容。

* 没有把服务部署在错误的子目录。

### 页面能打开但登录失败

确认网页输入的连接密钥与 `RELAY_SECRET` 完全一致，并检查 Zeabur 环境变量是否包含不可见空格。

### API 配置保存失败

确认：

* API Loop 已启动。

* `LOOP_CONFIG=/data/api_loop.config.json`。

* Volume 已挂载到 `/data`。

* Relay 能访问 `http://127.0.0.1:3020/loop/config`。

### 发送消息后没有 AI 回复

确认：

* 前端已切换到 `API`，不是 `Desktop`。

* `LLM_API_BASE` 没有填写 `/chat/completions`。

* `LLM_API_KEY` 有效。

* `LLM_MODEL` 是该服务真实存在的模型名称。

* 第三方 API 支持 OpenAI Chat Completions 格式。

* 查看 API Loop 的 HTTP 状态码和响应错误，但不要打印 API Key。

### 重启后数据消失

说明 `/data` Volume 没有挂载成功，或挂载到了错误路径；必须修正后重新部署。

## 九、安全要求

* 不要在聊天回复、构建日志或部署日志中打印 `RELAY_SECRET`、`LLM_API_KEY` 或 MCP Token。

* 不要把真实密钥写入 Git 文件、Dockerfile、前端代码或 `.env.example`。

* 不要把 API Key 返回给前端明文，只允许返回脱敏值。

* 不要开放数据库、配置文件或上传目录的下载接口。

* 如果用户没有提供密钥，不要替用户编造真实密钥；请要求用户在 Zeabur 控制台自行填写。

* 部署完成后，把公网域名、健康检查结果和下一步需要用户填写的配置项告诉用户。

## 十、最终报告格式

部署完成后，只报告以下内容：

1. Zeabur 服务是否构建成功。
2. 公网访问地址。
3. `/healthz` 是否返回 `ok: true`。
4. 是否确认 `/data` Volume 已挂载。
5. 用户还需要填写的第三方 API URL、Key、模型和 MCP 配置。
6. 如果失败，说明具体失败步骤和日志中的非敏感错误，不要输出任何密钥。

