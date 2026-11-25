# SalesPilot AI 开发与部署约定

SalesPilot AI 是面向企业销售和售前团队的多品类智能产品选型与 RAG 知识库工作台。本文档定义项目功能边界、代码结构、AI 自动化部署约定及构建校验规范。

## 1. 模块职责与目录结构

| 模块 / 目录 | 职责说明 |
| --- | --- |
| `src/` | Vue 3 + TypeScript + Vite 前端工作台，提供智能产品选型对话、知识库与资料版本管理、产品推荐历史及管理员模型 API 配置入口。 |
| `backend/app/api/` | FastAPI HTTP 接口层（会话认证、知识库与目录、资料版本上传/发布/撤回/删除、检索诊断与索引管理、选型对话与推荐历史、模型 API 配置）。 |
| `backend/app/auth/` | 同源 Cookie 会话鉴权与企业 OIDC（RS256 JWT）认证，按组织隔离资料范围、按账号隔离会话与推荐历史。 |
| `backend/app/contracts/` | 前后端及 RAG 核心共享的 Pydantic 数据契约与协议接口。 |
| `backend/app/infrastructure/` | 数据持久化（SQLite / PostgreSQL）、CAS 原件对象存储（本地 / S3/MinIO）、Redis 分布式锁与异步发布队列、检索遥测及 Fernet 密钥加密存储。 |
| `backend/app/rag/` | 文档解析、分块、向量化（远程兼容接口 / 本地推理 / 离线哈希）、混合检索、多轮选型分流、硬性约束核验与逐条引用校验。 |
| `fixtures/` | 自动化测试专用的合成规格书夹具（`fixtures/specs/`、`fixtures/rag/`），供 `backend/tests/` 读取。 |

## 2. 核心业务边界

- **资料发布与可见性**：仅管理员可创建知识库、上传、发布、撤回或删除资料版本；仅已发布且获授权的版本可被检索或引用。撤回或删除资料后，相关历史推荐记录中的引用内容和原件/报告下载立即失效。
- **可追溯推荐与防伪造**：选型回复中的推荐型号、参数事实与附件手册必须逐条绑定并校验自已授权的检索片段；无充分依据或上游模型不可用时返回明确的待补充或错误状态，禁止生成虚假推荐或空归档。
- **归档隔离**：仅当回复附带手册或推荐报告时才自动归档；普通技术问答、追问澄清或检索诊断测试不会产生推荐历史记录。

## 3. Docker 与 AI 部署约定

- 本仓库面向容器化与 AI 助手自动化部署，保留根目录及后端完整 Docker 编排：
  - `Dockerfile.web`、`backend/Dockerfile`、`backend/entrypoint.sh`
  - `docker-compose.yml`（轻量单机版：`web` + `backend`）
  - `docker-compose.enterprise.yml`（企业全栈版：`web` + `backend` + PostgreSQL/pgvector + Redis + MinIO + Keycloak）
  - `deploy/nginx.conf`、`.env.example`
- **环境与密钥规则**：
  - 部署前从 `.env.example` 复制生成 `.env`，为 `SALESPILOT_MODEL_CONFIG_KEY` 注入随机生成的 32 字节 urlsafe base64 Fernet 主密钥，并替换所有默认演示密码、数据库/MinIO/Keycloak 口令及 `SALESPILOT_WEB_ORIGINS`。
  - 禁止将 `.env`、真实 API 密钥、运行期数据库（`.salespilot/`）、本地密钥文件或真实客户/企业内部资料提交至版本库。

## 4. 开发与构建校验规范

- 保留 `.openai/hosting.json`、`worker/index.js`、`scripts/prepare-sites-build.mjs` 和 `tests/sites-worker.test.mjs`。
- 前端或静态托管相关改动后，执行构建与校验并确认生成 `dist/client/index.html`、`dist/server/index.js` 和 `dist/.openai/hosting.json`：
  ```bash
  npm run build
  npm run test:sites
  ```
- 后端代码或测试夹具改动后，在后端测试环境执行回归测试：
  ```bash
  pytest backend/tests -q
  ```
