# SalesPilot AI

SalesPilot AI 是面向企业销售与售前团队的多品类智能产品选型与 RAG 知识库工作台。团队上传产品规格书或技术手册后，销售可直接用客户原话发起选型对话，获取基于手册硬性指标比对的型号推荐、可追溯原文引用以及可下载的原件手册与 Markdown 推荐报告。

## 核心功能

- **多知识库与资料版本管理**：支持创建知识库与多级文件夹、选择本地或远程向量模型、批量上传（PDF / DOCX / Markdown / TXT）、版本发布与删除、原件与索引片段查看、检索诊断测试，以及后台索引重建与版本切换/回滚。
- **对话式智能产品选型**：支持自然语言需求输入与多轮追问澄清，基于已授权且已发布的资料执行检索与硬性参数核验，返回推荐型号、理由、待确认项、逐条出处引用及手册/报告附件；支持按日期分组、搜索、重命名、置顶与删除会话。
- **产品推荐历史自动归档**：当回复附带手册或推荐报告时自动归档，支持刷新与重启后追溯；当作为依据的资料版本被撤回或删除时，自动复核并失效对应引用与附件下载权限。
- **统一模型 API 配置中心**：管理员可配置、加密保存并探测验证 `chat`、`embedding`、`rerank` 与外部 `parser`（OCR/版面解析）服务，支持 DeepSeek、Kimi、智谱 GLM、OpenAI、MiniMax、Ollama 及自定义 OpenAI 兼容端点，并支持实时启用与切换。
- **企业级权限与多部署形态**：
  - **轻量单机模式**：FastAPI 同源 Cookie 会话 + SQLite + 本地 CAS 对象存储，开箱即用。
  - **企业全栈模式**：PostgreSQL 16 + `pgvector` + `pg_trgm` 混合检索、MinIO/S3 对象存储、Redis 分布式锁与发布队列、Keycloak OIDC 认证及知识库级 ACL。

## 技术栈

- **前端**：Vue 3、TypeScript、Vite（`src/`）
- **后端**：Python 3.12、FastAPI、Pydantic 2、LangGraph、SQLite / PostgreSQL + pgvector（`backend/`）
- **容器与反向代理**：Docker Compose、Nginx（`deploy/nginx.conf`）

## Docker 部署（推荐直接交由 AI 助手或使用 Docker Compose）

仓库包含完整的容器化部署文件，无需在本机手工搭建前后端运行环境：

- `Dockerfile.web`：前端多阶段构建与 Nginx 静态托管镜像
- `backend/Dockerfile` & `backend/entrypoint.sh`：FastAPI + RAG 后端镜像与启动入口
- `docker-compose.yml`：轻量单机版编排（`web` + `backend` + SQLite 持久化卷）
- `docker-compose.enterprise.yml`：企业全栈编排（`web` + `backend` + PostgreSQL/pgvector + Redis + MinIO + Keycloak）
- `deploy/nginx.conf`：Nginx 反向代理与 200 MB 无缓冲流式上传配置
- `.env.example`：环境变量配置模板

### 部署步骤

1. 从模板生成 `.env`，生成随机 32 字节 Fernet 主密钥填入 `SALESPILOT_MODEL_CONFIG_KEY`，并修改默认演示/数据库密码与 `SALESPILOT_WEB_ORIGINS`：

   ```bash
   cp .env.example .env
   python3 -c "import os, base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
   ```

2. 按部署规模启动容器：

   - **轻量单机版**（`web` + `backend`）：

     ```bash
     docker compose -f docker-compose.yml up -d --build
     ```

   - **企业全栈版**（`web` + `backend` + PostgreSQL/pgvector + Redis + MinIO + Keycloak）：

     ```bash
     docker compose -f docker-compose.enterprise.yml up -d --build
     ```

3. 启动后访问 `http://127.0.0.1:4173/`（可通过管理员账号登录「模型 API 配置」页面添加并测试外部对话、向量、重排或解析端点）。

### 关键环境变量与模型支持

| 变量 | 说明 |
| --- | --- |
| `SALESPILOT_MODEL_CONFIG_KEY` | 必填，32 字节 urlsafe base64 Fernet 主密钥，用于服务端加密存储模型 API 密钥 |
| `SALESPILOT_MODEL_ALLOWED_BASE_URLS` | 允许管理员填写的私有/自定义模型与解析网关地址白名单（逗号分隔） |
| `SALESPILOT_WEB_ORIGINS` | 允许访问后端的前端来源列表（逗号分隔，生产环境请填真实域名） |
| `SALESPILOT_RAG_MODE` | 后端检索运行模式（默认 `demo` 离线哈希模式，生产可结合模型配置中心使用外部接口） |
| `SALESPILOT_DEMO_PASSWORD_A` / `_B` | 轻量演示模式初始化账号 `sales-a` / `sales-b` 密码（12–256 位） |
| `POSTGRES_*` / `MINIO_ROOT_*` / `KC_ADMIN_*` / `SALESPILOT_OIDC_*` | 企业全栈版 PostgreSQL、MinIO 与 Keycloak OIDC 配置 |

支持的模型供应商预设包括 **DeepSeek**、**Kimi**、**智谱 GLM**、**OpenAI**、**MiniMax**、**Ollama** 以及任意 **OpenAI 兼容接口**。

## 安全与隐私说明

- 仓库中不包含任何生产密钥、真实客户资料、企业内部信息或运行期数据库文件。
- 模型 API 密钥在服务端使用 `SALESPILOT_MODEL_CONFIG_KEY` 加密落盘，绝不回传浏览器；`fixtures/` 下仅保留自动化测试使用的合成规格数据。
- 生产环境上线前，请务必替换 `.env` 中的默认账号密码、中间件口令及 `SALESPILOT_WEB_ORIGINS`。

## 开发约定

AI 助手或开发者修改代码、验证构建与打包产物时，请遵循 [agent.md](agent.md)。
