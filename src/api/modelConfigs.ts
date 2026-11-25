import { jsonRequest, post, request } from "./salespilot";

export type ModelProviderId = "deepseek" | "kimi" | "glm" | "openai" | "minimax" | "ollama" | "custom";
export type ModelCapability = "chat" | "embedding" | "rerank" | "parser";

export interface ModelProvider {
  provider: ModelProviderId;
  label: string;
  base_urls: string[];
  requires_key: boolean;
}

export interface ModelConfigInput {
  name: string;
  capability?: ModelCapability;
  provider: ModelProviderId;
  model_id: string;
  base_url: string | null;
  api_key?: string;
  clear_key?: boolean;
}

export interface ModelConfig extends Omit<ModelConfigInput, "api_key" | "clear_key" | "base_url" | "capability"> {
  config_id: string;
  capability: ModelCapability;
  base_url: string;
  has_key: boolean;
  dimensions?: number | null;
  revision: number;
  test_status: "untested" | "passed" | "failed";
  test_message: string | null;
  test_at: number | null;
  active: boolean;
}

export interface ActiveModel {
  mode: "configured" | "offline" | "unavailable";
  name: string;
  model_id: string | null;
  provider: ModelProviderId | null;
}

const basePath = "/model-configs";
const configPath = (id: string) => basePath + "/" + encodeURIComponent(id);

export const modelConfigs = {
  async list(capability?: ModelCapability): Promise<{ providers: ModelProvider[]; configs: ModelConfig[]; key_storage_ready: boolean }> {
    const query = capability ? "?capability=" + encodeURIComponent(capability) : "";
    return (await request(basePath + query)).json();
  },
  async active(capability: ModelCapability = "chat"): Promise<ActiveModel> {
    return (await request(basePath + "/active?capability=" + encodeURIComponent(capability))).json();
  },
  async save(input: ModelConfigInput, existing: ModelConfig | null): Promise<ModelConfig> {
    const response = existing
      ? await request(configPath(existing.config_id), jsonRequest({ ...input, revision: existing.revision }, "PATCH"))
      : await request(basePath, post(input));
    return response.json();
  },
  async action(config: ModelConfig, action: "test" | "activate" | "deactivate"): Promise<ModelConfig> {
    return (await request(configPath(config.config_id) + "/" + action, post({ revision: config.revision }))).json();
  },
  async remove(config: ModelConfig) {
    await request(configPath(config.config_id), { method: "DELETE" });
  },
};
