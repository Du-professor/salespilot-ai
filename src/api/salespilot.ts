export interface SessionUser {
  org_id: string;
  user_id: string;
  username: string;
  can_manage_workspace: boolean;
}
export interface UploadedFile { filename: string; content_base64: string }

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch("/api/v1" + path, { ...init, credentials: "same-origin" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(response.status, body?.code ?? "request_failed", body?.message ?? "请求失败，请稍后重试。");
  }
  return response;
}

export function jsonRequest(body: unknown, method: "POST" | "PATCH" = "POST"): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export const post = (body: unknown) => jsonRequest(body);

export const salespilot = {
  async login(username: string, password: string): Promise<SessionUser> {
    return (await request("/session", post({ username, password }))).json();
  },
  async session(): Promise<SessionUser> {
    return (await request("/session")).json();
  },
  async logout(): Promise<void> {
    await request("/session", { method: "DELETE" });
  },
};
