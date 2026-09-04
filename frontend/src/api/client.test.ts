import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./client";

function mockFetchOnce(response: { ok: boolean; status: number; statusText: string; body: string }) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: response.ok,
      status: response.status,
      statusText: response.statusText,
      text: () => Promise.resolve(response.body),
      json: () => Promise.resolve(JSON.parse(response.body)),
    })
  );
}

describe("client.ts request()", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves with the parsed JSON body on a successful response", async () => {
    mockFetchOnce({ ok: true, status: 200, statusText: "OK", body: '{"entities":["a","b"]}' });

    const result = await api.listEntities();

    expect(result).toEqual({ entities: ["a", "b"] });
  });

  it("throws an ApiError carrying the backend's detail message on a FastAPI-shaped error body", async () => {
    mockFetchOnce({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      body: '{"detail":"Historical sample data not found. Run \'python src/download_data.py\'."}',
    });

    await expect(api.listEntities()).rejects.toMatchObject({
      status: 503,
      detail: "Historical sample data not found. Run 'python src/download_data.py'.",
    });
  });

  it("the thrown ApiError is an instance of ApiError, distinguishable from a generic Error", async () => {
    mockFetchOnce({ ok: false, status: 503, statusText: "Service Unavailable", body: '{"detail":"unavailable"}' });

    try {
      await api.listEntities();
      expect.fail("expected api.listEntities() to reject");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
    }
  });

  it("falls back to the raw body when the error response isn't the {detail} JSON shape", async () => {
    mockFetchOnce({ ok: false, status: 500, statusText: "Internal Server Error", body: "plain text failure" });

    await expect(api.listEntities()).rejects.toMatchObject({
      status: 500,
      detail: "plain text failure",
    });
  });

  it("falls back to the status line when the error response body is empty", async () => {
    mockFetchOnce({ ok: false, status: 500, statusText: "Internal Server Error", body: "" });

    await expect(api.listEntities()).rejects.toMatchObject({
      status: 500,
      detail: "500 Internal Server Error",
    });
  });
});
