import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, getHealth, getImports, predictGame, trainModel } from "./api";

// Each test mocks `fetch` globally and restores afterward.
const originalFetch = globalThis.fetch;

function mockFetchOnce(status: number, body: unknown, ok = status < 400) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(body),
  } as Response);
}

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("ApiError", () => {
  it("formats its message and preserves status + detail", () => {
    const e = new ApiError(404, "not found");
    expect(e.status).toBe(404);
    expect(e.detail).toBe("not found");
    expect(e.message).toBe("API Error 404: not found");
    expect(e).toBeInstanceOf(Error);
  });
});

describe("api request helpers", () => {
  it("getHealth returns the parsed body on 200", async () => {
    mockFetchOnce(200, { status: "ok", timestamp: "2026-05-29T00:00:00Z" });
    await expect(getHealth()).resolves.toEqual({
      status: "ok",
      timestamp: "2026-05-29T00:00:00Z",
    });
  });

  it("getImports throws ApiError with parsed string detail on non-2xx", async () => {
    mockFetchOnce(500, { detail: "boom" }, false);
    try {
      await getImports();
      throw new Error("expected ApiError");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).status).toBe(500);
      expect((e as ApiError).detail).toBe("boom");
    }
  });

  it("flattens object-shaped error details (FastAPI validation payload)", async () => {
    mockFetchOnce(422, { detail: { message: "File failed validation", errors: ["x"] } }, false);
    try {
      await getImports();
      throw new Error("expected ApiError");
    } catch (e) {
      expect((e as ApiError).status).toBe(422);
      expect((e as ApiError).detail).toBe("File failed validation");
    }
  });

  it("falls back to statusText when the error body has no detail field", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      statusText: "Unauthorized",
      json: () => Promise.reject(new Error("not json")),
    } as Response);
    try {
      await getImports();
      throw new Error("expected ApiError");
    } catch (e) {
      expect((e as ApiError).status).toBe(401);
      expect((e as ApiError).detail).toBe("Unauthorized");
    }
  });

  it("trainModel POSTs JSON with the right method + body", async () => {
    mockFetchOnce(200, {
      model_id: 7,
      model_type: "random_forest",
      metrics: { accuracy: 0.64, precision_score: 0.64, recall: 0.64, f1_score: 0.63, confidence: 0.65, confidence_reliable: true },
    });
    await trainModel({ import_id: 2, model_type: "random_forest", params: { n_estimators: 100 } });

    const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[0]).toMatch(/\/api\/train$/);
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body)).toEqual({
      import_id: 2,
      model_type: "random_forest",
      params: { n_estimators: 100 },
    });
  });

  it("predictGame surfaces 404 as ApiError (no trained model)", async () => {
    mockFetchOnce(404, { detail: "No trained model found for type 'random_forest'" }, false);
    try {
      await predictGame({ home_team: "1610612747", away_team: "1610612738", season: "2019-20", model_type: "random_forest" });
      throw new Error("expected ApiError");
    } catch (e) {
      expect((e as ApiError).status).toBe(404);
      expect((e as ApiError).detail).toContain("No trained model");
    }
  });
});
