import { API_URL, extractCv, screen, toBase64 } from "@/lib/api";
import { EXTRACTED, makeResult, mockFetchOnce } from "./fixtures";

function file(name: string, content = "## Experience\n- built things") {
  return new File([content], name, { type: "text/plain" });
}

describe("screen()", () => {
  it("posts the cv, job and top_k as JSON", async () => {
    const fetchMock = mockFetchOnce(makeResult());
    await screen("cv text", "job text");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_URL}/screen`);
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ resume: "cv text", job: "job text", top_k: 3 });
  });

  it("returns the parsed result", async () => {
    mockFetchOnce(makeResult({ percent: "42.0%" }));
    await expect(screen("c", "j")).resolves.toMatchObject({ percent: "42.0%" });
  });

  it("preserves a null score rather than coercing it to zero", async () => {
    mockFetchOnce(makeResult({ score: null, percent: "n/a" }));
    expect((await screen("c", "j")).score).toBeNull();
  });

  it("surfaces the API's own error message", async () => {
    mockFetchOnce({ error: "cannot reach Ollama" }, { ok: false, status: 503 });
    await expect(screen("c", "j")).rejects.toThrow("cannot reach Ollama");
  });

  it("falls back to the status code when the error body has no message", async () => {
    mockFetchOnce({ nope: true }, { ok: false, status: 500 });
    await expect(screen("c", "j")).rejects.toThrow("HTTP 500");
  });

  it("falls back to the status code when the error body is not JSON", async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new SyntaxError("Unexpected token <");
      },
    }) as unknown as typeof fetch;
    await expect(screen("c", "j")).rejects.toThrow("HTTP 502");
  });

  it("ignores a non-string error field", async () => {
    mockFetchOnce({ error: { code: 7 } }, { ok: false, status: 400 });
    await expect(screen("c", "j")).rejects.toThrow("HTTP 400");
  });

  it("propagates a network failure", async () => {
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;
    await expect(screen("c", "j")).rejects.toThrow("Failed to fetch");
  });
});

describe("toBase64()", () => {
  it("strips the data-URL prefix", async () => {
    const encoded = await toBase64(file("cv.txt", "hello"));
    expect(encoded).toBe(Buffer.from("hello").toString("base64"));
  });

  it("round trips binary-ish content", async () => {
    const encoded = await toBase64(file("cv.txt", "café ☕"));
    expect(Buffer.from(encoded, "base64").toString("utf-8")).toBe("café ☕");
  });
});

describe("extractCv()", () => {
  it("posts the filename and base64 content", async () => {
    const fetchMock = mockFetchOnce(EXTRACTED);
    await extractCv(file("cv.pdf", "hello"));

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_URL}/extract`);
    const body = JSON.parse(init.body);
    expect(body.filename).toBe("cv.pdf");
    expect(Buffer.from(body.content_base64, "base64").toString()).toBe("hello");
  });

  it("returns the extraction, warnings included", async () => {
    mockFetchOnce(EXTRACTED);
    const result = await extractCv(file("cv.pdf"));
    expect(result.confidence).toBe("approximate");
    expect(result.warnings).toHaveLength(1);
  });

  it("surfaces an unusable-file error", async () => {
    mockFetchOnce({ error: "no text could be extracted" }, { ok: false, status: 422 });
    await expect(extractCv(file("cv.pdf"))).rejects.toThrow("no text could be extracted");
  });
});

describe("API_URL", () => {
  it("has no trailing slash, so route joins never double up", () => {
    expect(API_URL).not.toMatch(/\/$/);
  });
});
