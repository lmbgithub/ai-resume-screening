import type { Point, RequirementResult, ScreenResult } from "@/lib/api";

export function makeRequirement(overrides: Partial<RequirementResult> = {}): RequirementResult {
  return {
    text: "5+ years of production Python engineering",
    kind: "required",
    section: "Requirements",
    weight: 1,
    verdict: "covered",
    reason: "seven years of Python services",
    evidence: [
      {
        text: "Backend and ML engineer, seven years building Python services in production.",
        section: "Summary",
        specificity: 0.2468,
        cosine: 0.8213,
      },
    ],
    ...overrides,
  };
}

export function makePoint(overrides: Partial<Point> = {}): Point {
  return {
    requirement: "5+ years of production Python engineering",
    kind: "required",
    verdict: "covered",
    detail: "seven years of Python services",
    evidence: "Backend and ML engineer, seven years building Python services.",
    ...overrides,
  };
}

export function makeResult(overrides: Partial<ScreenResult> = {}): ScreenResult {
  return {
    score: 0.769,
    percent: "76.9%",
    calibrated: true,
    scored_weight: 6.5,
    unscored: [],
    chunk_count: 9,
    requirements_source: "headings" as const,
    requirements: [makeRequirement()],
    strengths: [makePoint()],
    weaknesses: [
      makePoint({
        requirement: "Kubernetes and container orchestration in production",
        verdict: "absent",
        detail: "no orchestration experience shown",
        evidence: null,
      }),
    ],
    suggestions: { items: ["State the request volume your service handled"], error: null },
    ...overrides,
  };
}

export function mockFetchOnce(body: unknown, { ok = true, status = 200 } = {}) {
  const fetchMock = jest.fn().mockResolvedValue({ ok, status, json: async () => body });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

export const EXTRACTED = {
  text: "## Experience\n- built a streaming ASR gateway in Python",
  format: "pdf" as const,
  confidence: "approximate" as const,
  warnings: ["PDF text extraction is approximate: reading order is not recovered."],
  words: 48,
};
