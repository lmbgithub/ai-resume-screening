export type Verdict = "covered" | "partial" | "absent" | "unknown";

export interface Evidence {
  text: string;
  section: string;
  specificity: number;
  cosine: number;
}

export interface RequirementResult {
  text: string;
  kind: "required" | "preferred";
  section: string;
  weight: number;
  verdict: Verdict;
  reason: string;
  evidence: Evidence[];
}

/** A strong or weak point, traced back to the requirement behind it. */
export interface Point {
  requirement: string;
  kind: "required" | "preferred";
  verdict: Verdict;
  detail: string;
  evidence: string | null;
}

export interface Suggestions {
  items: string[];
  /** Non-null when the model produced no usable advice — not the same as "none needed". */
  error: string | null;
}

export interface ScreenResult {
  /** null when nothing could be scored — never coerce this to 0. */
  score: number | null;
  percent: string;
  calibrated: boolean;
  scored_weight: number;
  unscored: string[];
  chunk_count: number;
  /** "headings" if parsed from structure, "model" if read out of prose. */
  requirements_source: "headings" | "model";
  requirements: RequirementResult[];
  strengths: Point[];
  weaknesses: Point[];
  suggestions: Suggestions;
}

export interface Extracted {
  text: string;
  format: "text" | "docx" | "pdf";
  confidence: "exact" | "approximate";
  warnings: string[];
  words: number;
}

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (body && typeof body.error === "string") return body.error;
  } catch {
    // fall through to the status line
  }
  return `request failed with HTTP ${response.status}`;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function screen(resume: string, job: string, topK = 3): Promise<ScreenResult> {
  return post<ScreenResult>("/screen", { resume, job, top_k: topK });
}

/** Read a File as base64 without the `data:...;base64,` prefix. */
export function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`could not read ${file.name}`));
    reader.onload = () => {
      const result = String(reader.result);
      const comma = result.indexOf(",");
      resolve(comma === -1 ? result : result.slice(comma + 1));
    };
    reader.readAsDataURL(file);
  });
}

export async function extractCv(file: File): Promise<Extracted> {
  return post<Extracted>("/extract", {
    filename: file.name,
    content_base64: await toBase64(file),
  });
}
