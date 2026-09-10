import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AnalyseForm } from "@/components/analyse-form";
import { EXTRACTED, makeResult } from "./fixtures";

const JOB = "## Requirements\n- build streaming inference services";

/** One fake for both routes, so a test states only the outcome it cares about. */
function api({ result = makeResult() as unknown, status = 200, extracted = EXTRACTED } = {}) {
  const fetchMock = jest.fn(async (url: string, init?: RequestInit) => {
    void init;
    if (String(url).endsWith("/extract")) {
      return { ok: true, status: 200, json: async () => extracted };
    }
    return { ok: status < 400, status, json: async () => result };
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

async function fill(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Job description"), JOB);
  await user.type(screen.getByLabelText(/paste your CV/i), "## Experience\n- built things");
}

describe("the analyse button", () => {
  it("is disabled until both the job and the CV have content", async () => {
    const user = userEvent.setup();
    api();
    render(<AnalyseForm />);

    const button = screen.getByRole("button", { name: /^analyse$/i });
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText("Job description"), JOB);
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText(/paste your CV/i), "## Experience\n- built things");
    expect(button).toBeEnabled();
  });

  it("stays disabled for whitespace-only input", async () => {
    const user = userEvent.setup();
    api();
    render(<AnalyseForm />);
    await user.type(screen.getByLabelText("Job description"), "   ");
    await user.type(screen.getByLabelText(/paste your CV/i), "   ");
    expect(screen.getByRole("button", { name: /^analyse$/i })).toBeDisabled();
  });

  it("becomes enabled after a CV is uploaded", async () => {
    const user = userEvent.setup();
    api();
    render(<AnalyseForm />);
    await user.type(screen.getByLabelText("Job description"), JOB);

    const input = screen.getByLabelText("Upload CV").querySelector("input")!;
    await user.upload(input, new File(["x"], "cv.pdf", { type: "application/pdf" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /^analyse$/i })).toBeEnabled());
  });
});

describe("a successful analysis", () => {
  it("sends the CV as 'resume' and the job as 'job'", async () => {
    const user = userEvent.setup();
    const fetchMock = api();
    render(<AnalyseForm />);
    await fill(user);
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));

    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([u]) => String(u).endsWith("/screen"))).toBe(true)
    );
    const call = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/screen"))!;
    const body = JSON.parse(call[1]!.body as string);
    expect(body).toEqual({ resume: "## Experience\n- built things", job: JOB, top_k: 3 });
  });

  it("renders the score, points and suggestions", async () => {
    const user = userEvent.setup();
    api({ result: makeResult({ percent: "76.9%" }) });
    render(<AnalyseForm />);
    await fill(user);
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByText("76.9%")).toBeInTheDocument();
    expect(screen.getByText("Strong points")).toBeInTheDocument();
    expect(screen.getByText("Weak points")).toBeInTheDocument();
    expect(screen.getByText(/How to improve your chances/)).toBeInTheDocument();
    expect(screen.getByText("State the request volume your service handled")).toBeInTheDocument();
  });

  it("renders an unscorable run as n/a", async () => {
    const user = userEvent.setup();
    api({ result: makeResult({ score: null, percent: "n/a", unscored: ["a", "b"] }) });
    render(<AnalyseForm />);
    await fill(user);
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByText("n/a")).toBeInTheDocument();
    expect(screen.getByText(/2 requirement\(s\) unscored/)).toBeInTheDocument();
  });
});

describe("failures", () => {
  it("shows the API's message instead of a score", async () => {
    const user = userEvent.setup();
    api({ result: { error: "the job description produced no requirements" }, status: 422 });
    render(<AnalyseForm />);
    await fill(user);
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByText(/produced no requirements/)).toBeInTheDocument();
    expect(screen.queryByText(/weighted coverage/)).not.toBeInTheDocument();
  });

  it("clears a previous error on the next run", async () => {
    const user = userEvent.setup();
    api({ result: { error: "first failure" }, status: 503 });
    render(<AnalyseForm />);
    await fill(user);
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));
    expect(await screen.findByText("first failure")).toBeInTheDocument();

    api({ result: makeResult() });
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));
    await waitFor(() => expect(screen.queryByText("first failure")).not.toBeInTheDocument());
  });

  it("reports a thrown network error as text", async () => {
    const user = userEvent.setup();
    api();
    render(<AnalyseForm />);
    await fill(user);
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));
    expect(await screen.findByText("Failed to fetch")).toBeInTheDocument();
  });
});

describe("while analysing", () => {
  it("locks the inputs and shows progress", async () => {
    const user = userEvent.setup();
    api();
    render(<AnalyseForm />);
    await fill(user);

    let release: (value: unknown) => void = () => {};
    global.fetch = jest.fn(
      () =>
        new Promise((resolve) => {
          release = resolve;
        })
    ) as unknown as typeof fetch;
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));

    const running = await screen.findByRole("button", { name: /analysing/i });
    expect(running).toBeDisabled();
    expect(screen.getByLabelText("Job description")).toBeDisabled();

    release({ ok: true, status: 200, json: async () => makeResult() });
    await waitFor(() => expect(screen.getByRole("button", { name: /^analyse$/i })).toBeEnabled());
  });

  it("clears a previous result while the next run is in flight", async () => {
    const user = userEvent.setup();
    api({ result: makeResult({ percent: "76.9%" }) });
    render(<AnalyseForm />);
    await fill(user);
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));
    expect(await screen.findByText("76.9%")).toBeInTheDocument();

    let release: (value: unknown) => void = () => {};
    global.fetch = jest.fn(
      () =>
        new Promise((resolve) => {
          release = resolve;
        })
    ) as unknown as typeof fetch;
    await user.click(screen.getByRole("button", { name: /^analyse$/i }));
    await waitFor(() => expect(screen.queryByText("76.9%")).not.toBeInTheDocument());
    release({ ok: true, status: 200, json: async () => makeResult() });
  });
});
