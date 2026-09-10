import { render, screen } from "@testing-library/react";

import { AnalysisView } from "@/components/analysis-view";
import { makePoint, makeRequirement, makeResult } from "./fixtures";

describe("score header", () => {
  it("shows the percentage the API computed", () => {
    render(<AnalysisView result={makeResult({ percent: "76.9%" })} />);
    expect(screen.getByText("76.9%")).toBeInTheDocument();
  });

  it("renders n/a and never 0% when nothing could be scored", () => {
    render(<AnalysisView result={makeResult({ score: null, percent: "n/a" })} />);
    expect(screen.getByText("n/a")).toBeInTheDocument();
    expect(screen.queryByText("0.0%")).not.toBeInTheDocument();
  });

  it("explains that an unscorable run is not evidence about the candidate", () => {
    render(<AnalysisView result={makeResult({ score: null, percent: "n/a" })} />);
    expect(screen.getByText(/not 0%/)).toBeInTheDocument();
  });

  it("warns when the run was uncalibrated", () => {
    render(<AnalysisView result={makeResult({ calibrated: false })} />);
    expect(screen.getByText(/Uncalibrated/)).toBeInTheDocument();
  });

  it("shows no calibration warning on a normal run", () => {
    render(<AnalysisView result={makeResult()} />);
    expect(screen.queryByText(/Uncalibrated/)).not.toBeInTheDocument();
  });

  it("counts unscored requirements", () => {
    render(<AnalysisView result={makeResult({ unscored: ["a", "b"] })} />);
    expect(screen.getByText(/2 requirement\(s\) unscored/)).toBeInTheDocument();
  });

  it("always says the score is not comparable across candidates", () => {
    render(<AnalysisView result={makeResult()} />);
    expect(screen.getByText(/not comparable across candidates/)).toBeInTheDocument();
  });
});

describe("strong points", () => {
  it("lists them with their evidence quoted", () => {
    render(<AnalysisView result={makeResult()} />);
    expect(screen.getByText("Strong points")).toBeInTheDocument();
    // The same sentence also appears as evidence in the per-requirement list.
    expect(screen.getAllByText(/seven years building Python services/).length).toBeGreaterThan(0);
  });

  it("says so when there are none", () => {
    render(<AnalysisView result={makeResult({ strengths: [] })} />);
    expect(screen.getByText(/No requirement was fully evidenced/)).toBeInTheDocument();
  });

  it("shows the requirement kind", () => {
    render(<AnalysisView result={makeResult({ strengths: [makePoint({ kind: "preferred" })] })} />);
    expect(screen.getAllByText("preferred").length).toBeGreaterThan(0);
  });
});

describe("weak points", () => {
  it("lists them with the verdict that produced them", () => {
    render(<AnalysisView result={makeResult()} />);
    expect(
      screen.getByText("Kubernetes and container orchestration in production")
    ).toBeInTheDocument();
    expect(screen.getAllByText("absent").length).toBeGreaterThan(0);
  });

  it("shows the model's reason", () => {
    render(<AnalysisView result={makeResult()} />);
    expect(screen.getByText("no orchestration experience shown")).toBeInTheDocument();
  });

  it("says so when there are none", () => {
    render(<AnalysisView result={makeResult({ weaknesses: [] })} />);
    expect(screen.getByText(/Every requirement was evidenced/)).toBeInTheDocument();
  });

  it("renders a point with no evidence without crashing", () => {
    const point = makePoint({ verdict: "partial", evidence: null, detail: "" });
    render(<AnalysisView result={makeResult({ weaknesses: [point] })} />);
    expect(screen.getAllByText("partial").length).toBeGreaterThan(0);
  });
});

describe("suggestions", () => {
  it("lists them in order", () => {
    const suggestions = { items: ["first advice", "second advice"], error: null };
    render(<AnalysisView result={makeResult({ suggestions })} />);
    expect(screen.getByText("first advice")).toBeInTheDocument();
    expect(screen.getByText("second advice")).toBeInTheDocument();
  });

  it("says they are model-written and do not affect the score", () => {
    render(<AnalysisView result={makeResult()} />);
    expect(
      screen.getByText(/written by the model and do not affect the score/)
    ).toBeInTheDocument();
  });

  it("reports the reason when none were produced", () => {
    // An empty list under this heading would read as "nothing to improve".
    const suggestions = { items: [], error: "model call failed: connection refused" };
    render(<AnalysisView result={makeResult({ suggestions })} />);
    expect(screen.getByText(/No suggestions were produced/)).toBeInTheDocument();
    expect(screen.getByText(/connection refused/)).toBeInTheDocument();
  });

  it("does not claim a reason it was not given", () => {
    const suggestions = { items: [], error: null };
    render(<AnalysisView result={makeResult({ suggestions })} />);
    expect(screen.getByText(/No suggestions were produced\./)).toBeInTheDocument();
  });
});

describe("every requirement", () => {
  it("keeps the raw cosine next to the calibrated specificity", () => {
    render(<AnalysisView result={makeResult()} />);
    expect(screen.getByText(/specificity \+0\.247 · raw cosine 0\.821/)).toBeInTheDocument();
  });

  it("keeps the sign on a negative specificity", () => {
    const item = makeRequirement({
      evidence: [{ text: "a line", section: "Skills", specificity: -0.05, cosine: 0.4 }],
    });
    render(<AnalysisView result={makeResult({ requirements: [item] })} />);
    expect(screen.getByText(/specificity -0\.050/)).toBeInTheDocument();
  });

  it("labels an unknown verdict as unscored", () => {
    const item = makeRequirement({ verdict: "unknown" });
    render(<AnalysisView result={makeResult({ requirements: [item], weaknesses: [] })} />);
    expect(screen.getByText("unscored")).toBeInTheDocument();
  });

  it("falls back to the unscored label for an unrecognised verdict", () => {
    const item = makeRequirement({ verdict: "brilliant" as never });
    render(<AnalysisView result={makeResult({ requirements: [item], weaknesses: [] })} />);
    expect(screen.getByText("unscored")).toBeInTheDocument();
  });

  it("renders a requirement with no evidence", () => {
    const item = makeRequirement({ verdict: "absent", reason: "nothing found", evidence: [] });
    render(<AnalysisView result={makeResult({ requirements: [item], weaknesses: [] })} />);
    expect(screen.getByText("nothing found")).toBeInTheDocument();
    expect(screen.queryByText(/raw cosine/)).not.toBeInTheDocument();
  });
});

describe("how the requirements were obtained", () => {
  it("says nothing extra when they were parsed from headings", () => {
    render(<AnalysisView result={makeResult({ requirements_source: "headings" })} />);
    expect(screen.queryByText(/read them out of the prose/)).not.toBeInTheDocument();
  });

  it("warns that a model-read requirement list is not reproducible", () => {
    render(<AnalysisView result={makeResult({ requirements_source: "model" })} />);
    expect(screen.getByText(/read them out of the prose/)).toBeInTheDocument();
    expect(screen.getByText(/not reproducible/)).toBeInTheDocument();
  });
});
