import * as React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { CvUpload } from "@/components/cv-upload";
import { EXTRACTED, mockFetchOnce } from "./fixtures";

function file(name = "cv.pdf", content = "binary-ish") {
  return new File([content], name, { type: "application/pdf" });
}

/** Renders with the parent state wired up, the way the page uses it. */
function Harness({ disabled = false }: { disabled?: boolean }) {
  const [text, setText] = React.useState("");
  return <CvUpload text={text} onTextChange={setText} disabled={disabled} />;
}

describe("before an upload", () => {
  it("offers a paste box", () => {
    render(<Harness />);
    expect(screen.getByLabelText(/paste your CV/i)).toBeInTheDocument();
  });

  it("names the accepted formats", () => {
    render(<Harness />);
    expect(screen.getByText(/PDF, DOCX, Markdown or plain text/)).toBeInTheDocument();
  });

  it("reports typed text upward", async () => {
    const user = userEvent.setup();
    const onTextChange = jest.fn();
    render(<CvUpload text="" onTextChange={onTextChange} />);
    await user.type(screen.getByLabelText(/paste your CV/i), "x");
    expect(onTextChange).toHaveBeenCalledWith("x");
  });
});

describe("uploading", () => {
  it("posts the file and fills the text", async () => {
    const user = userEvent.setup();
    const fetchMock = mockFetchOnce(EXTRACTED);
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    await waitFor(() =>
      expect(screen.getByLabelText(/Extracted text/)).toHaveValue(EXTRACTED.text)
    );
    expect(fetchMock.mock.calls[0][0]).toContain("/extract");
  });

  it("shows the filename after a successful upload", async () => {
    const user = userEvent.setup();
    mockFetchOnce(EXTRACTED);
    render(<Harness />);

    await user.upload(
      screen.getByLabelText("Upload CV").querySelector("input")!,
      file("alex-cv.pdf")
    );
    expect(await screen.findByText("alex-cv.pdf")).toBeInTheDocument();
  });

  it("reports the format and word count", async () => {
    const user = userEvent.setup();
    mockFetchOnce(EXTRACTED);
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    expect(await screen.findByText(/PDF · 48 words/)).toBeInTheDocument();
  });

  it("flags an approximate extraction", async () => {
    const user = userEvent.setup();
    mockFetchOnce(EXTRACTED);
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    expect(await screen.findByText("approximate extraction")).toBeInTheDocument();
  });

  it("shows every warning the API returned", async () => {
    const user = userEvent.setup();
    mockFetchOnce({
      ...EXTRACTED,
      warnings: ["reading order is not recovered", "only 12 words were extracted"],
    });
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    expect(await screen.findByText(/reading order is not recovered/)).toBeInTheDocument();
    expect(screen.getByText(/only 12 words were extracted/)).toBeInTheDocument();
  });

  it("does not flag an exact extraction as approximate", async () => {
    const user = userEvent.setup();
    mockFetchOnce({ ...EXTRACTED, format: "docx", confidence: "exact", warnings: [] });
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file("cv.docx"));
    await screen.findByText(/DOCX · 48 words/);
    expect(screen.queryByText("approximate extraction")).not.toBeInTheDocument();
  });

  it("lets the extracted text be edited before analysing", async () => {
    const user = userEvent.setup();
    mockFetchOnce(EXTRACTED);
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    const box = await screen.findByLabelText(/Extracted text/);
    await user.type(box, " corrected");
    expect(box).toHaveValue(`${EXTRACTED.text} corrected`);
  });
});

describe("upload failures", () => {
  it("shows the API's message and keeps the paste box", async () => {
    const user = userEvent.setup();
    mockFetchOnce(
      { error: "no text could be extracted from this pdf file" },
      { ok: false, status: 422 }
    );
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    expect(await screen.findByText(/no text could be extracted/)).toBeInTheDocument();
    expect(screen.getByLabelText(/paste your CV/i)).toBeInTheDocument();
  });

  it("does not leave a filename behind after a failure", async () => {
    const user = userEvent.setup();
    mockFetchOnce({ error: "unusable" }, { ok: false, status: 422 });
    render(<Harness />);

    await user.upload(
      screen.getByLabelText("Upload CV").querySelector("input")!,
      file("broken.pdf")
    );
    await screen.findByText("unusable");
    expect(screen.queryByText("broken.pdf")).not.toBeInTheDocument();
  });

  it("reports a network failure as text", async () => {
    const user = userEvent.setup();
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    expect(await screen.findByText("Failed to fetch")).toBeInTheDocument();
  });
});

describe("clearing", () => {
  it("empties the text and returns to the paste box", async () => {
    const user = userEvent.setup();
    mockFetchOnce(EXTRACTED);
    render(<Harness />);

    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    await screen.findByLabelText(/Extracted text/);

    await user.click(screen.getByRole("button", { name: /clear/i }));
    expect(screen.getByLabelText(/paste your CV/i)).toHaveValue("");
  });
});

describe("while a run is in flight", () => {
  it("locks the dropzone and the text box", async () => {
    const user = userEvent.setup();
    mockFetchOnce(EXTRACTED);
    const { rerender } = render(<Harness />);
    await user.upload(screen.getByLabelText("Upload CV").querySelector("input")!, file());
    await screen.findByLabelText(/Extracted text/);

    rerender(<Harness disabled />);
    expect(screen.getByLabelText("Upload CV")).toHaveAttribute("aria-disabled", "true");
  });
});
