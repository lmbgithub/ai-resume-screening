"use client";

import * as React from "react";
import { FileText, Loader2, Upload, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { extractCv, type Extracted } from "@/lib/api";

const ACCEPT = ".txt,.md,.markdown,.docx,.pdf";

export function CvUpload({
  text,
  onTextChange,
  disabled = false,
}: {
  text: string;
  onTextChange: (text: string) => void;
  disabled?: boolean;
}) {
  const [extracted, setExtracted] = React.useState<Extracted | null>(null);
  const [filename, setFilename] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [dragging, setDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  async function ingest(file: File) {
    setBusy(true);
    setError(null);
    setExtracted(null);
    setFilename(file.name);
    try {
      const result = await extractCv(file);
      setExtracted(result);
      onTextChange(result.text);
    } catch (err) {
      setFilename(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function clear() {
    setExtracted(null);
    setFilename(null);
    setError(null);
    onTextChange("");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload CV"
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file && !disabled) void ingest(file);
        }}
        className={[
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-8 text-center transition-colors",
          dragging ? "border-primary bg-accent" : "border-input",
          disabled ? "pointer-events-none opacity-50" : "hover:bg-accent/50",
        ].join(" ")}
      >
        {busy ? (
          <Loader2 className="text-muted-foreground size-6 animate-spin" />
        ) : filename ? (
          <FileText className="text-muted-foreground size-6" />
        ) : (
          <Upload className="text-muted-foreground size-6" />
        )}
        <p className="text-sm font-medium">
          {busy ? "Extracting…" : filename ? filename : "Drop your CV here, or click to browse"}
        </p>
        <p className="text-muted-foreground text-xs">
          PDF, DOCX, Markdown or plain text · max 5 MB
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          aria-hidden="true"
          tabIndex={-1}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void ingest(file);
          }}
        />
      </div>

      {error && <p className="text-destructive text-sm">{error}</p>}

      {extracted && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted-foreground">
              {extracted.format.toUpperCase()} · {extracted.words} words
            </span>
            {extracted.confidence === "approximate" && (
              <span className="text-warning font-medium">approximate extraction</span>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={clear}
              disabled={disabled}
              className="ml-auto"
            >
              <X />
              Clear
            </Button>
          </div>

          {extracted.warnings.map((warning, index) => (
            <p key={index} className="text-warning text-xs">
              {warning}
            </p>
          ))}

          <Label htmlFor="cv-text">
            Extracted text — edit before analysing if it came out wrong
          </Label>
          <Textarea
            id="cv-text"
            value={text}
            disabled={disabled}
            onChange={(e) => onTextChange(e.target.value)}
            className="h-64 resize-y font-mono text-xs"
          />
        </div>
      )}

      {!extracted && !busy && (
        <div className="space-y-2">
          <Label htmlFor="cv-text">…or paste your CV</Label>
          <Textarea
            id="cv-text"
            value={text}
            disabled={disabled}
            placeholder="Paste plain text or Markdown. Bullets under headings work best."
            onChange={(e) => onTextChange(e.target.value)}
            className="h-40 resize-y font-mono text-xs"
          />
        </div>
      )}
    </div>
  );
}
