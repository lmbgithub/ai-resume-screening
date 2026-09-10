"use client";

import * as React from "react";
import { Loader2, Sparkles } from "lucide-react";

import { AnalysisView } from "@/components/analysis-view";
import { CvUpload } from "@/components/cv-upload";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { screen, type ScreenResult } from "@/lib/api";

export function AnalyseForm() {
  const [job, setJob] = React.useState("");
  const [cv, setCv] = React.useState("");
  const [result, setResult] = React.useState<ScreenResult | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [running, setRunning] = React.useState(false);

  const canRun = job.trim().length > 0 && cv.trim().length > 0 && !running;

  async function onAnalyse() {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      setResult(await screen(cv, job));
    } catch (err) {
      // The API distinguishes a bad request from an unreachable model; both
      // arrive here as a message, and neither is rendered as a score.
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              <Label htmlFor="job">Job description</Label>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              id="job"
              value={job}
              disabled={running}
              onChange={(e) => setJob(e.target.value)}
              placeholder={
                "## Requirements\n- one requirement per bullet\n\n## Preferred qualifications\n- ..."
              }
              className="h-96 resize-y font-mono text-xs"
            />
            <p className="text-muted-foreground mt-2 text-xs">
              Requirements must be bullets under a heading such as “Requirements”. Prose is ignored,
              so the “About us” paragraph never becomes a requirement.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Your CV</CardTitle>
          </CardHeader>
          <CardContent>
            <CvUpload text={cv} onTextChange={setCv} disabled={running} />
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={() => void onAnalyse()} disabled={!canRun} size="lg">
          {running ? <Loader2 className="animate-spin" /> : <Sparkles />}
          {running ? "Analysing…" : "Analyse"}
        </Button>
        <span className="text-muted-foreground text-sm">
          Runs entirely on your machine. The first analysis also pays for the model loading into
          memory.
        </span>
      </div>

      {error && (
        <Card className="border-destructive/50">
          <CardContent>
            <p className="text-destructive text-sm font-medium">{error}</p>
          </CardContent>
        </Card>
      )}

      {running && (
        <Card>
          <CardContent className="space-y-3 py-2">
            <Skeleton className="h-10 w-48" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-2/3" />
          </CardContent>
        </Card>
      )}

      {result && <AnalysisView result={result} />}
    </div>
  );
}
