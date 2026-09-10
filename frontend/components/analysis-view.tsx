import { CheckCircle2, Lightbulb, AlertTriangle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { Point, RequirementResult, ScreenResult, Verdict } from "@/lib/api";

const VERDICT_BADGE: Record<
  Verdict,
  { label: string; variant: "success" | "warning" | "destructive" | "outline" }
> = {
  covered: { label: "covered", variant: "success" },
  partial: { label: "partial", variant: "warning" },
  absent: { label: "absent", variant: "destructive" },
  unknown: { label: "unscored", variant: "outline" },
};

function PointRow({ point, tone }: { point: Point; tone: "strong" | "weak" }) {
  const badge = VERDICT_BADGE[point.verdict] ?? VERDICT_BADGE.unknown;
  return (
    <li className="space-y-1 py-3">
      <div className="flex flex-wrap items-start gap-2">
        {tone === "strong" ? (
          <CheckCircle2 className="text-success mt-0.5 size-4 shrink-0" />
        ) : (
          <AlertTriangle className="text-warning mt-0.5 size-4 shrink-0" />
        )}
        <span className="flex-1 text-sm font-medium">{point.requirement}</span>
        <Badge variant="secondary">{point.kind}</Badge>
        {tone === "weak" && <Badge variant={badge.variant}>{badge.label}</Badge>}
      </div>
      {point.detail && <p className="text-muted-foreground pl-6 text-sm">{point.detail}</p>}
      {tone === "strong" && point.evidence && (
        <p className="text-muted-foreground pl-6 text-xs italic">“{point.evidence}”</p>
      )}
    </li>
  );
}

function RequirementRow({ item }: { item: RequirementResult }) {
  const badge = VERDICT_BADGE[item.verdict] ?? VERDICT_BADGE.unknown;
  const top = item.evidence[0];
  return (
    <div className="space-y-2 py-4">
      <div className="flex flex-wrap items-start gap-2">
        <Badge variant={badge.variant}>{badge.label}</Badge>
        <Badge variant="secondary">{item.kind}</Badge>
        <span className="flex-1 text-sm font-medium">{item.text}</span>
      </div>
      {item.reason && <p className="text-muted-foreground pl-1 text-sm">{item.reason}</p>}
      {top && (
        <div className="bg-muted/50 rounded-md px-3 py-2 text-sm">
          <p className="text-muted-foreground text-xs uppercase tracking-wide">
            best evidence · {top.section}
          </p>
          <p className="mt-1">{top.text}</p>
          <p className="text-muted-foreground mt-1 font-mono text-xs">
            specificity {top.specificity >= 0 ? "+" : ""}
            {top.specificity.toFixed(3)} · raw cosine {top.cosine.toFixed(3)}
          </p>
        </div>
      )}
    </div>
  );
}

export function AnalysisView({ result }: { result: ScreenResult }) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex flex-wrap items-baseline gap-3">
            <span className="text-4xl font-semibold tabular-nums">{result.percent}</span>
            <span className="text-muted-foreground text-sm font-normal">
              weighted coverage over {result.requirements.length} requirements ·{" "}
              {result.chunk_count} CV lines
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {result.score === null && (
            <p className="text-destructive">
              No requirement could be scored. This is reported as n/a, not 0% — a parsing failure is
              not evidence about the candidate.
            </p>
          )}
          {!result.calibrated && (
            <p className="text-warning">
              Uncalibrated: with a single requirement or a single CV line there is nothing to centre
              against, so evidence was ranked on raw cosine.
            </p>
          )}
          {result.unscored.length > 0 && (
            <p className="text-warning">
              {result.unscored.length} requirement(s) unscored and excluded from the denominator.
            </p>
          )}
          {result.requirements_source === "model" && (
            <p className="text-warning">
              The job description had no bulleted requirements, so the model read them out of the
              prose. That step is not reproducible the way parsing is — check the requirement list
              below before trusting the score.
            </p>
          )}
          <p className="text-muted-foreground">
            Scores are calibrated within this CV-and-job pair only, and are not comparable across
            candidates.
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Strong points</CardTitle>
          </CardHeader>
          <CardContent className="py-0">
            {result.strengths.length === 0 ? (
              <p className="text-muted-foreground py-3 text-sm">
                No requirement was fully evidenced.
              </p>
            ) : (
              <ul className="divide-border divide-y">
                {result.strengths.map((point, index) => (
                  <PointRow key={index} point={point} tone="strong" />
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Weak points</CardTitle>
          </CardHeader>
          <CardContent className="py-0">
            {result.weaknesses.length === 0 ? (
              <p className="text-muted-foreground py-3 text-sm">Every requirement was evidenced.</p>
            ) : (
              <ul className="divide-border divide-y">
                {result.weaknesses.map((point, index) => (
                  <PointRow key={index} point={point} tone="weak" />
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Lightbulb className="size-4" />
            How to improve your chances
          </CardTitle>
        </CardHeader>
        <CardContent>
          {result.suggestions.items.length > 0 ? (
            <ol className="list-decimal space-y-2 pl-5 text-sm">
              {result.suggestions.items.map((suggestion, index) => (
                <li key={index}>{suggestion}</li>
              ))}
            </ol>
          ) : (
            // Never render an empty list here: it reads as "nothing to improve",
            // which is not what a failed model call means.
            <p className="text-warning text-sm">
              No suggestions were produced
              {result.suggestions.error ? ` — ${result.suggestions.error}` : ""}.
            </p>
          )}
          <Separator className="my-4" />
          <p className="text-muted-foreground text-xs">
            Suggestions are written by the model and do not affect the score. Everything above them
            is derived from the per-requirement verdicts.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Every requirement</CardTitle>
        </CardHeader>
        <CardContent className="divide-border divide-y py-0">
          {result.requirements.map((item, index) => (
            <RequirementRow key={index} item={item} />
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
