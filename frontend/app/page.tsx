import { AnalyseForm } from "@/components/analyse-form";

export default function Home() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-8 space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">CV screen</h1>
        <p className="text-muted-foreground max-w-3xl text-sm">
          Paste a job description, upload your CV, and get a score with the requirements you
          evidence, the ones you do not, and what to change. Each requirement is judged separately
          by a local model against evidence that retrieval selected — the arithmetic happens in
          Python, not in the model.
        </p>
      </header>
      <AnalyseForm />
      <footer className="text-muted-foreground mt-12 text-xs">
        No login and no accounts. Nothing is stored: your CV lives in memory for the length of the
        request. Not suitable for making real hiring decisions.
      </footer>
    </main>
  );
}
