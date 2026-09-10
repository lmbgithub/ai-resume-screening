# resume-screen

Upload a CV, paste a job description, and get a score with the requirements it
evidences, the ones it does not, and what to change — using a **local Ollama
model**. One page, no login, no accounts, nothing stored.

The Python side is **standard library only** (`urllib`, `json`, `math`,
`zipfile`, `zlib`, `xml`) — the API and the PDF/DOCX extraction included, so
its container installs nothing at all. `pytest` and `ruff` are the only dev
dependencies. The web app is Next.js 16, Tailwind v4 and shadcn/ui.

**381 pytest tests and 65 Jest tests**, none of which need a model, a server or
a network. Both suites run in CI on every push, alongside `ruff check`,
`ruff format --check`, `eslint`, `prettier --check` and `tsc --noEmit`.

```bash
./start.sh          # picks the fastest inference backend on this machine
```

```
CV file ──► /extract ──► text + confidence ──► (you check it)
                                                     │
job description ─────────────────────────────────────┤
                                                     ▼
                                                 /screen
                                                     │
   chunks + requirements → embeddings → double-centering → top-k evidence
                                                     │
                              per-requirement verdict (model)
                                                     │
        ┌────────────────────────────┬───────────────┴──────────────┐
        ▼                            ▼                              ▼
  weighted score              strong / weak points            suggestions
  (deterministic)             (derived from verdicts)      (model-written)
```

## A real run

`llama3.2:3b` and `nomic-embed-text` in Docker, against the bundled sample:

```
$ resume-screen --resume data/resume.md --job data/job.md
Match: 69.2%  (weighted coverage over 8 requirements)

Strong points:
  + (required) 5+ years of production Python engineering
      Backend and ML engineer, seven years building Python services in production.
  + (required) Build and maintain streaming inference services with sub-second latency
      Built a streaming ASR gateway in Python handling 400 concurrent WebSocket sessions
  + (preferred) Familiarity with Prometheus and Grafana for service observability
      Instrumented services with Prometheus metrics and built the on-call Grafana dashboards

Weak points (required first):
  - (required/absent) Kubernetes and container orchestration in production
      No mention of Kubernetes or container orchestration
  - (preferred/absent) Experience with quantised local LLM inference
      No mention of LLM inference

Suggestions (written by the model, not scored):
  1. Quantized local LLM inference: if applicable, quantify the latency reduction
     achieved and describe the approach used.
  2. Kubernetes and container orchestration: if not already mentioned, describe your
     experience with containerization tools (e.g. Docker) and orchestration tools.
```

Look at the Kubernetes line. The CV says nothing about Kubernetes, and the
best-matching CV line — about ETL pipelines — still scores **0.553** cosine.
That number is why this project exists.

## The twelve decisions worth discussing

### 1. The LLM never produces the score

The obvious build is "paste the resume and the job description into the model,
ask for 0–100". That number is unauditable, moves between runs, and is
demonstrably swayed by names, employers and schools — attributes the model can
see and a screening tool has no business weighting.

Here the model answers one narrow, checkable question per requirement — _do
these specific lines demonstrate this?_ — with three allowed answers. All
arithmetic happens in `scoring.py`, which never imports a client. The prompt
explicitly forbids inferring from job titles, employers or schools, and the
model only ever sees the lines retrieval already selected, so it cannot read a
name off the top of the resume.

Consequence: you can change the model and diff the verdicts, requirement by
requirement, instead of watching one opaque number move.

### 2. A raw cosine is not a match percentage

`nomic-embed-text` over this resume and job description produces cosines in
**0.346 – 0.821, mean 0.483 across all 72 pairs** — including every unrelated
one. Rescale that to "48% match" and you have published a number whose origin
is an artefact of the embedding model's geometry. Swap the model and every
score moves, with no change in any candidate.

So the raw cosine is reported but never aggregated. What drives retrieval is
the double-centered residual:

```
residual[r][c] = S[r][c] − mean(row r) − mean(column c) + mean(S)
```

Subtracting the row mean removes how verbose or generic the requirement is;
subtracting the column mean removes how generic the resume line is; the grand
mean is added back so the residuals sum to zero. What survives is only the part
specific to _that_ pair. It is the same idea as IDF, done in embedding space.

### 3. Centering changes which evidence the model sees — measurably

Not a theoretical improvement. `examples/centering_effect.py` reruns it:

```
top-3 evidence set changed for 7/8 requirements
top-1 evidence changed for 2/8 requirements
most generic resume line (mean cosine 0.545): Backend and ML engineer, seven years
  building Python services in production.
```

The most generic line is the summary — the one written to resemble everything.
Under raw cosine it is the top hit for _open-source contributions_; under
centering that requirement retrieves the line about authoring a benchmark tool,
which is the actual evidence. Since retrieved evidence is the model's entire
input, this is not a display detail — it changes verdicts.

### 4. An unparseable answer is `unknown`, and `unknown` leaves the denominator

The common shortcut is `json.loads(response)` inside a `try`, with the `except`
scoring the requirement zero. That turns _our_ parser failure into _the
candidate's_ rejection, and it is silent.

`parse_verdict` never raises. Fenced JSON, leading prose and nested braces are
recovered by a string-aware brace scanner; anything else becomes `unknown` with
the reason recorded. `aggregate` then removes `unknown` from **both** the
numerator and the denominator, names the affected requirements in the report,
and the CLI exits **1** so a calling script cannot mistake a partial run for a
clean pass. If nothing parsed, the score is `None` and prints as `n/a` — never
`0.0%`.

One specific trap covered: `isinstance(True, int)` is `True` in Python, and
models do emit `{"verdict": true}`. The check is `isinstance(x, str)`, so a
bare boolean lands on `unknown` rather than being coerced into a pass.

### 5. Degenerate inputs are flagged, not silently scored

With one requirement or one resume line, every residual is exactly zero by
construction — centering has nothing to compare against. Ranking on those zeros
would emit a confident, arbitrary ordering. `is_degenerate` detects it, the
pipeline falls back to raw cosine, and the report prints `uncalibrated`.

### 6. Structure when it exists, the model when it does not

A job description is a list of separately scorable claims, and the heading
above each one says how much it counts: `Preferred qualifications` is checked
before `Requirements` because it contains both words and the qualifier is the
informative half. An unrecognised heading defaults to **required**, since
under-weighting a genuine requirement is the worse error.

That parser is deterministic, free, and exactly reproducible, so it runs first
and wins whenever it finds anything. Prose is dropped, so the "About us"
paragraph never becomes a requirement — in the offline example that one line
alone moved the score from 80.8% to 70.0% while carrying no information about
the candidate.

**But it only ever worked on job descriptions somebody had already tidied up.**
Real postings are prose pasted out of a job board with no Markdown in sight,
and rejecting those with _"it needs headings such as 'Requirements'"_ was a
rule serving the parser rather than the user. So when the rules find nothing,
the model is asked to extract the requirements from the prose, with a prompt
that forbids inventing or generalising them and tells it to ignore benefits,
culture and application instructions.

The result records **which path ran**, and the UI says so, because the two are
not equivalent: a parsed requirement list is byte-identical every run, and a
model-extracted one is not. The score is only as reproducible as the list it
was computed from, and hiding that behind a single number would be the same
mistake as reporting a raw cosine as a percentage.

The honest caveat: a 3B model under-extracts. On a five-paragraph posting it
found 5 of the 6 requirements a reader would list. That is why the extracted
requirements are shown in full rather than summarised away.

### 7. Model choice moves the score more than any tuning does

Same resume, same job description, same code, three runs each:

| Runtime                     | Model         | Score |
| --------------------------- | ------------- | ----- |
| Host Ollama (Metal GPU)     | `gemma4:e2b`  | 76.9% |
| Host Ollama (Metal GPU)     | `llama3.2:3b` | 92.3% |
| Container Ollama (CPU only) | `llama3.2:3b` | 69.2% |

Each figure was stable across three runs on its own runtime. Two things follow.

**Models disagree, and per-requirement verdicts show you where.** On
_Kubernetes and container orchestration_, where the resume says nothing about
Kubernetes, `llama3.2:3b` on the host answered `covered` with the reason
"Docker and Prometheus mentioned". `gemma4:e2b` answered `partial` and said
Kubernetes was not mentioned. That is a bad verdict you can point at, argue
about, and regress-test. A single 0–100 number would have shown a 16-point
gap with nothing to inspect.

**Temperature 0 is reproducibility, not determinism.** The identical model at
temperature 0 scored 92.3% on the host and 69.2% in the CPU-only container.
Each is stable where it runs; neither ports. Anything that pins a threshold to
an absolute score will break the first time it moves to different hardware.

### 8. The API is `http.server`, and its image installs nothing

FastAPI is the reflex. It was rejected: this is three routes with no auth, no
sessions and no schema evolution, and `ThreadingHTTPServer` already serves
concurrent requests. Because nothing in the package leaves the standard
library, the API container is `python:3.12-slim` plus `COPY src/` — no `pip`
step, no lockfile, no wheel cache, and no dependency surface to patch.

Routes are pure functions (`handle_screen(config, body)`) with the socket work
in a thin handler, so every status path — 400, 413, 422, 503 — is tested
without binding a port. The pipeline's distinctions survive the transport:
`score` is JSON `null` rather than `0` when nothing parsed, `calibrated`
travels beside the score, and an unreachable Ollama is a **503** while a
malformed body is a **400**. The browser renders "n/a", never "0.0%".

One trap repeated at the boundary: `{"top_k": true}` would silently mean
`top_k=1`, because `isinstance(True, int)` is `True`. The check excludes `bool`
explicitly.

### 9. Extraction confidence is returned, not hidden

Three upload formats, three genuinely different levels of trust, and the caller
is told which one it got:

| Format         | How                                                          | Confidence    |
| -------------- | ------------------------------------------------------------ | ------------- |
| `.txt` / `.md` | decode                                                       | `exact`       |
| `.docx`        | it is a zip of XML; `w:t` holds runs, `w:p` holds paragraphs | `exact`       |
| `.pdf`         | inflate the content streams, read the show-text operators    | `approximate` |

PDF is approximate and unavoidably so: a PDF stores glyph placement, not
paragraphs, so a two-column CV has no reading order to recover without layout
analysis. The alternative — an extractor that silently interleaves two columns
into one line — poisons every downstream score _while looking like it worked_.

**The extractor is measured, not asserted.** macOS ships PDFKit, so there is a
ground truth to check against: a 40-line Swift program prints `PDFDocument.string`,
and the suite of real PDFs on this machine becomes a corpus. Across the **120
text-bearing PDFs** found under `/System` and `/Applications`:

|                                               | first version | now     |
| --------------------------------------------- | ------------- | ------- |
| hard failures                                 | 0             | **0**   |
| under-extracted (<50% of PDFKit's word count) | 6             | **1**   |
| within range                                  | 114           | **119** |

The six failures were all one bug: text set in hex strings (`<0044…> Tj`) rather
than literal strings, which is what any PDF with a subset or CID font uses.
Reading them needs the document's ToUnicode CMaps, so those are parsed
(`beginbfchar` and `beginbfrange`) and merged. A Chinese-language PDF went from
6 words to 3322, against PDFKit's 1888.

Two more real-world cases the first version got wrong:

- **An encrypted PDF reported "it contains images, not text."** Wrong diagnosis,
  and it sends people hunting for a scanner problem they do not have. `/Encrypt`
  in the trailer is now detected up front and the message says to re-export
  without a password.
- **Streams missing a zlib header or truncated** were dropped whole.
  `decompressobj` with a raw-deflate fallback returns what it can read.

So the pipeline is two round trips on purpose. `/extract` returns the text, its
confidence, and any warnings; the UI shows it in an editable box; only then does
`/screen` see it. Whether the extraction is trustworthy is a judgement the
person holding the CV can make and the server cannot.

**This is not hypothetical.** The same CV, screened twice:

| Input                       | Score     |
| --------------------------- | --------- |
| the Markdown original       | **69.2%** |
| the same CV exported to PDF | **38.5%** |

The PDF was produced by a plain-text-to-PDF converter that hard-wraps at a fixed
column, mid-word. _Build and maintain streaming inference services_ flipped from
covered to **absent**, and the evidence line the report shows for it is
`ions with 180 ms p95 first-partial latency` — the tail of a sentence that began
`…handling 400 concurrent WebSocket sess` on the previous line. A 31-point swing
with no change to the candidate, the job, the model or the scoring. That is the
whole argument for the review box.

Two smaller calls in the same module:

- **Format is sniffed from magic bytes before the extension.** A CV exported as
  PDF and named `.txt` would otherwise be handed to the model as binary.
- **`utf-16` is only tried when a BOM asks for it.** `bytes.decode("utf-16")`
  accepts almost any even-length input and returns CJK-looking mojibake, so
  offering it as a blind fallback silently mangles Latin-1 files. An early
  version did exactly that, and a test caught it.

### 10. Strong and weak points are derived; only the suggestions are generated

Asking the model to "list the strengths" lets it name a strength the scoring
never credited, and a report that contradicts its own number is worse than no
report. So strong and weak points are a deterministic reading of the
per-requirement verdicts that already produced the score — same source, no
second opinion. `weak_points` excludes `unknown` for the same reason the
denominator does: presenting our parser failure as the candidate's weakness
would be a lie.

Suggestions are the one genuinely open-ended part, so they are the one part the
model authors. They are labelled as model-written in the UI, they cannot move
the score (a test asserts the score is identical with and without them), and the
prompt forbids suggesting the candidate claim experience the CV gives no sign
of.

**An empty suggestion list is never rendered as an empty list.** Zero
suggestions under a "How to improve your chances" heading reads as "nothing to
improve", which is a very different claim from "the model returned nothing
usable". `Suggestions` carries an `error`, and both the CLI and the UI print the
reason instead.

### 11. A container is the slowest place to run the model, by 28x

Docker Desktop on macOS runs a Linux VM with **no Metal passthrough**, so an
ollama container has no GPU at all — its own log says so:

```
msg="inference compute" id=cpu library=cpu description=cpu total="11.7 GiB"
```

Measured on the same machine, same model (`llama3.2:3b`), same prompt:

| Where ollama runs    | Throughput     | One screening run |
| -------------------- | -------------- | ----------------- |
| Host (Metal)         | **99.7 tok/s** | **6 s**           |
| Container (CPU only) | **3.5 tok/s**  | minutes           |

A 28x gap, from a decision that looks like pure packaging. So the bundled
ollama moved behind a Compose profile and **the default is the host's**, via
`host.docker.internal`. `./start.sh` picks:

| Detected                                 | Backend                                                  |
| ---------------------------------------- | -------------------------------------------------------- |
| ollama answering on the host             | the host's — Metal on macOS, CUDA on Linux               |
| else an NVIDIA GPU and container runtime | bundled ollama, GPU passed through                       |
| else                                     | bundled ollama on CPU, with the slowness stated up front |

The fallback is loud rather than silent, because a silent fallback is exactly
how a 28x slowdown goes unnoticed. `./start.sh host|gpu|cpu` forces one, and
`./start.sh --down` stops everything.

The bundled ollama also **stopped publishing 11434**. It had been shadowing the
host daemon on the same port, so `localhost:11434` reached the container — which
had only the models _it_ had pulled. That cost an hour of confusion once.

### 12. A slow request that logs nothing is indistinguishable from a hang

`BaseHTTPRequestHandler.log_message` fires when the response is sent. For a
route that takes one model call per requirement plus two, that means a
multi-minute request logs _nothing at all_ until it finishes, and the only
reasonable conclusion is that the thing is stuck. It is a real defect even
though no output is wrong.

So the handler logs on request **start**, and `screen()` takes an `on_progress`
callback that the server prints:

```
POST /screen (5732 bytes) — started
  … reading requirements from the job description
  … embedding 8 requirements and 9 CV lines
  … judging 1/8: 5+ years of production Python engineering
  … judging 5/8: Kubernetes and container orchestration in production
  … writing suggestions
  … done
POST /screen -> 200
```

The callback defaults to `None` and the pipeline never imports a logger, so the
library stays quiet and the server decides what to print.

## Layout

```
backend/            standard-library Python: the pipeline, the API, the CLI
  src/resume_screen/
  tests/            381 pytest tests
  examples/         runnable with no network and no API key
  data/             a sample CV and job description
  Dockerfile        python:3.12-slim + COPY src — no pip step
frontend/           Next.js 16, Tailwind v4, shadcn/ui
  app/ components/ lib/
  __tests__/        65 Jest tests
  Dockerfile        multi-stage, standalone output
docker-compose.yml     api + web; ollama behind the `bundled` profile
docker-compose.gpu.yml NVIDIA passthrough for the bundled ollama
start.sh               picks the fastest backend, then brings the stack up
```

## Design

| Module            | Knows about                                                    |
| ----------------- | -------------------------------------------------------------- |
| `ollama.py`       | HTTP, urllib, retry-free error mapping. The only network code. |
| `documents.py`    | Text structure: bullets, paragraphs, headings. No models.      |
| `requirements.py` | Job-description shape and requirement weights.                 |
| `scoring.py`      | Cosine, centering, aggregation. Imports no client.             |
| `judge.py`        | The one prompt, and refusing to over-read the answer.          |
| `pipeline.py`     | Orchestration only; holds no scoring logic.                    |
| `fakes.py`        | A deterministic backend with hand-checkable similarities.      |
| `serialize.py`    | The JSON shape of a result. No transport, no HTTP.             |
| `api.py`          | Routes as pure functions, plus a thin `http.server` handler.   |
| `extract.py`      | Uploads → text, plus how much to trust it. No models.          |
| `advice.py`       | Points derived from verdicts; suggestions asked of the model.  |
| `server.py`       | Environment and flags → a `Config`. The container entry point. |

Test suites, and the boundary each one defends:

| Suite                              | Tests | Guards                                                                 |
| ---------------------------------- | ----- | ---------------------------------------------------------------------- |
| `test_api.py`                      | 63    | Every status path, upload handling, a real socket round trip           |
| `test_extract.py`                  | 51    | Each format, and each way each format goes wrong                       |
| `test_scoring.py`                  | 37    | Cosine, double-centering, aggregation — residuals checked by hand      |
| `test_advice.py`                   | 38    | Derived ordering, and everything a model can return that is not advice |
| `test_judge.py`                    | 27    | Everything a model can emit that is not clean JSON                     |
| `test_pipeline.py`                 | 31    | Orchestration, reproducibility, degenerate inputs                      |
| `test_report.py`                   | 27    | Text formatting, built from hand-made results                          |
| `test_documents.py`                | 22    | Chunking: bullets, headings, empty documents                           |
| `test_requirements.py`             | 38    | Requirement extraction and weighting                                   |
| `test_serialize.py`                | 19    | The JSON contract `frontend/lib/api.ts` mirrors                        |
| `test_ollama.py`                   | 17    | HTTP failures, with `urlopen` patched                                  |
| `test_cli.py`                      | 11    | Flags, exit codes, `--offline`                                         |
| `frontend/__tests__/analysis-view` | 25    | That the UI cannot launder a null score, a warning, or missing advice  |
| `frontend/__tests__/cv-upload`     | 15    | Upload, warnings, editing, and every failure path                      |
| `frontend/__tests__/api`           | 14    | Request shape, base64 encoding, error fallbacks                        |
| `frontend/__tests__/analyse-form`  | 11    | Run gating, wiring, in-flight state                                    |

The front-end suite exists for one reason: a UI is the easiest place to undo
everything the pipeline was careful about. Rendering `score: null` as "0%",
dropping the uncalibrated warning, or hiding the raw cosine behind a progress
bar would each turn an honest result into a confident one. Those are the cases
`result-view.test.tsx` asserts, and they fail if someone "tidies up" the
rendering.

Everything above `ollama.py` depends on the `Client` protocol — two methods —
so the entire pipeline runs offline. `FakeClient` returns hashed bag-of-words
embeddings: same text, same vector, on every machine, and shared vocabulary
lands closer than unrelated text. The centering tests assert residuals computed
by hand, because a scoring tool never checked against known values is measuring
its own bugs.

The client is injected as a _factory_, not an instance, so each request builds
its own frozen `OllamaClient` and the threaded handler shares no mutable state.

### Web app

```
frontend/
  app/           layout, page, globals.css (Tailwind v4 theme tokens)
  components/ui/ shadcn primitives, vendored — button, card, textarea, badge, …
  components/    analyse-form.tsx, cv-upload.tsx, analysis-view.tsx
  lib/api.ts     typed API client; ScreenResult mirrors serialize.py
```

shadcn/ui is not a dependency — it is a registry you copy from, so those
components live in the repo and are edited like any other file. Tailwind v4
needs no `tailwind.config.js`: the theme is CSS custom properties in
`globals.css` and an `@theme inline` block. `cv-upload.tsx` owns the whole upload path — dropzone, `/extract` call,
warnings, and the editable text — and reports only the text upward. The file is
read to base64 in the browser and never touched again, so the parent has one
piece of state to reason about.

The UI is built to not launder the pipeline's honesty. It renders `n/a` when
`score` is null, shows the uncalibrated warning when `calibrated` is false,
counts unscored requirements, prints the raw cosine next to the calibrated
specificity on every piece of evidence, surfaces every extraction warning, and
says _why_ there are no suggestions rather than showing an empty list. Those
are the cases `analysis-view.test.tsx` and `cv-upload.test.tsx` assert, and
they fail if someone "tidies up" the rendering.

## Running the whole stack

```bash
./start.sh
```

That is the whole thing. It detects the fastest inference backend available
(see decision 11), pulls any missing models, builds the images and brings the
stack up:

```
resume-screen — starting
  backend: ollama on the host (Metal GPU)
  nomic-embed-text:latest already present
  llama3.2:3b already present
  chat model:  llama3.2:3b
  embed model: nomic-embed-text

  web  http://localhost:3000
  api  http://localhost:8000/health
  logs docker compose logs -f api
```

| Command             | Backend                                    |
| ------------------- | ------------------------------------------ |
| `./start.sh`        | auto: host GPU → bundled GPU → bundled CPU |
| `./start.sh host`   | the ollama already running on your machine |
| `./start.sh gpu`    | bundled ollama with NVIDIA passthrough     |
| `./start.sh cpu`    | bundled ollama, CPU only                   |
| `./start.sh --down` | stop everything                            |

Or drive Compose directly. The default profile runs only `api` and `web` and
points them at `host.docker.internal`; the bundled model host is opt-in:

```bash
docker compose up --build                    # uses the host's ollama
docker compose --profile bundled up --build  # runs ollama in a container too
OFFLINE=true docker compose up               # fake backend, no model at all
CHAT_MODEL=qwen3:4b ./start.sh               # a different judge
```

`OFFLINE=true` is the useful one for wiring: the whole stack comes up and
answers real requests with the deterministic fake, with no model to download.

Under `--profile bundled`, a one-shot `ollama-pull` job fetches the models into
the `ollama-models` volume and exits, and the API waits on
`service_completed_successfully`, so the first request never races a
half-downloaded model. That volume is the only thing persisted — an uploaded CV
lives in memory for the length of the request and is never written to disk.

- Web app — <http://localhost:3000>
- API — <http://localhost:8000/health>
- Progress — `docker compose logs -f api`

Defaults are `llama3.2:3b` (2.0 GB) and `nomic-embed-text` (274 MB); override
with `CHAT_MODEL` and `EMBED_MODEL`.

## Usage

```bash
ollama serve
ollama pull nomic-embed-text        # 274 MB, the embedding model
ollama pull llama3.2:3b             # 2.0 GB, or any small instruct model

cd backend
pip install -e ".[dev]"
resume-screen --resume data/resume.md --job data/job.md

ruff check . && ruff format --check . && pytest -q
```

Model choice is a flag, not a rewrite:

```bash
resume-screen --resume r.md --job j.md --chat-model qwen3:4b --embed-model mxbai-embed-large
resume-screen --resume r.md --job j.md --offline     # fake backend, no server, no network
```

Exit codes: `0` fully scored, `1` some requirement unscored, `2` bad input,
`3` the model backend was unreachable.

As a library:

```python
from resume_screen import OllamaClient, extract, render, screen

cv_text = extract("cv.pdf", open("cv.pdf", "rb").read()).text

result = screen(OllamaClient(), cv_text, job_text)
print(result.summary.percent)

for point in result.strengths:
    print("+", point.kind, point.requirement)
for point in result.weaknesses:
    print("-", point.kind, point.verdict, point.requirement)
for suggestion in result.suggestions.items:
    print("→", suggestion)
```

Runnable examples:

```bash
python examples/offline_screen.py      # full pipeline, no network, no API key
python examples/centering_effect.py    # the measurement in decision 3 (needs Ollama)
pytest -q                              # 381 tests, no server required
```

The web app:

```bash
cd frontend
npm install
npm test            # 65 Jest tests
npm run lint        # eslint, next/core-web-vitals + typescript
npm run format      # prettier
npx tsc --noEmit
npm run dev         # expects the API on :8000
```

### The API by hand

```bash
resume-screen-api --port 8000 --offline
```

| Route           | Does                                                                         |
| --------------- | ---------------------------------------------------------------------------- |
| `GET /health`   | liveness                                                                     |
| `POST /extract` | `{filename, content_base64}` → `{text, format, confidence, warnings, words}` |
| `POST /screen`  | `{resume, job, top_k?}` → the full analysis                                  |

`/screen` returns `requirements_source`: `"headings"` when the job description
was parsed structurally, `"model"` when the requirements were read out of prose.

```bash
# extract a CV, then screen the text it returned
curl -s -X POST localhost:8000/extract -H 'Content-Type: application/json' \
  -d "{\"filename\": \"cv.pdf\", \"content_base64\": \"$(base64 -i cv.pdf)\"}"

curl -s -X POST localhost:8000/screen -H 'Content-Type: application/json' \
  -d '{"resume": "## Experience\n- built a streaming ASR gateway in Python",
       "job": "## Requirements\n- build streaming inference services"}'
```

The upload is base64 inside JSON rather than `multipart/form-data`. That costs
33% in transfer for a file measured in kilobytes, and buys an API with exactly
one content type, no multipart parser to get wrong, and a request that is
inspectable in a terminal. Multipart would be the right call for large or many
files; this is one CV.

Status codes: **400** malformed body, **413** oversized body, **422** a file
with no extractable text or a job description with no requirements, **503** the
model backend was unreachable.

A `/screen` run prints its progress to the API log — see decision 12:

```bash
docker compose logs -f api
```

## On "standard" scores

The obvious question about a tool like this is which standard it implements.
The honest answer, after going looking:

**There is no standard CV match score.** The "aim for 75–85%" figures that
circulate are vendor guidance — Jobscan's, mostly — not a standard from ISO,
HR Open Standards, or anyone else. Nothing defines what a 70% means, so nothing
makes two tools' 70% comparable. This project's number is a _weighted coverage
of the requirements this job description states_, calibrated within one
CV-and-job pair, and it says so wherever it appears.

What **is** standardised is the auditing of tools like this one:

- **EEOC Uniform Guidelines (UGESP), the four-fifths rule** — a selection rate
  below 80% of the best-performing group's rate is the threshold for adverse
  impact.
- **NYC Local Law 144** — an automated employment decision tool must have an
  annual independent bias audit publishing impact ratios by race, ethnicity and
  sex, including intersectional categories.
- **EU AI Act** — employment and worker-selection systems are Annex III
  high-risk.

That is the gap worth naming: the score is unstandardised and the _audit_ is
not, and this repo currently ships neither an audit harness nor the labelled
data one would need. See "Not included".

There are, however, real standards for the _inputs_, which is where this is
heading next:

| Thing             | Standard                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| CV                | [JSON Resume](https://docs.jsonresume.org/schema) — stable 1.0.0                                     |
| Job posting       | [schema.org/JobPosting](https://schema.org/JobPosting)                                               |
| Skills vocabulary | [ESCO](https://esco.ec.europa.eu/en/classification) — 3,039 occupations, 13,939 skills, 28 languages |

## Not included

- **Authentication, authorisation and rate limiting.** There are none, CORS is
  open to `*`, and the API will screen anything anyone posts it. That is a
  deliberate choice for a local demo and it is stated in `api.py` so it cannot
  be mistaken for an oversight. Do not put this on a network you do not own.
- **Saving anything.** There are no accounts and no storage: a CV is held in
  memory for one request. Reusing a job description means pasting it again.
- **Scanned CVs.** A scan is an image; there is no OCR here, and a PDF with no
  text layer is rejected with a message saying so rather than scored on an
  empty string.
- **Layout-aware PDF extraction.** Multi-column CVs will interleave. The
  extractor says `approximate` and shows you the text; it does not pretend to
  have solved reading order.
- **Server-side rendering of results.** The browser calls the API directly, so
  `NEXT_PUBLIC_API_URL` is baked into the client bundle at build time. A
  deployment behind a real domain would proxy through a route handler instead.
- **Bias auditing.** The prompt forbids inferring from employers and schools and
  the model only sees retrieved lines, which reduces exposure but does not
  _measure_ it. Doing this properly means the four-fifths impact ratios above,
  computed over a labelled set this repo does not have, plus counterfactual
  name-swap testing. Claiming fairness without that measurement would be worse
  than not claiming it.
- **Standards-based input representation.** The CV and job description are
  plain text end to end. JSON Resume, schema.org/JobPosting and ESCO-grounded
  skill extraction are the obvious next step and are not built yet.
- **Candidate ranking.** Scores are calibrated _within_ one resume-and-job pair.
  Comparing two candidates needs a shared null distribution across resumes; the
  current numbers are not comparable across runs and are not presented as if
  they were.
- **Retries and rate limiting.** A local single-user Ollama server needs
  neither; both would be required against a hosted API.

## License

MIT — see [LICENSE](LICENSE).
