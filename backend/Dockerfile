# The API installs nothing. The package is standard-library only, so the image
# is the Python base plus source — no pip step, no wheel cache, no lockfile,
# and nothing to audit for CVEs beyond the interpreter itself.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY src/ ./src/

# Run unprivileged. There is no auth on this API by design; that is no reason
# to also hand it root inside its own container.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["python", "-m", "resume_screen.server"]
