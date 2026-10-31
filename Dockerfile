FROM python:3.12-slim

LABEL org.opencontainers.image.title="agent-eval-kit" \
      org.opencontainers.image.description="Framework-neutral, task-level evaluation and regression harness for AI agents" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/laninenterline/agent-eval-harness"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["aeh"]
CMD ["--help"]
