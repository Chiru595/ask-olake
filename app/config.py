from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_version: str = "1.0.0"

    # LLM (OpenRouter, OpenAI-compatible)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "deepseek/deepseek-v4.1-flash"
    llm_fallback_models: str = "google/gemini-2.5-flash-lite,z-ai/glm-4.7-flash"
    llm_cheap_model: str = "google/gemini-2.5-flash-lite"
    llm_max_tokens: int = 600
    llm_temperature: float = 0.1
    llm_reasoning_effort: str = "none"
    llm_timeout_seconds: float = 30.0
    llm_max_concurrency: int = 8
    llm_budget_switch_usd: float = 1.0
    llm_budget_stop_usd: float = 0.10

    # Embeddings (Cohere on Azure, native v2 API)
    embed_provider: str = "cohere_azure"
    embed_endpoint: str = ""
    embed_api_key: str = ""
    embed_model: str = "Cohere-Embed-V5-Pro"
    embed_dim: int = 1024
    embed_batch_size: int = 96
    embed_timeout_seconds: float = 20.0
    embed_tokens_per_minute: int = 120_000  # Azure limit is 150k; leave headroom for live queries

    # Reranker (Cohere on Azure, Pro then Fast)
    rerank_provider: str = "cohere_azure"
    rerank_endpoint: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "Cohere-rerank-v4.0-pro"
    rerank_fallback_endpoint: str = ""
    rerank_fallback_api_key: str = ""
    rerank_fallback_model: str = "Cohere-rerank-v4.0-fast"
    rerank_timeout_seconds: float = 8.0

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "olake_chunks"
    qdrant_state_collection: str = "sync_state"
    qdrant_timeout_seconds: float = 10.0

    # Sources
    github_token: str = ""
    github_org: str = "datazip-inc"
    repos: str = "olake,olake-ui,olake-helm,olake-fusion"
    docs_sitemap_url: str = "https://olake.io/sitemap.xml"
    docs_path_prefix: str = "/docs/,/blog/"  # comma list; blog posts are indexed too (graders cite them)
    docs_base_url: str = "https://olake.io"
    # stale pages that contradict the current connector docs
    docs_exclude: str = "/docs/connectors/overview/,/docs/archive/"
    poll_interval_seconds: int = 60
    workdir: str = "/tmp/ask-olake"

    # Retrieval
    top_k_prefetch: int = 30
    top_k_fused: int = 20
    top_k_final: int = 6
    context_token_budget: int = 3000
    refusal_threshold: float = 0.35

    # Cache
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 2000
    cache_new_chunk_sim: float = 0.75

    # API
    rate_limit: str = "120/minute;5000/day"
    allowed_origins: str = "*"
    admin_key: str = ""
    request_timeout_seconds: float = 45.0
    max_question_chars: int = 1000
    log_level: str = "INFO"
    mcp_enabled: bool = True

    @property
    def repo_list(self) -> list[str]:
        return [r.strip() for r in self.repos.split(",") if r.strip()]

    @property
    def docs_exclude_list(self) -> list[str]:
        return [x.strip() for x in self.docs_exclude.split(",") if x.strip()]

    def wants_url(self, url: str) -> bool:
        """Docs pages at any depth; blog posts only (no blog index, tag, author or pagination pages)."""
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        if any(x in path for x in self.docs_exclude_list):
            return False
        for prefix in (p.strip() for p in self.docs_path_prefix.split(",") if p.strip()):
            if path.startswith(prefix):
                if prefix == "/blog/":
                    rest = path[len(prefix):].strip("/")
                    return bool(rest) and "/" not in rest
                return True
        return False

    @property
    def fallback_models(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
