from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 127.0.0.1, not localhost: the container publishes IPv4 loopback only, and a "localhost" lookup tries::1 first and
    # stalls for the connect timeout before falling back.
    pg_dsn: str = "postgresql://postgres:postgres@127.0.0.1:5432/hansard"
    # The local GPU service (gpu/api.py): query embedding and reranking.
    gpu_url: str = "http://127.0.0.1:8001"
    deepseek_key: str = ""
    # Hugging Face access token.
    hf_token: str = ""

    @property
    def huggingface_token(self) -> str:
        """The token under any of the names people actually write in.env."""
        import os

        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
                     "HUGGINGFACE_TOKEN", "HUGGINGFACETOKEN"):  # fmt: skip
            if value := os.environ.get(name, "").strip():
                return value
        return (self.hf_token or "").strip()

    # The account offers "deepseek-flash" and "deepseek-v4-pro".
    bulk_model: str = "deepseek-flash"
    # A hard floor on the provider balance, in USD. 0 disables it: the maintainer lifted the floor because a floor above
    # the live balance blocked the 1p checks that exist to avoid waste.
    min_balance_usd: float = 0.0
    # What a single run may cost before it has to be approved.
    spend_confirm_gbp: float = 0.50
    reason_model: str = "deepseek-v4-pro"
    deepseek_base: str = "https://api.deepseek.com"


settings = Settings()
