from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 系统环境变量 > .env 文件 > 默认值
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    dashscope_base_url: Optional[str] = Field(default=None, alias="DASHSCOPE_BASE_URL")

    # PostgreSQL
    db_name: str = Field(..., alias="DB_NAME")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")


# @lru_cache(maxsize=1) # 开发环境禁用，生产环境启用
def get_settings() -> Settings:
    return Settings()


if __name__ == "__main__":
    s = get_settings()
    print("base_url   :", s.dashscope_base_url)
    print("api_key    :", s.openai_api_key[:8] + "…")