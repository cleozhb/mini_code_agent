"""应用配置."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
