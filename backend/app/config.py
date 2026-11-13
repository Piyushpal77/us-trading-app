from pathlib import Path
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_cache"
MODELS_DIR = BASE_DIR / "models_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ScreenerConfig(BaseModel):
    """Configurable Triple-Barrier, liquidity, and scan parameters for US Equities."""

    target_atr_mult: float = Field(default=2.5, ge=0.5, le=10.0)
    stop_atr_mult: float = Field(default=2.0, ge=0.5, le=5.0)
    horizon_days: int = Field(default=15, ge=3, le=60)
    min_avg_dollar_vol_m: float = Field(
        default=25.0,
        ge=0.0,
        le=5000.0,
        description="Minimum 20D average daily dollar volume in USD Millions.",
    )
    min_target_gain_pct: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Minimum potential target gain % required to enter Top N recommendations.",
    )
    min_win_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=0.95,
        description="Minimum ML win probability required to recommend a trade (0.0 = always show top_k).",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    history_years: int = Field(default=10, ge=2, le=15)
    provisional_cutoff_et: str = Field(default="15:50")
    universe_limit: int = Field(
        default=550,
        ge=10,
        le=600,
        description="Max S&P 500 + Nasdaq 100 symbols to scan.",
    )


DEFAULT_CONFIG = ScreenerConfig()
