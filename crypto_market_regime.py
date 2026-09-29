"""
crypto_market_regime.py
========================
멀티거래소(Binance / OKX / Bitget) 실시간 데이터 기반
크립토 시장 국면(상승/횡보/하락) 판단 + 코인 스크리너 + TP/SL 제안 도구

⚠️ 중요 전제 (반드시 읽고 사용하세요)
--------------------------------
1. 이 스크립트는 "수익을 보장하는 시그널 생성기"가 아닙니다.
   - 시장 국면을 객관적 지표로 요약하고, 조건에 맞는 후보를 걸러주는 "의사결정 보조 도구"입니다.
   - 여기서 나온 TP/SL/방향은 참고용이며, 최종 진입/청산 판단과 책임은 사용자 본인에게 있습니다.
2. 실전 투입 전 반드시 아래 순서를 거치세요:
   a) 최소 3~6개월 페이퍼 트레이딩(모의투자)으로 신호 품질 검증
   b) backtest_wfo() 로 워크포워드 검증 (과거 특정 구간에만 맞춰진 과최적화 여부 확인)
   c) 실전 투입 시 레버리지는 규칙 기반으로 상한을 강제 (이 스크립트는 레버리지 추천을 하지 않습니다)
3. 네트워크가 막힌 환경(샌드박스)에서는 실행되지 않습니다. 로컬/서버에서 다음을 설치 후 실행하세요:
   pip install ccxt requests pandas numpy

작성 방식: 단일 파일, 모듈형 함수 구성. main() 에서 전체 파이프라인을 한 번에 실행합니다.
"""

import time
import json
import os
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import requests
import pandas as pd
import numpy as np

try:
    import ccxt
except ImportError:
    ccxt = None  # 실행 시 pip install ccxt 안내

# --------------------------------------------------------------------------
# 0. 설정
# --------------------------------------------------------------------------

EXCHANGES = ["bitget", "okx", "binance"]          # 앞쪽일수록 우선 사용(Bitget = 실제 거래 거래소). 일부 거래소는 서버 지역에 따라 차단될 수 있음
QUOTE = "USDT"
TOP_N_BY_VOLUME = 30                              # 거래량 상위 N개 코인만 스크리닝
TIMEFRAME = "4h"                                  # 스윙 트레이딩 기준 봉
OHLCV_LIMIT = 200                                 # 캔들 개수
HISTORY_FILE = "market_regime_history.csv"        # BTC.D / USDT.D / TOTAL2,3 스냅샷 누적 저장 (트렌드 판단용)
REGIME_LOG_FILE = "regime_confirmation_log.csv"   # 국면 whipsaw 방지용 확정 이력

CoinGeckoGlobalURL = "https://api.coingecko.com/api/v3/global"

RegimeType = Literal["uptrend", "downtrend", "sideways"]


@dataclass
class RiskConfig:
    """계좌 단위 리스크 관리 설정. '얼마나 걸지'는 신호와 완전히 분리해서 관리해야 합니다."""
    account_balance: float          # 계좌 총 잔고 (USDT 기준)
    risk_per_trade_pct: float = 1.0  # 트레이드 1건당 허용 손실 (계좌 대비 %). 권장 0.5~2%
    max_correlated_exposure_pct: float = 3.0  # BTC 방향에 동조된 포지션들의 합산 리스크 상한(%)
    max_concurrent_setups: int = 3   # 같은 방향(롱 또는 숏) 동시 보유 최대 개수


def calculate_position_size(entry: float, sl: float, risk_cfg: RiskConfig) -> Dict:
    """RR이 아무리 좋아도 '얼마를 걸지'는 항상 이 공식으로만 결정합니다.
    포지션 크기 = (계좌잔고 * 리스크%) / |entry - sl|
    → SL에 닿아도 계좌 손실이 risk_per_trade_pct를 넘지 않도록 강제."""
    risk_amount = risk_cfg.account_balance * (risk_cfg.risk_per_trade_pct / 100)
    per_unit_risk = abs(entry - sl)
    if per_unit_risk <= 0:
        return {"size": 0, "risk_amount": risk_amount, "notional": 0}
    size = risk_amount / per_unit_risk
    notional = size * entry
    return {"size": size, "risk_amount": risk_amount, "notional": notional}


def cap_correlated_exposure(setups: List["CoinSetup"], risk_cfg: RiskConfig,
                             assumed_correlation: float = 0.6) -> List["CoinSetup"]:
    """같은 방향(롱/숏) 알트코인들은 BTC와 0.5~0.8 수준으로 동조화되는 경우가 흔해서
    (여러 개 들고 있어도 사실상 '하나의 큰 베팅'과 비슷) 두 단계로 제한합니다:

    1) 개수 제한: RR 상위 max_concurrent_setups개만 남김
    2) 상관조정: '유효 독립 베팅 수' = n / (1 + (n-1) * 평균상관계수) 공식으로
       실제 분산 효과가 얼마나 되는지 계산해서 함께 출력 (n=1이면 전혀 분산 안 된 것)
    """
    def _long_rank(s):
        return (s.rs, s.asymmetry if s.asymmetry is not None else 0.0, s.rr_ratio)

    def _short_rank(s):
        return (-s.rs, -(s.asymmetry if s.asymmetry is not None else 0.0), s.rr_ratio)

    long_like = sorted([s for s in setups if s.bias in ("long", "wait_breakout_long", "range_fade_long")],
                        key=_long_rank, reverse=True)[:risk_cfg.max_concurrent_setups]
    short_like = sorted([s for s in setups if s.bias in ("short", "wait_breakout_short", "range_fade_short")],
                         key=_short_rank, reverse=True)[:risk_cfg.max_concurrent_setups]

    for group, label in [(long_like, "롱"), (short_like, "숏")]:
        n = len(group)
        if n > 1:
            n_eff = n / (1 + (n - 1) * assumed_correlation)
            print(f"[상관관계] {label} {n}개 동시 보유 → 유효 독립 베팅 수 ≈ {n_eff:.1f}개 "
                  f"(가정 상관계수 {assumed_correlation}) — 실제 분산 효과는 숫자보다 훨씬 작습니다.")

    return long_like + short_like


# --------------------------------------------------------------------------
# 1. 거시 지표: BTC.D, USDT.D, TOTAL2, TOTAL3
# --------------------------------------------------------------------------

def fetch_global_snapshot() -> Dict:
    """CoinGecko 글로벌 마켓캡 데이터에서 BTC.D, USDT.D, TOTAL2, TOTAL3를 계산.
    (CoinGecko 무료 API는 '현재 스냅샷'만 제공하므로, 추세는 HISTORY_FILE에
    스냅샷을 누적 저장해서 별도로 계산합니다. 이 스크립트를 주기적으로
    cron 등으로 돌리면 시간이 지날수록 추세 판단 정확도가 올라갑니다.)
    """
    resp = requests.get(CoinGeckoGlobalURL, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]

    total_mcap = data["total_market_cap"]["usd"]
    btc_pct = data["market_cap_percentage"].get("btc", 0)
    eth_pct = data["market_cap_percentage"].get("eth", 0)
    usdt_pct = data["market_cap_percentage"].get("usdt", 0)

    total2 = total_mcap * (1 - btc_pct / 100)
    total3 = total_mcap * (1 - btc_pct / 100 - eth_pct / 100)

    snapshot = {
        "timestamp": int(time.time()),
        "total_mcap": total_mcap,
        "btc_d": btc_pct,
        "usdt_d": usdt_pct,
        "eth_d": eth_pct,
        "total2": total2,
        "total3": total3,
    }
    _append_history(snapshot)
    return snapshot


def _append_history(snapshot: Dict) -> None:
    df_new = pd.DataFrame([snapshot])
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(HISTORY_FILE, index=False)


def macro_history_span_hours(window_days: float = 7.0) -> float:
    """BTC.D/USDT.D/TOTAL2·3 스냅샷이 몇 시간 분량 쌓였는지 (추세 판단 가능 여부 확인용)."""
    if not os.path.exists(HISTORY_FILE):
        return 0.0
    df = pd.read_csv(HISTORY_FILE)
    if df.empty or "timestamp" not in df.columns:
        return 0.0
    df = df[df["timestamp"] >= df["timestamp"].max() - window_days * 86400]
    if len(df) < 2:
        return 0.0
    return float((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 3600)


_MACRO_THRESHOLD_PCT = {"btc_d": 1.0, "usdt_d": 1.0, "total2": 5.0, "total3": 5.0}


def macro_trend_from_history(column: str, window_days: float = 7.0,
                              min_span_hours: float = 24.0) -> RegimeType:
    """누적 스냅샷으로 지표(btc_d, usdt_d, total2, total3)의 추세를 '시간' 기준으로 판단.
    (실행 빈도와 무관하게 일관되도록 '몇 개 쌓였나'가 아니라 '최근 window_days일 변화율'을 봅니다.)
    - 기록 기간이 min_span_hours 미만이면 판단 근거 부족 → 'sideways'
    - 변화율 기준: 도미넌스는 ±1%, TOTAL2/3는 ±5% (상대 변화율)"""
    if not os.path.exists(HISTORY_FILE):
        return "sideways"
    df = pd.read_csv(HISTORY_FILE)
    if df.empty or column not in df.columns or "timestamp" not in df.columns:
        return "sideways"
    df = df[df["timestamp"] >= df["timestamp"].max() - window_days * 86400]
    if len(df) < 2:
        return "sideways"
    span_hours = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 3600
    if span_hours < min_span_hours:
        return "sideways"
    first, last = float(df[column].iloc[0]), float(df[column].iloc[-1])
    if first == 0:
        return "sideways"
    pct_change = (last - first) / first * 100
    th = _MACRO_THRESHOLD_PCT.get(column, 2.0)
    if pct_change > th:
        return "uptrend"
    if pct_change < -th:
        return "downtrend"
    return "sideways"


# --------------------------------------------------------------------------
# 2. BTC 가격 추세 (거시 국면의 핵심 축)
# --------------------------------------------------------------------------

_EX_CACHE: Dict = {}


def _get_ex(exchange_id: str):
    """거래소 연결을 한 번만 만들어 재사용. (호출마다 새로 만들면 시장 목록을 매번 다시 받아
    매우 느려지고 API 차단 위험이 커집니다.)"""
    if ccxt is None:
        raise RuntimeError("ccxt가 설치되어 있지 않습니다. `pip install ccxt` 후 재시도하세요.")
    if exchange_id not in _EX_CACHE:
        _EX_CACHE[exchange_id] = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15000})
    return _EX_CACHE[exchange_id]


def fetch_btc_df() -> pd.DataFrame:
    """BTC 4h 캔들. 한 거래소가 지역 차단/장애여도 다음 거래소로 넘어가도록 순차 시도."""
    for ex_id in EXCHANGES:
        df = fetch_ohlcv(ex_id, f"BTC/{QUOTE}")
        if df is not None and len(df) > 0:
            return df
    raise RuntimeError("모든 거래소에서 BTC 데이터를 가져오지 못했습니다 (네트워크/지역 차단/거래소 장애 확인)")


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str = TIMEFRAME,
                 limit: int = OHLCV_LIMIT) -> Optional[pd.DataFrame]:
    if ccxt is None:
        raise RuntimeError("ccxt가 설치되어 있지 않습니다. `pip install ccxt` 실행 후 재시도하세요.")
    try:
        ex = _get_ex(exchange_id)
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception as e:
        print(f"[warn] {exchange_id} {symbol} OHLCV 조회 실패: {e}")
        return None


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> float:
    """추세 강도(ADX) 계산 — 값이 높을수록 '방향성 있는 추세', 낮으면 '횡보'로 해석.
    ⚠️ Wilder 정의상 한 봉에서 +DM/-DM은 상호 배타적입니다(그 봉에서 더 크게 움직인 쪽만 인정).
    이걸 지키지 않고 둘 다 독립적으로 클리핑하면(예: 고가↑ 저가↓가 동시에 발생하는 흔한 봉에서
    +DM과 -DM이 동시에 양수가 됨) 방향성이 없는 순수 노이즈에서도 ADX가 과대평가되어
    '횡보'를 '추세'로 오판하게 됩니다."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return float(dx.rolling(period).mean().iloc[-1]) if not dx.empty else 0.0


def classify_price_trend(df: pd.DataFrame, adx_threshold: float = 20.0) -> RegimeType:
    """EMA50 vs EMA200 배열 + ADX 강도로 상승/하락/횡보 분류."""
    if df is None or len(df) < 60:
        return "sideways"

    ema_fast = ema(df["close"], 50)
    ema_slow = ema(df["close"], min(200, len(df) - 1))
    strength = adx(df)

    if strength < adx_threshold:
        return "sideways"
    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "uptrend"
    return "downtrend"


# --------------------------------------------------------------------------
# 3. 전체 국면 종합 판단
# --------------------------------------------------------------------------

@dataclass
class MarketRegime:
    btc_trend: RegimeType
    btc_d_trend: RegimeType
    usdt_d_trend: RegimeType
    total2_trend: RegimeType
    total3_trend: RegimeType
    overall: RegimeType
    snapshot: Dict = field(default_factory=dict)
    headline: str = ""            # 한 줄 결론 (예: "횡보 후 상승 우세")
    score: float = 0.0            # -1(강한 하락)~+1(강한 상승) 종합 방향 점수
    confidence_label: str = ""    # "높음"/"보통"/"낮음" — 근거들이 서로 얼마나 일치하는지
    explanation: str = ""         # 근거를 풀어 쓴 설명 문장
    breakout_up: Optional[float] = None    # 횡보일 때: 이 가격 위로 뚫으면 상승 전환으로 볼 기준선
    breakout_down: Optional[float] = None  # 횡보일 때: 이 가격 아래로 이탈하면 하락 전환으로 볼 기준선
    lean_verified: Optional[bool] = None   # 횡보 기울기 근거가 검증됐는지(=validate_lean_auto 결과 반영 여부)


def fetch_btc_daily_trend() -> RegimeType:
    """BTC 일봉(HTF) 추세. 한 거래소가 막혀도 다음 거래소로 순차 시도."""
    for ex_id in EXCHANGES:
        try:
            return get_htf_trend(ex_id, f"BTC/{QUOTE}")
        except Exception:
            continue
    return "sideways"


def _confidence_label(agreement: float) -> str:
    if agreement >= 0.6:
        return "높음"
    if agreement >= 0.3:
        return "보통"
    return "낮음"


def build_market_narrative(btc_df: pd.DataFrame, overall: RegimeType, btc_trend: RegimeType,
                            btc_d_trend: RegimeType, usdt_d_trend: RegimeType,
                            total2_trend: RegimeType, total3_trend: RegimeType,
                            daily_trend: RegimeType, macro_span_hours: float) -> Dict:
    """지표들을 한 줄 결론 + 방향 점수 + 풀어쓴 설명으로 종합. 칩 나열 대신 사람이 바로
    이해할 수 있는 '종합 서사'를 만드는 게 목적."""
    macro_ready = macro_span_hours >= 24

    if overall != "sideways":
        # 추세장: 몇 개 지표가 같은 방향을 가리키는지로 강도·신뢰도를 매김
        agree, total = 1, 1  # btc_trend 자신
        if macro_ready:
            for t, invert in ((btc_d_trend, True), (usdt_d_trend, True),
                              (total2_trend, False), (total3_trend, False)):
                total += 1
                t_aligned = ("downtrend" if invert else "uptrend") if overall == "uptrend" else \
                            ("uptrend" if invert else "downtrend")
                if t == t_aligned:
                    agree += 1
        agreement = agree / total
        score = agreement if overall == "uptrend" else -agreement
        strong = agreement >= 0.75
        headline = ("강한 상승장" if strong else "상승장") if overall == "uptrend" else \
                   ("강한 하락장" if strong else "하락장")

        parts = [f"BTC가 {'상승' if overall == 'uptrend' else '하락'} 추세예요"]
        if macro_ready:
            if btc_d_trend == ("downtrend" if overall == "uptrend" else "uptrend"):
                parts.append("BTC 도미넌스도 알트코인 쪽에 힘을 실어주는 방향이고" if overall == "uptrend"
                             else "BTC 도미넌스도 같은 방향으로 힘을 싣고 있고")
            if usdt_d_trend == ("downtrend" if overall == "uptrend" else "uptrend"):
                parts.append("현금성 자금(USDT)이 시장에서 빠져나가지 않고 있어요" if overall == "uptrend"
                             else "현금성 자금(USDT)으로 자금이 이동하고 있어요")
            if total2_trend == overall and total3_trend == overall:
                parts.append("알트코인 시가총액(TOTAL2/3)도 같이 움직이고 있어 폭넓게 힘이 실려 있어요")
        else:
            parts.append(f"(도미넌스·TOTAL 지표는 기록이 {macro_span_hours:.0f}시간 쌓여 아직 판단에서 제외됨)")
        explanation = ". ".join(parts) + "."

        return {"headline": headline, "score": float(score),
                "confidence_label": _confidence_label(agreement), "explanation": explanation,
                "breakout_up": None, "breakout_down": None, "lean_verified": None}

    # 횡보장: HTF 추세 + 박스 내부 구조 + 매집/분산 거래량으로 다음 방향 기울기를 계산
    lean = compute_sideways_lean(btc_df, daily_trend)
    box_high = float(btc_df["high"].tail(20).max())
    box_low = float(btc_df["low"].tail(20).min())

    if lean["score"] > 0.3:
        headline = "횡보 후 상승 우세"
    elif lean["score"] < -0.3:
        headline = "횡보 후 하락 우세"
    else:
        headline = "횡보 · 방향 대기"

    def _lbl(v: float, pos: str, neg: str) -> str:
        return pos if v > 0.3 else (neg if v < -0.3 else "뚜렷한 신호 없음")

    parts = [
        f"BTC가 박스권 안에서 쉬고 있어요 ({fmt_range(box_low)} ~ {fmt_range(box_high)})",
        f"일봉 추세는 {'상승' if daily_trend == 'uptrend' else '하락' if daily_trend == 'downtrend' else '중립'}",
        f"박스 안 구조는 {_lbl(lean['structure'], '저점이 높아지는(상승 우호) 패턴', '고점이 낮아지는(하락 우호) 패턴')}",
        f"거래량은 {_lbl(lean['accumulation'], '매집(상승 선호)', '분산(하락 선호)')} 신호예요",
    ]
    explanation = ". ".join(parts) + ". 이 방향 판단은 아직 실측으로 검증되지 않았으니 참고만 하세요."

    return {"headline": headline, "score": float(np.clip(lean["score"], -1, 1)),
            "confidence_label": _confidence_label(min(abs(lean["score"]) / 0.6, 1.0)),
            "explanation": explanation, "breakout_up": box_high, "breakout_down": box_low,
            "lean_verified": False}


def fmt_range(x: float) -> str:
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 1:
        return f"{x:,.4g}"
    return f"{x:.6g}"


def determine_overall_regime() -> MarketRegime:
    snap = fetch_global_snapshot()
    btc_df = fetch_btc_df()

    btc_trend = classify_price_trend(btc_df)
    btc_d_trend = macro_trend_from_history("btc_d")
    usdt_d_trend = macro_trend_from_history("usdt_d")
    total2_trend = macro_trend_from_history("total2")
    total3_trend = macro_trend_from_history("total3")

    # 종합 판단 로직 (단순 다수결 + BTC 가격추세 가중)
    votes = [btc_trend]
    if btc_d_trend == "downtrend":
        votes.append("uptrend")
    elif btc_d_trend == "uptrend":
        votes.append("downtrend")
    if usdt_d_trend == "uptrend":
        votes.append("downtrend")
    elif usdt_d_trend == "downtrend":
        votes.append("uptrend")
    votes.append(total2_trend)
    votes.append(total3_trend)

    up = votes.count("uptrend")
    down = votes.count("downtrend")
    if up >= down + 2:
        raw_regime = "uptrend"
    elif down >= up + 2:
        raw_regime = "downtrend"
    else:
        raw_regime = "sideways"

    # 도미넌스/TOTAL 추세는 기록이 24시간 이상 쌓여야 판단 가능 — 그 전에는 전부 '횡보'로 찍혀
    # BTC 추세를 희석시키므로, 기록이 부족한 동안은 BTC 가격 추세만으로 국면을 판단
    span_h = macro_history_span_hours()
    snap["macro_span_hours"] = span_h
    if span_h < 24:
        raw_regime = btc_trend

    overall = _apply_regime_hysteresis(raw_regime)
    daily_trend = fetch_btc_daily_trend() if overall == "sideways" else "sideways"
    narrative = build_market_narrative(btc_df, overall, btc_trend, btc_d_trend, usdt_d_trend,
                                       total2_trend, total3_trend, daily_trend, span_h)

    return MarketRegime(btc_trend, btc_d_trend, usdt_d_trend,
                         total2_trend, total3_trend, overall, snap, **narrative)


def _apply_regime_hysteresis(raw_regime: RegimeType, confirm_count: int = 2) -> RegimeType:
    """국면이 매번 스크립트를 돌릴 때마다 뒤집히면(whipsaw) 실전에서 혼란만 커집니다.
    직전 raw_regime 기록을 REGIME_LOG_FILE에 누적하고, 최근 confirm_count회 연속
    같은 국면이 나와야만 실제로 '전환'된 것으로 인정합니다. 그 전까지는 이전에
    확정됐던 국면을 그대로 유지합니다 (스크립트를 주기적으로 여러 번 돌릴수록 효과적)."""
    log_entry = pd.DataFrame([{"ts": int(time.time()), "raw_regime": raw_regime}])
    if os.path.exists(REGIME_LOG_FILE):
        log = pd.concat([pd.read_csv(REGIME_LOG_FILE), log_entry], ignore_index=True)
    else:
        log = log_entry
    log.to_csv(REGIME_LOG_FILE, index=False)

    recent = log["raw_regime"].tail(confirm_count).tolist()
    if len(recent) < confirm_count or len(set(recent)) > 1:
        # 아직 연속 확인이 안 됐으면, 마지막으로 '확정'됐던 국면을 그대로 유지
        confirmed = log.get("confirmed_regime")
        if confirmed is not None and len(confirmed.dropna()) > 0:
            return confirmed.dropna().iloc[-1]
        return raw_regime  # 첫 실행이라 이전 확정 기록이 없으면 raw 그대로 사용

    log.loc[log.index[-1], "confirmed_regime"] = raw_regime
    log.to_csv(REGIME_LOG_FILE, index=False)
    return raw_regime


# --------------------------------------------------------------------------
# 4. 코인 스크리너 (거래량 상위 + 상대강도 + 오더블록 + 변동성)
# --------------------------------------------------------------------------

@dataclass
class CoinSetup:
    symbol: str
    exchange: str
    bias: Literal["long", "short", "wait_breakout_long", "wait_breakout_short",
                  "range_fade_long", "range_fade_short"]
    entry_note: str
    entry_price: float   # 추격이 아닌, 되돌림 지정가(limit) 진입가
    current_price: float # 참고용 현재가 (추격 여부 비교용)
    tp1: float
    tp2: float
    sl: float
    rr_ratio: float
    is_chase: bool = False  # True면 아직 되돌림 전(=추격 구간)이라는 경고 플래그
    poc_confluence: bool = False  # True면 진입가가 POC/Value Area와도 겹치는 고신뢰 구간
    rs: float = 0.0  # 지수(BTC) 대비 상대강도(%) — 클수록 시장 대비 강함
    asymmetry: Optional[float] = None  # 상승포착률-하락포착률 — 클수록 '오를 때 크게 빠질 때 작게'
    bitget_perp: Optional[bool] = None  # Bitget USDT 무기한 선물 거래 가능 여부(None=확인 불가)


def get_top_volume_symbols(exchange_id: str, top_n: int = TOP_N_BY_VOLUME) -> List[str]:
    ex = _get_ex(exchange_id)
    markets = ex.load_markets()
    tickers = ex.fetch_tickers()
    usdt_pairs = [
        s for s in markets
        if s.endswith(f"/{QUOTE}") and markets[s].get("active", True) and markets[s].get("spot", True)
    ]
    ranked = sorted(
        usdt_pairs,
        key=lambda s: tickers.get(s, {}).get("quoteVolume") or 0,
        reverse=True,
    )
    return ranked[:top_n]


def relative_strength_vs_btc(coin_df: pd.DataFrame, btc_df: pd.DataFrame,
                              windows: List[int] = [10, 20, 50]) -> float:
    """지수(BTC) 대비 상대강도를 여러 구간(10/20/50봉)의 평균으로 계산.
    - 한 구간만 보면 우연한 단기 스파이크에 흔들리기 쉬워서, 여러 구간을 평균내
      '꾸준히 지수보다 강한' 코인을 더 정확히 골라내도록 함
    - 값이 클수록 같은 기간 동안 BTC보다 더 많이 오르고(하락장이면 덜 빠지고) 있다는 뜻
    - 반환값(%): 각 구간별 (코인 수익률 - BTC 수익률)의 단순 평균"""
    scores = []
    for w in windows:
        n = min(len(coin_df), len(btc_df), w)
        if n < 2:
            continue
        coin_ret = coin_df["close"].iloc[-1] / coin_df["close"].iloc[-n] - 1
        btc_ret = btc_df["close"].iloc[-1] / btc_df["close"].iloc[-n] - 1
        scores.append((coin_ret - btc_ret) * 100)
    return float(np.mean(scores)) if scores else 0.0


def detect_order_block(df: pd.DataFrame, lookback: int = 30) -> Dict:
    """단순화된 오더블록 탐지:
    - 강한 상승 임펄스 직전의 마지막 음봉 = 불리시 오더블록
    - 강한 하락 임펄스 직전의 마지막 양봉 = 베어리시 오더블록
    엄밀한 스마트머니 컨셉 정의와는 차이가 있는 '실전 근사치'입니다."""
    recent = df.tail(lookback).reset_index(drop=True)
    body = (recent["close"] - recent["open"]).abs()
    avg_body = body.mean()

    bullish_ob, bearish_ob = None, None
    for i in range(1, len(recent) - 1):
        is_impulse_up = (recent["close"][i] - recent["open"][i]) > 2 * avg_body
        is_impulse_down = (recent["open"][i] - recent["close"][i]) > 2 * avg_body
        prev_is_bear = recent["close"][i - 1] < recent["open"][i - 1]
        prev_is_bull = recent["close"][i - 1] > recent["open"][i - 1]

        if is_impulse_up and prev_is_bear:
            bullish_ob = {"low": float(recent["low"][i - 1]), "high": float(recent["high"][i - 1])}
        if is_impulse_down and prev_is_bull:
            bearish_ob = {"low": float(recent["low"][i - 1]), "high": float(recent["high"][i - 1])}

    return {"bullish_ob": bullish_ob, "bearish_ob": bearish_ob}


def get_htf_trend(exchange_id: str, symbol: str, htf_timeframe: str = "1d") -> RegimeType:
    """상위 타임프레임(HTF) 추세 필터. 진입 타임프레임(4h) 대비 6배 비율(1d)을 사용.
    ⚠️ 반드시 '마감된 봉'만 사용 — 마지막 봉은 아직 진행 중일 수 있으므로 제외합니다
    (미래참조/look-ahead 오류 방지)."""
    df = fetch_ohlcv(exchange_id, symbol, timeframe=htf_timeframe, limit=120)
    if df is None or len(df) < 60:
        return "sideways"
    closed_df = df.iloc[:-1]  # 마지막 봉(미완성 가능성) 제외
    return classify_price_trend(closed_df)


def get_spread_pct(exchange_id: str, symbol: str) -> Optional[float]:
    """호가 스프레드(%) 조회 — 스프레드가 넓으면 그만큼 즉시 손실을 안고 시작하는 셈이라
    슬리피지 리스크가 큰 종목을 걸러내는 데 사용."""
    try:
        ex = _get_ex(exchange_id)
        ticker = ex.fetch_ticker(symbol)
        bid, ask = ticker.get("bid"), ticker.get("ask")
        if not bid or not ask:
            return None
        return (ask - bid) / bid * 100
    except Exception:
        return None


# --------------------------------------------------------------------------
# 4.1 서킷브레이커 (일일/주간 손실 한도) — 실제 체결 결과를 기록해두면
#     이 한도를 넘었을 때 스크립트가 신규 신호를 아예 막아버립니다.
# --------------------------------------------------------------------------

TRADE_LOG_FILE = "trade_results_log.csv"


def log_trade_result(pnl_usdt: float, symbol: str = "", note: str = "") -> None:
    """실제 체결 후 손익을 여기에 직접 기록하세요 (수동). 이 로그가 쌓여야
    서킷브레이커와, 나중에 Kelly 기반 사이징으로 넘어갈 때의 승률/손익비 계산이 가능합니다."""
    entry = pd.DataFrame([{"ts": int(time.time()), "pnl_usdt": pnl_usdt, "symbol": symbol, "note": note}])
    if os.path.exists(TRADE_LOG_FILE):
        log = pd.concat([pd.read_csv(TRADE_LOG_FILE), entry], ignore_index=True)
    else:
        log = entry
    log.to_csv(TRADE_LOG_FILE, index=False)


def circuit_breaker_triggered(risk_cfg: RiskConfig, max_daily_loss_pct: float = 5.0,
                               max_weekly_loss_pct: float = 10.0) -> Optional[str]:
    """오늘/이번 주 실현 손실이 한도를 넘었으면 신규 진입을 전면 차단.
    이유: 손실 중 감정적으로 만회하려는 시도(revenge trading)가 계좌를 가장 크게
    파괴하는 패턴이므로, 규칙 기반으로 강제 중단하는 것이 중요합니다."""
    if not os.path.exists(TRADE_LOG_FILE):
        return None
    log = pd.read_csv(TRADE_LOG_FILE)
    if log.empty:
        return None

    log["dt"] = pd.to_datetime(log["ts"], unit="s")
    now = pd.Timestamp.now()

    today_pnl = log[log["dt"].dt.date == now.date()]["pnl_usdt"].sum()
    week_pnl = log[log["dt"] >= now - pd.Timedelta(days=7)]["pnl_usdt"].sum()

    today_loss_pct = -today_pnl / risk_cfg.account_balance * 100
    week_loss_pct = -week_pnl / risk_cfg.account_balance * 100

    if today_loss_pct >= max_daily_loss_pct:
        return f"일일 손실 한도 초과 ({today_loss_pct:.1f}% ≥ {max_daily_loss_pct}%) — 오늘 신규 진입 중단"
    if week_loss_pct >= max_weekly_loss_pct:
        return f"주간 손실 한도 초과 ({week_loss_pct:.1f}% ≥ {max_weekly_loss_pct}%) — 이번 주 신규 진입 중단"
    return None


def get_funding_rate(exchange_id: str, symbol: str) -> Optional[float]:
    """USDT-M 무기한 선물(예: 'SOL/USDT:USDT')의 현재 펀딩비 조회.
    (현물 심볼로 조회하면 항상 실패해서 필터가 무력화되므로 반드시 선물 심볼을 사용)
    선물이 없거나 조회 실패 시 None → 이 경우 필터를 건너뜁니다.
    ⚠️ 연환산은 8시간 정산 기준 근사치입니다(코인/거래소에 따라 정산 주기가 다를 수 있음)."""
    try:
        ex = _get_ex(exchange_id)
        ex.load_markets()
        swap_symbol = f"{symbol}:{QUOTE}"
        if swap_symbol not in ex.markets:
            return None
        fr = ex.fetch_funding_rate(swap_symbol)
        return fr.get("fundingRate")
    except Exception:
        return None


def funding_rate_ok(exchange_id: str, symbol: str, bias: str,
                     max_annualized_pct: float = 20.0) -> bool:
    """펀딩비가 과열(쏠림)된 방향으로는 진입하지 않도록 걸러냄.
    - 8시간마다 정산 → 하루 3회 → 연 1095회
    - 롱인데 펀딩비가 크게 플러스(롱 과열, 내가 숏에게 계속 돈을 냄) → 제외
    - 숏인데 펀딩비가 크게 마이너스(숏 과열, 내가 롱에게 계속 돈을 냄) → 제외
    데이터가 없으면(현물 등) 통과시킴 — 무기한 선물이 아니면 해당 없음."""
    rate = get_funding_rate(exchange_id, symbol)
    if rate is None:
        return True
    annualized_pct = rate * 1095 * 100

    if "long" in bias and annualized_pct > max_annualized_pct:
        return False
    if "short" in bias and annualized_pct < -max_annualized_pct:
        return False
    return True


def atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def calculate_capture_ratios(coin_df: pd.DataFrame, btc_df: pd.DataFrame, lookback: int = 100) -> Dict:
    """지수(BTC) 상승/하락 국면에서 코인이 얼마나 비대칭적으로 반응하는지 측정.
    - 상승 포착률(up_capture) > 1  : 지수 오를 때 그 이상으로 따라 오름 (베타가 큼)
    - 하락 포착률(down_capture) < 1 (0에 가깝거나 음수면 더 좋음) : 지수 빠질 때 덜 빠지거나 버팀
    - asymmetry = up_capture - down_capture : 클수록 '오를 땐 크게, 빠질 땐 작게'인 이상적 비대칭

    ⚠️ 표본이 적으면(횡보장이 길게 이어져 상승/하락 구간 수가 적으면) 신뢰도가 떨어집니다.
    최소 각 구간 10봉 이상 확보되지 않으면 None으로 처리해서 오판을 막습니다."""
    n = min(len(coin_df), len(btc_df), lookback)
    if n < 30:
        return {"up_capture": None, "down_capture": None, "asymmetry": None}

    coin_ret = coin_df["close"].tail(n).pct_change().dropna()
    btc_ret = btc_df["close"].tail(n).pct_change().dropna()
    m = min(len(coin_ret), len(btc_ret))
    coin_ret, btc_ret = coin_ret.iloc[-m:].reset_index(drop=True), btc_ret.iloc[-m:].reset_index(drop=True)

    up_mask = btc_ret > 0
    down_mask = btc_ret < 0

    if up_mask.sum() < 10 or down_mask.sum() < 10:
        return {"up_capture": None, "down_capture": None, "asymmetry": None}

    btc_up_avg = btc_ret[up_mask].mean()
    btc_down_avg = btc_ret[down_mask].mean()
    coin_up_avg = coin_ret[up_mask].mean()
    coin_down_avg = coin_ret[down_mask].mean()

    up_capture = coin_up_avg / btc_up_avg if btc_up_avg != 0 else None
    down_capture = coin_down_avg / btc_down_avg if btc_down_avg != 0 else None

    if up_capture is None or down_capture is None:
        return {"up_capture": up_capture, "down_capture": down_capture, "asymmetry": None}

    return {"up_capture": float(up_capture), "down_capture": float(down_capture),
             "asymmetry": float(up_capture - down_capture)}


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 50, lookback: int = 100) -> Dict:
    """POC(Point of Control)·Value Area 근사 계산.
    - 각 캔들의 거래량을 그 캔들의 고가~저가 구간에 균등 분산시켜 가격대별 거래량을 누적
    - 거래량이 가장 많이 쌓인 구간 = POC (매물대 핵심, '가격 자석' 역할)
    - POC를 중심으로 누적거래량 70%에 도달할 때까지 확장한 상/하단 = Value Area High/Low
      (그 안쪽은 '시장이 공정가로 받아들인 가격대', 바깥쪽은 '거부된 가격대'로 해석)
    ⚠️ 캔들(OHLCV) 데이터 기반 근사치입니다. 틱/오더북 데이터 기반 진짜 볼륨프로파일보다는
    정밀도가 떨어지지만, 실전에서 널리 쓰이는 근사 방식입니다."""
    recent = df.tail(lookback)
    price_min, price_max = recent["low"].min(), recent["high"].max()
    if price_max <= price_min:
        return {"poc": None, "vah": None, "val": None}

    bins = np.linspace(price_min, price_max, num_bins + 1)
    vol_by_bin = np.zeros(num_bins)

    for _, row in recent.iterrows():
        lo_idx = np.searchsorted(bins, row["low"], side="right") - 1
        hi_idx = np.searchsorted(bins, row["high"], side="right") - 1
        lo_idx, hi_idx = max(0, lo_idx), min(num_bins - 1, hi_idx)
        span = max(hi_idx - lo_idx + 1, 1)
        vol_by_bin[lo_idx:hi_idx + 1] += row["volume"] / span

    poc_idx = int(np.argmax(vol_by_bin))
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

    total_vol = vol_by_bin.sum()
    target = total_vol * 0.7
    lo, hi = poc_idx, poc_idx
    acc = vol_by_bin[poc_idx]
    while acc < target and (lo > 0 or hi < num_bins - 1):
        expand_lo = vol_by_bin[lo - 1] if lo > 0 else -1
        expand_hi = vol_by_bin[hi + 1] if hi < num_bins - 1 else -1
        if expand_hi >= expand_lo:
            hi = min(hi + 1, num_bins - 1)
            acc += vol_by_bin[hi]
        else:
            lo = max(lo - 1, 0)
            acc += vol_by_bin[lo]

    return {"poc": float(poc_price), "val": float(bins[lo]), "vah": float(bins[hi + 1])}


def near_level(price: float, level: Optional[float], tolerance_atr: float, a: float) -> bool:
    if level is None or a <= 0:
        return False
    return abs(price - level) <= tolerance_atr * a


def compute_sideways_lean(df: pd.DataFrame, htf_trend: RegimeType, lookback: int = 30,
                           weights: tuple = (0.5, 0.3, 0.2)) -> Dict:
    """횡보장에서 다음 방향에 대한 '확률적 기울기'를 계산.
    ⚠️ 이건 확정 예측이 아니라 여러 객관적 근거를 가중평균한 확률적 기울기입니다.
    score는 -1(하락 우세)~+1(상승 우세), |score| < 0.3이면 '중립'으로 판단해 방향을 강제하지 않습니다."""
    recent = df.tail(lookback)
    htf_component = {"uptrend": 1.0, "downtrend": -1.0, "sideways": 0.0}[htf_trend]

    half = len(recent) // 2
    first_half, second_half = recent.iloc[:half], recent.iloc[half:]
    if len(first_half) and len(second_half):
        low_diff = second_half["low"].min() - first_half["low"].min()    # 저점 상승폭
        high_diff = second_half["high"].max() - first_half["high"].max()  # 고점 상승폭
        rng = max(recent["high"].max() - recent["low"].min(), 1e-9)
        structure_component = float(np.clip(((low_diff - high_diff) / 2) / rng, -1, 1))
    else:
        structure_component = 0.0

    up_bars = recent[recent["close"] > recent["open"]]
    down_bars = recent[recent["close"] < recent["open"]]
    if len(up_bars) and len(down_bars):
        up_vol, down_vol = up_bars["volume"].mean(), down_bars["volume"].mean()
        accum_component = float(np.clip((up_vol - down_vol) / max(up_vol + down_vol, 1e-9), -1, 1))
    else:
        accum_component = 0.0

    w_htf, w_struct, w_accum = weights
    score = w_htf * htf_component + w_struct * structure_component + w_accum * accum_component
    if score > 0.3:
        label = "상승쪽 우세"
    elif score < -0.3:
        label = "하락쪽 우세"
    else:
        label = "중립(방향성 불명확)"

    return {"score": score, "label": label, "htf": htf_component,
            "structure": structure_component, "accumulation": accum_component}


def build_setup(symbol: str, exchange_id: str, df: pd.DataFrame, btc_df: pd.DataFrame,
                regime: RegimeType, htf_trend: RegimeType = "sideways") -> Optional[CoinSetup]:
    """추격(chase) 대신 '되돌림 지정가 진입'을 기본으로 계산합니다.
    - 상승국면: 최근 임펄스 이후 눌림목(불리시 오더블록 또는 EMA20)까지 되돌아왔을 때만 롱 진입가 부여
    - 하락국면: 반등 후 베어리시 오더블록 또는 EMA20까지 되돌아왔을 때만 숏 진입가 부여
    - 현재가가 아직 그 되돌림 구간에 도달 못 했으면 is_chase=True로 표시해서
      '지금 들어가면 추격'이라는 걸 명시적으로 경고합니다."""
    if df is None or len(df) < 40:
        return None

    price = df["close"].iloc[-1]
    vol_avg20 = df["volume"].tail(20).mean()
    vol_now = df["volume"].iloc[-1]
    rel_vol = vol_now / vol_avg20 if vol_avg20 else 1
    rs = relative_strength_vs_btc(df, btc_df)
    ob = detect_order_block(df)
    a = atr(df)
    ema20 = ema(df["close"], 20).iloc[-1]
    vp = calculate_volume_profile(df)
    cap = calculate_capture_ratios(df, btc_df)

    swing_high = df["high"].tail(20).max()
    swing_low = df["low"].tail(20).min()

    if regime == "uptrend":
        if rel_vol < 1.3:  # 상승국면은 임펄스(거래량 급증) 확인이 전제조건
            return None
        if rs <= 0:
            return None
        # 되돌림 진입가: 불리시 오더블록 상단이 있으면 그 값, 없으면 EMA20을 대용
        pullback_entry = ob["bullish_ob"]["high"] if ob["bullish_ob"] else ema20
        sl = (ob["bullish_ob"]["low"] if ob["bullish_ob"] else pullback_entry - 1.5 * a)
        # 오더블록 없이 EMA20을 대용 진입가로 쓰면 손절(1.5ATR)=목표(1.5ATR)라 손익비가 항상 1.0이 되어
        # RR 필터에서 전부 탈락하므로, 이 경우 목표를 2.4ATR로 넓혀 손익비 1.6을 확보
        tp1 = pullback_entry + (1.5 * a if ob["bullish_ob"] else 2.4 * a)
        tp2 = swing_high * 1.02
        rr = (tp1 - pullback_entry) / max(pullback_entry - sl, 1e-9)
        is_chase = price > pullback_entry * 1.005  # 현재가가 진입가보다 훨씬 위면 = 추격 구간
        # 되돌림 진입가가 POC/Value Area Low(매물대 하단=지지)와도 겹치면 컨플루언스 신뢰도 상향
        poc_conf = near_level(pullback_entry, vp["poc"], 0.5, a) or near_level(pullback_entry, vp["val"], 0.5, a)
        conf_tag = " + POC/매물대 지지 겹침(컨플루언스)" if poc_conf else ""
        asym = cap["asymmetry"]
        asym_tag = f", 비대칭점수 {asym:+.2f}(상승↑{cap['up_capture']:.2f}/하락↓{cap['down_capture']:.2f})" if asym is not None else ""
        note = (f"상대강도 {rs:.1f}%, 거래량 {rel_vol:.1f}배 임펄스 확인{conf_tag}{asym_tag} → "
                f"{'⚠️추격 주의: 되돌림 대기' if is_chase else '되돌림 진입가 도달'}")
        return CoinSetup(symbol, exchange_id, "long", note,
                          pullback_entry, price, tp1, tp2, sl, rr, is_chase, poc_conf, rs, asym)

    if regime == "downtrend":
        if rel_vol < 1.3:  # 하락국면은 임펄스(거래량 급증) 확인이 전제조건
            return None
        if rs >= 0:
            return None
        pullback_entry = ob["bearish_ob"]["low"] if ob["bearish_ob"] else ema20
        sl = (ob["bearish_ob"]["high"] if ob["bearish_ob"] else pullback_entry + 1.5 * a)
        tp1 = pullback_entry - (1.5 * a if ob["bearish_ob"] else 2.4 * a)
        tp2 = swing_low * 0.98
        rr = (pullback_entry - tp1) / max(sl - pullback_entry, 1e-9)
        is_chase = price < pullback_entry * 0.995  # 현재가가 진입가보다 훨씬 아래면 = 추격 구간
        # 되돌림 진입가가 POC/Value Area High(매물대 상단=저항)와도 겹치면 컨플루언스 신뢰도 상향
        poc_conf = near_level(pullback_entry, vp["poc"], 0.5, a) or near_level(pullback_entry, vp["vah"], 0.5, a)
        conf_tag = " + POC/매물대 저항 겹침(컨플루언스)" if poc_conf else ""
        asym = cap["asymmetry"]
        asym_tag = f", 비대칭점수 {asym:+.2f}(상승↑{cap['up_capture']:.2f}/하락↓{cap['down_capture']:.2f})" if asym is not None else ""
        note = (f"상대강도 {rs:.1f}%, 거래량 {rel_vol:.1f}배 임펄스 확인{conf_tag}{asym_tag} → "
                f"{'⚠️추격 주의: 반등 대기' if is_chase else '반등 진입가 도달'}")
        return CoinSetup(symbol, exchange_id, "short", note,
                          pullback_entry, price, tp1, tp2, sl, rr, is_chase, poc_conf, rs, asym)

    # --- 횡보(sideways): 두 가지 경우를 분리해서 처리 ---
    # (A) 거래량 급증 + 박스 경계 = '돌파 시도' → 기존처럼 리테스트 대기(방향 예측 안 함)
    # (B) 거래량 평소 이하 + 박스 경계 = '조용한 되돌림' → 레인지 평균회귀(페이드) 후보
    #     (횡보장의 대부분은 거래량이 오히려 잠잠한 구간이라, 이 케이스를 안 다루면
    #      횡보장에서 신호가 거의 안 나오는 문제가 생김 — 실제로 발견된 구조적 문제였음)
    range_high, range_low = swing_high, swing_low
    box_size = range_high - range_low
    lean = compute_sideways_lean(df, htf_trend)
    lean_tag = f" [기울기: {lean['label']} (score {lean['score']:+.2f})]"

    if rel_vol >= 1.3:
        if price >= range_high * 0.995:
            if lean["score"] < -0.3:  # 하락 우세 국면에서 상단 돌파 롱은 억제 (역행 확률 낮음)
                return None
            entry = range_high
            long_sl = range_low
            extension = 1.5 if lean["score"] > 0.3 else 0.5  # 기울기가 지지하면 목표 확장
            long_tp1 = range_high + box_size * (0.5 if lean["score"] <= 0.3 else 0.8)
            is_chase = price > range_high * 1.01
            rr = (long_tp1 - entry) / max(entry - long_sl, 1e-9)
            return CoinSetup(symbol, exchange_id, "wait_breakout_long",
                              f"박스 상단 돌파 감지 (거래량 {rel_vol:.1f}배){lean_tag} → "
                              f"{'⚠️추격 주의: 상단 리테스트 대기' if is_chase else '상단 리테스트 구간'}",
                              entry, price, long_tp1, range_high + box_size * extension, long_sl, rr, is_chase,
                              rs=rs, asymmetry=cap["asymmetry"])

        if price <= range_low * 1.005:
            if lean["score"] > 0.3:  # 상승 우세 국면에서 하단 이탈 숏은 억제
                return None
            entry = range_low
            short_sl = range_high
            extension = 1.5 if lean["score"] < -0.3 else 0.5
            short_tp1 = range_low - box_size * (0.5 if lean["score"] >= -0.3 else 0.8)
            is_chase = price < range_low * 0.99
            rr = (entry - short_tp1) / max(short_sl - entry, 1e-9)
            return CoinSetup(symbol, exchange_id, "wait_breakout_short",
                              f"박스 하단 이탈 감지 (거래량 {rel_vol:.1f}배){lean_tag} → "
                              f"{'⚠️추격 주의: 하단 리테스트 대기' if is_chase else '하단 리테스트 구간'}",
                              entry, price, short_tp1, range_low - box_size * extension, short_sl, rr, is_chase,
                              rs=rs, asymmetry=cap["asymmetry"])
        return None

    # 거래량이 평소 이하(조용한 구간)일 때만 레인지 페이드 후보로 검토
    if rel_vol <= 1.1:
        fade_sl_buffer = 0.5 * a

        if price <= range_low * 1.01:  # 박스 하단 근처 = 지지 매수(롱) 후보
            if lean["score"] < -0.3:  # 하락 우세 국면에서 지지 매수는 억제 (지지 붕괴 확률 높음)
                return None
            entry = price
            sl = range_low - fade_sl_buffer
            tp1 = vp["poc"] if vp["poc"] and vp["poc"] > entry else (range_low + box_size * 0.5)
            tp2 = range_high * (1.02 if lean["score"] > 0.3 else 1.0)  # 기울기 지지 시 목표 확장
            rr = (tp1 - entry) / max(entry - sl, 1e-9)
            poc_conf = near_level(entry, vp["val"], 0.5, a)
            conf_tag = " + Value Area Low 겹침" if poc_conf else ""
            note = (f"박스 하단 지지 + 거래량 평소 이하({rel_vol:.1f}배, 조용한 되돌림){conf_tag}{lean_tag} "
                    f"→ 레인지 평균회귀 매수")
            return CoinSetup(symbol, exchange_id, "range_fade_long", note,
                              entry, price, tp1, tp2, sl, rr, False, poc_conf, rs, cap["asymmetry"])

        if price >= range_high * 0.99:  # 박스 상단 근처 = 저항 매도(숏) 후보
            if lean["score"] > 0.3:  # 상승 우세 국면에서 저항 매도는 억제 (저항 돌파 확률 높음)
                return None
            entry = price
            sl = range_high + fade_sl_buffer
            tp1 = vp["poc"] if vp["poc"] and vp["poc"] < entry else (range_high - box_size * 0.5)
            tp2 = range_low * (0.98 if lean["score"] < -0.3 else 1.0)
            rr = (entry - tp1) / max(sl - entry, 1e-9)
            poc_conf = near_level(entry, vp["vah"], 0.5, a)
            conf_tag = " + Value Area High 겹침" if poc_conf else ""
            note = (f"박스 상단 저항 + 거래량 평소 이하({rel_vol:.1f}배, 조용한 되돌림){conf_tag}{lean_tag} "
                    f"→ 레인지 평균회귀 매도")
            return CoinSetup(symbol, exchange_id, "range_fade_short", note,
                              entry, price, tp1, tp2, sl, rr, False, poc_conf, rs, cap["asymmetry"])

    return None  # 박스 중간권이거나 애매한 거래량 구간이면 신호 없음


BITGET_ONLY = True  # True면 Bitget USDT 무기한 선물이 있는 코인만 추천 (Bitget에서 거래하므로)
_EXCLUDED_BASES = {"USDC", "FDUSD", "TUSD", "DAI", "USDE", "USDD", "PYUSD", "BUSD", "USD1", "USDP", "EUR"}


def _is_excluded_symbol(symbol: str) -> bool:
    """스테이블코인·레버리지 토큰은 추천 대상에서 제외."""
    base = symbol.split("/")[0]
    return base in _EXCLUDED_BASES or base.endswith(("3L", "3S", "5L", "5S"))


def bitget_perp_symbols() -> set:
    """Bitget USDT-M 무기한 선물 심볼 집합(예: 'SOL/USDT:USDT'). 조회 실패 시 빈 집합."""
    try:
        ex = _get_ex("bitget")
        ex.load_markets()
        return {m["symbol"] for m in ex.markets.values()
                if m.get("swap") and m.get("linear") and m.get("active", True) and m.get("quote") == QUOTE}
    except Exception as e:
        print(f"[warn] Bitget 선물 목록 조회 실패: {e}")
        return set()


def build_universe() -> Dict[str, str]:
    """거래소별 거래량 상위 코인을 합쳐 중복 제거. 같은 코인은 EXCHANGES 순서상 먼저 나온
    거래소(기본 Bitget)의 캔들을 사용 → 실제 거래하는 곳의 가격 기준으로 분석하고 API 호출도 절약."""
    universe: Dict[str, str] = {}
    for exchange_id in EXCHANGES:
        try:
            symbols = get_top_volume_symbols(exchange_id, TOP_N_BY_VOLUME)
        except Exception as e:
            print(f"[warn] {exchange_id} 심볼 조회 실패: {e}")
            continue
        for sym in symbols:
            if not _is_excluded_symbol(sym):
                universe.setdefault(sym, exchange_id)
    return universe


def screen_market(regime: RegimeType, progress_cb=None) -> List[CoinSetup]:
    btc_df = fetch_btc_df()
    universe = build_universe()
    perps = bitget_perp_symbols() if BITGET_ONLY else set()
    if BITGET_ONLY and not perps:
        print("[warn] Bitget 선물 목록을 못 가져와 '선물 거래 가능 여부' 필터를 건너뜁니다.")
    fund_ex = "bitget" if perps else None
    setups: List[CoinSetup] = []
    items = list(universe.items())

    for idx, (symbol, exchange_id) in enumerate(items):
        if progress_cb:
            progress_cb(idx, len(items), symbol)
        swap_symbol = f"{symbol}:{QUOTE}"
        if BITGET_ONLY and perps and swap_symbol not in perps:
            print(f"[skip] {symbol}: Bitget 선물 미지원")
            continue
        try:
            df = fetch_ohlcv(exchange_id, symbol)
            if df is None:
                continue
            # 일봉(HTF)은 횡보 기울기 계산에만 먼저 필요 — 그 외 국면은 신호가 난 뒤에만 조회(API 절약)
            htf_trend = get_htf_trend(exchange_id, symbol) if regime == "sideways" else None
            setup = build_setup(symbol, exchange_id, df, btc_df, regime, htf_trend or "sideways")
            if not setup:
                continue

            if setup.bias in ("long", "short"):
                if htf_trend is None:
                    htf_trend = get_htf_trend(exchange_id, symbol)
                if setup.bias == "long" and htf_trend == "downtrend":
                    print(f"[skip] {symbol}: 일봉 추세 하락인데 4h 롱 신호 — 상위추세 역행이라 제외")
                    continue
                if setup.bias == "short" and htf_trend == "uptrend":
                    print(f"[skip] {symbol}: 일봉 추세 상승인데 4h 숏 신호 — 상위추세 역행이라 제외")
                    continue

            spread = get_spread_pct(exchange_id, symbol)
            if spread is not None and spread > 0.3:
                print(f"[skip] {symbol}: 스프레드 {spread:.2f}% — 유동성 부족으로 슬리피지 리스크 커서 제외")
                continue

            if not funding_rate_ok(fund_ex or exchange_id, symbol, setup.bias):
                print(f"[skip] {symbol}: 펀딩비 과열 방향이라 제외 ({setup.bias})")
                continue

            setup.bitget_perp = (swap_symbol in perps) if perps else None
            setups.append(setup)
        except Exception as e:  # 코인 하나의 실패가 전체 스캔을 멈추지 않도록
            print(f"[warn] {symbol} 분석 실패: {e}")

    if progress_cb:
        progress_cb(len(items), len(items), "")

    # 리스크 대비 보상(RR) 최소 기준 — POC/매물대 컨플루언스가 있으면 1.3, 없으면 1.5
    setups = [x for x in setups if x.rr_ratio >= (1.3 if x.poc_confluence else 1.5)]
    setups.sort(key=lambda x: x.rr_ratio, reverse=True)
    return setups


def run_analysis(risk_cfg: Optional[RiskConfig] = None, progress_cb=None) -> Dict:
    """전체 파이프라인(국면 판단 → 스캔 → 상관 제한)을 실행하고 결과를 dict로 반환.
    (웹 화면(app.py)이 사용. main()은 같은 내용을 콘솔에 출력하는 버전)"""
    if risk_cfg is None:
        risk_cfg = RiskConfig(account_balance=1000, risk_per_trade_pct=1.0, max_concurrent_setups=3)
    regime = determine_overall_regime()
    breaker = circuit_breaker_triggered(risk_cfg)
    all_setups: List[CoinSetup] = []
    if not breaker:
        all_setups = screen_market(regime.overall, progress_cb)
    setups = cap_correlated_exposure(all_setups, risk_cfg) if all_setups else []
    return {"regime": regime, "breaker": breaker, "setups": setups, "all_setups": all_setups,
            "risk_cfg": risk_cfg, "asof": pd.Timestamp.now()}


# --------------------------------------------------------------------------
# 5.1 실전 로직 기반 WFO — build_setup()을 과거 데이터에 그대로 재사용
# --------------------------------------------------------------------------

def fetch_extended_ohlcv(exchange_id: str, symbol: str, timeframe: str = TIMEFRAME,
                          total_bars: int = 3000) -> pd.DataFrame:
    """긴 과거 기간을 페이지네이션으로 수집. WFO는 데이터가 많을수록 신뢰도가 올라갑니다."""
    if ccxt is None:
        raise RuntimeError("ccxt가 설치되어 있지 않습니다.")
    ex = _get_ex(exchange_id)
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - total_bars * tf_ms
    rows: List = []
    while len(rows) < total_bars:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.drop_duplicates(subset="ts").reset_index(drop=True)
    return df.tail(total_bars).reset_index(drop=True)


def htf_trend_at(daily_df: pd.DataFrame, current_ts) -> RegimeType:
    """현재 4h 봉 시점 기준, 그 이전에 '마감된' 일봉들만으로 HTF 추세 판단
    (미래참조 방지 — 당일 진행 중인 일봉은 절대 포함하지 않음)."""
    closed = daily_df[daily_df["ts"] < pd.Timestamp(current_ts).normalize()]
    if len(closed) < 60:
        return "sideways"
    return classify_price_trend(closed)


def compute_trade_R(direction: str, entry: float, sl: float, exit_price: float,
                     holding_bars: int, fee_pct: float = 0.05, slippage_pct: float = 0.03,
                     funding_pct_per_8h: float = 0.01, bars_per_8h: float = 2) -> float:
    """손익을 'R 배수'(최초 리스크 대비 몇 배)로 환산. 계좌 크기와 무관하게 전략 자체의
    품질을 비교할 수 있어서 WFO/기대값 계산에 표준적으로 쓰입니다. 수수료·슬리피지·펀딩비를
    전부 비용으로 차감한 '순(net) R'입니다."""
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return 0.0
    raw = (exit_price - entry) if direction == "long" else (entry - exit_price)
    cost_price = entry * 2 * (fee_pct + slippage_pct) / 100          # 왕복 수수료+슬리피지
    funding_price = entry * (funding_pct_per_8h / 100) * (holding_bars / bars_per_8h)
    net = raw - cost_price - funding_price
    return net / risk_per_unit


def simulate_strategy_history(df: pd.DataFrame, btc_df: pd.DataFrame, daily_df: pd.DataFrame,
                               pending_expiry_bars: int = 15, max_hold_bars: int = 60,
                               min_lookback: int = 80) -> List[float]:
    """build_setup()을 과거 매 봉마다 그대로 호출해서, 실제 라이브 로직과 100% 동일한
    기준으로 신호를 만들고, 그 신호를 봉 단위로 앞으로 재생하며 체결/청산을 시뮬레이션합니다.

    - 추격 구간(is_chase=True) 신호는 'pending' 지정가 주문으로 등록 → 이후 봉에서
      실제로 그 가격에 닿아야 체결(라이브에서 하려는 것과 동일한 동작)
    - 체결 전에 SL 레벨이 종가 기준으로 붕괴되면 그 지정가 주문은 무효화(취소)
    - 체결 후에는 SL/TP1 중 먼저 닿는 쪽으로 청산, max_hold_bars 넘으면 종가 청산
    - 한 번에 하나의 포지션만 시뮬레이션 (중첩 방지로 결과를 보수적으로 유지)
    """
    results: List[float] = []
    i = min_lookback
    pending: Optional[Dict] = None
    open_trade: Optional[Dict] = None

    while i < len(df):
        row = df.iloc[i]

        if open_trade:
            direction = open_trade["direction"]
            holding_bars = i - open_trade["entry_idx"]
            if direction == "long":
                hit_sl = row["low"] <= open_trade["sl"]
                hit_tp = row["high"] >= open_trade["tp1"]
            else:
                hit_sl = row["high"] >= open_trade["sl"]
                hit_tp = row["low"] <= open_trade["tp1"]

            if hit_sl:  # SL을 TP보다 먼저 체크 = 보수적 가정 (같은 봉에 둘 다 닿았을 경우)
                results.append(compute_trade_R(direction, open_trade["entry"], open_trade["sl"],
                                                open_trade["sl"], holding_bars))
                open_trade = None
            elif hit_tp:
                results.append(compute_trade_R(direction, open_trade["entry"], open_trade["sl"],
                                                open_trade["tp1"], holding_bars))
                open_trade = None
            elif holding_bars >= max_hold_bars:
                results.append(compute_trade_R(direction, open_trade["entry"], open_trade["sl"],
                                                row["close"], holding_bars))
                open_trade = None
            i += 1
            continue

        if pending:
            direction = pending["direction"]
            filled = row["low"] <= pending["entry"] if direction == "long" else row["high"] >= pending["entry"]
            invalidated = row["close"] < pending["sl"] if direction == "long" else row["close"] > pending["sl"]
            if filled:
                open_trade = {**pending, "entry_idx": i}
                pending = None
            elif invalidated or i >= pending["expiry_idx"]:
                pending = None
            i += 1
            continue

        # 새 셋업 탐색 — 지금까지의 데이터만 사용 (미래 데이터 절대 참조 안 함)
        lo = max(0, i + 1 - OHLCV_LIMIT)  # 라이브(fetch_ohlcv limit=200)와 동일한 윈도우
        window_df = df.iloc[lo:i + 1]
        window_btc = btc_df.iloc[lo:i + 1]
        regime_i = classify_price_trend(window_btc)
        htf_i = htf_trend_at(daily_df, row["ts"])  # 횡보 기울기 계산에도 쓰이므로 build_setup 호출 전에 계산
        setup = build_setup(f"bt_{i}", "backtest", window_df, window_btc, regime_i, htf_i)

        if setup:
            direction = "long" if setup.bias in ("long", "wait_breakout_long", "range_fade_long") else "short"
            # 추세추종(long/short)만 HTF 역행 필터 적용 — 횡보(wait_breakout_*, range_fade_*)는
            # build_setup 내부에서 이미 htf_trend를 기울기 계산에 반영해 자체적으로 걸러졌으므로 중복 필터 안 함
            aligned = True
            if setup.bias in ("long", "short"):
                aligned = (setup.bias == "long" and htf_i != "downtrend") or \
                          (setup.bias == "short" and htf_i != "uptrend")
            if aligned:
                cand = {"direction": direction, "entry": setup.entry_price,
                        "sl": setup.sl, "tp1": setup.tp1}
                if setup.is_chase:
                    pending = {**cand, "expiry_idx": i + pending_expiry_bars}
                else:
                    open_trade = {**cand, "entry_idx": i}
        i += 1

    return results


def backtest_wfo_real(exchange_id: str, symbol: str, timeframe: str = TIMEFRAME,
                       total_bars: int = 2000, train_bars: int = 1000,
                       test_bars: int = 200) -> pd.DataFrame:
    """실제 build_setup() 로직으로 구간별 워크포워드 검증.
    ⚠️ 지금 build_setup()에는 그리드서치할 자유 파라미터가 없으므로(임계값이 코드에 고정),
    이건 엄밀히는 '최적화 후 검증'이 아니라 '고정 규칙의 롤링 아웃오브샘플 검증'입니다.
    (오히려 파라미터를 데이터에 맞출 기회가 없다는 점에서 과최적화 위험은 더 낮습니다.)

    train_avg_R/test_avg_R가 구간마다 꾸준히 비슷한 부호·크기로 나오면 신뢰할 만한 신호,
    구간마다 들쭉날쭉하거나 test 구간에서 계속 마이너스면 이 전략은 재검토가 필요합니다."""
    df = fetch_extended_ohlcv(exchange_id, symbol, timeframe, total_bars)
    btc_symbol = f"BTC/{QUOTE}"
    btc_df = df.copy() if symbol == btc_symbol else fetch_extended_ohlcv(exchange_id, btc_symbol, timeframe, total_bars)
    daily_bars_needed = total_bars // 6 + 100  # 4h 기준 1일=6봉이므로 대략 환산 + 여유분
    daily_df = fetch_extended_ohlcv(exchange_id, symbol, "1d", daily_bars_needed)

    rows = []
    start = 0
    while start + train_bars + test_bars <= len(df):
        train_df = df.iloc[start:start + train_bars].reset_index(drop=True)
        train_btc = btc_df.iloc[start:start + train_bars].reset_index(drop=True)
        test_df = df.iloc[start + train_bars:start + train_bars + test_bars].reset_index(drop=True)
        test_btc = btc_df.iloc[start + train_bars:start + train_bars + test_bars].reset_index(drop=True)

        train_R = simulate_strategy_history(train_df, train_btc, daily_df)
        test_R = simulate_strategy_history(test_df, test_btc, daily_df)

        rows.append({
            "train_start": train_df["ts"].iloc[0], "train_end": train_df["ts"].iloc[-1],
            "test_start": test_df["ts"].iloc[0], "test_end": test_df["ts"].iloc[-1],
            "train_trades": len(train_R), "train_avg_R": float(np.mean(train_R)) if train_R else 0.0,
            "train_total_R": float(sum(train_R)),
            "test_trades": len(test_R), "test_avg_R": float(np.mean(test_R)) if test_R else 0.0,
            "test_total_R": float(sum(test_R)),
        })
        start += test_bars

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5.15 횡보 '기울기(lean)'의 예측력 검증 — 점수가 실제로 방향을 맞추는가?
# --------------------------------------------------------------------------

def evaluate_lean_predictions(df: pd.DataFrame, btc_df: pd.DataFrame, daily_df: pd.DataFrame,
                               weights: tuple = (0.5, 0.3, 0.2), forward_bars: int = 12,
                               min_lookback: int = 80, threshold: float = 0.3,
                               cost_pct: float = 0.16,
                               ctx: Optional[Dict] = None) -> List[Dict]:
    """BTC 기준 국면이 '횡보'인 시점마다 lean 점수를 계산하고, 강한 기울기(|score|>threshold)가
    나온 경우 이후 forward_bars봉 뒤 실제 수익률과 비교합니다.
    - 겹치는 표본으로 적중률이 부풀려지지 않도록 forward_bars 간격으로만 샘플링
    - hit = 방향 적중(부호), net_ret_pct = 왕복 비용(cost_pct, 수수료+슬리피지 근사) 차감 후 수익률
    - 미래 데이터는 예측 시점 이후 forward_bars 결과 확인에만 사용"""
    records: List[Dict] = []
    ctx = ctx if ctx is not None else {}   # 봉별 (국면, HTF) 캐시 — 가중치를 바꿔 반복 평가할 때 재사용
    i = min_lookback
    while i + forward_bars < len(df):
        key = df["ts"].iloc[i]
        if key not in ctx:
            # 라이브와 동일하게 최근 OHLCV_LIMIT봉만으로 국면 판단 (속도 + 라이브 일관성)
            window_btc = btc_df.iloc[max(0, i + 1 - OHLCV_LIMIT):i + 1]
            regime_i = classify_price_trend(window_btc)
            htf_i = htf_trend_at(daily_df, key) if regime_i == "sideways" else None
            ctx[key] = (regime_i, htf_i)
        regime_i, htf_i = ctx[key]
        if regime_i == "sideways":
            lean = compute_sideways_lean(df.iloc[max(0, i + 1 - OHLCV_LIMIT):i + 1], htf_i, weights=weights)
            if abs(lean["score"]) > threshold:
                direction = 1 if lean["score"] > 0 else -1
                fwd_ret_pct = (df["close"].iloc[i + forward_bars] / df["close"].iloc[i] - 1) * 100
                records.append({
                    "ts": df["ts"].iloc[i], "score": lean["score"], "direction": direction,
                    "fwd_ret_pct": fwd_ret_pct,
                    # 방향 적중은 부호만으로 판정(무엇도 못 맞히면 50% → z검정 기준이 유효).
                    # 비용은 net_ret_pct(평균 순수익률)에서 따로 반영합니다.
                    "hit": direction * fwd_ret_pct > 0,
                    "net_ret_pct": direction * fwd_ret_pct - cost_pct,
                })
                i += forward_bars   # 표본 간 겹침 방지
                continue
        i += 1
    return records


def summarize_lean(records: List[Dict]) -> Dict:
    """적중률과 '우연 대비 유의성'(z-score, 50% 기준)을 요약. |z|<2면 우연과 구분 어렵다는 뜻."""
    n = len(records)
    if n == 0:
        return {"n": 0, "hit_rate": None, "z": None, "avg_net_ret_pct": None}
    hits = sum(1 for r in records if r["hit"])
    hit_rate = hits / n
    z = (hit_rate - 0.5) / math.sqrt(0.25 / n)
    return {"n": n, "hit_rate": hit_rate, "z": z,
            "avg_net_ret_pct": float(np.mean([r["net_ret_pct"] for r in records]))}


def backtest_lean_wfo(exchange_id: str, symbol: str, timeframe: str = TIMEFRAME,
                       total_bars: int = 3000, train_bars: int = 1200, test_bars: int = 400,
                       weight_grid: Optional[List[tuple]] = None, forward_bars: int = 12,
                       min_train_samples: int = 15) -> pd.DataFrame:
    """횡보 기울기 점수의 예측력을 롤링 워크포워드로 검증.
    - weight_grid를 주면: 학습 구간에서 적중률 최고 가중치를 고르고(표본 min_train_samples 이상만),
      다음 test 구간에서 그 가중치로 검증 (진짜 OOS)
    - 안 주면: 기본 가중치(0.5/0.3/0.2) 고정으로 구간별 적중률만 측정
    test_hit_rate가 구간마다 꾸준히 50%를 넘고 z가 커야 의미 있는 엣지입니다."""
    df = fetch_extended_ohlcv(exchange_id, symbol, timeframe, total_bars)
    btc_symbol = f"BTC/{QUOTE}"
    btc_df = df.copy() if symbol == btc_symbol else fetch_extended_ohlcv(exchange_id, btc_symbol, timeframe, total_bars)
    daily_df = fetch_extended_ohlcv(exchange_id, symbol, "1d", total_bars // 6 + 100)
    grid = weight_grid or [(0.5, 0.3, 0.2)]

    rows, start = [], 0
    while start + train_bars + test_bars <= len(df):
        tr = slice(start, start + train_bars)
        te = slice(start + train_bars, start + train_bars + test_bars)
        # 지표 계산에 lookback이 필요하므로 test 구간은 앞선 이력을 포함해 평가하되,
        # 기록은 test 구간 안의 시점만 채택
        def _eval(sl_start, sl_end, w):
            recs = evaluate_lean_predictions(df.iloc[:sl_end].reset_index(drop=True),
                                             btc_df.iloc[:sl_end].reset_index(drop=True),
                                             daily_df, weights=w, forward_bars=forward_bars,
                                             min_lookback=max(80, sl_start))
            return recs

        best_w, best_hit, best_train = grid[0], -1.0, {"n": 0}
        for w in grid:
            s = summarize_lean(_eval(tr.start, tr.stop, w))
            if s["n"] >= min_train_samples and s["hit_rate"] is not None and s["hit_rate"] > best_hit:
                best_w, best_hit, best_train = w, s["hit_rate"], s
        if best_hit < 0:
            best_train = summarize_lean(_eval(tr.start, tr.stop, best_w))

        test_s = summarize_lean(_eval(te.start, te.stop, best_w))
        rows.append({
            "test_start": df["ts"].iloc[te.start], "test_end": df["ts"].iloc[te.stop - 1],
            "weights": best_w, "train_n": best_train.get("n"), "train_hit": best_train.get("hit_rate"),
            "test_n": test_s["n"], "test_hit": test_s["hit_rate"], "test_z": test_s["z"],
            "test_avg_net_ret_pct": test_s["avg_net_ret_pct"],
        })
        start += test_bars
    return pd.DataFrame(rows)


def pooled_lean_evaluation(exchange_id: str, symbols: List[str], timeframe: str = TIMEFRAME,
                            total_bars: int = 3000, weight_grid: Optional[List[tuple]] = None,
                            split_ratio: float = 0.6, forward_bars: int = 12) -> pd.DataFrame:
    """여러 코인의 기울기 신호를 합쳐서 평가 (코인 1개는 표본이 너무 적어 통계적으로 무의미).
    - 각 코인의 앞쪽 split_ratio 구간 = 학습(가중치 선택), 뒤쪽 = 검증(OOS)
    - 가중치는 '학습 풀 전체'에서 적중률 최고인 조합 1개를 고르고, 검증 풀에서 그대로 평가
    - 결과의 n과 z를 반드시 확인: n이 100 미만이거나 |z|<2면 '엣지 있다'고 결론내리지 마세요."""
    grid = weight_grid or [(0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.3, 0.4, 0.3), (0.34, 0.33, 0.33)]
    btc_symbol = f"BTC/{QUOTE}"
    btc_full = fetch_extended_ohlcv(exchange_id, btc_symbol, timeframe, total_bars)

    datasets = []
    for sym in symbols:
        try:
            d = fetch_extended_ohlcv(exchange_id, sym, timeframe, total_bars)
            daily = fetch_extended_ohlcv(exchange_id, sym, "1d", total_bars // 6 + 100)
            m = min(len(d), len(btc_full))
            datasets.append((sym, d.tail(m).reset_index(drop=True),
                             btc_full.tail(m).reset_index(drop=True), daily, {}))
        except Exception as e:
            print(f"[warn] {sym} 수집 실패: {e}")

    def _pool(part: str, w: tuple) -> List[Dict]:
        pool: List[Dict] = []
        for sym, d, b, daily, ctx in datasets:
            cut = int(len(d) * split_ratio)
            if part == "train":
                dd, bb = d.iloc[:cut].reset_index(drop=True), b.iloc[:cut].reset_index(drop=True)
                recs = evaluate_lean_predictions(dd, bb, daily, weights=w, forward_bars=forward_bars, ctx=ctx)
            else:
                # 검증 구간: 앞선 이력은 지표 계산에만 쓰고, 기록은 cut 이후 시점만 채택
                recs = evaluate_lean_predictions(d, b, daily, weights=w, forward_bars=forward_bars,
                                                 min_lookback=max(80, cut), ctx=ctx)
            pool += recs
        return pool

    rows = []
    for w in grid:
        tr, te = summarize_lean(_pool("train", w)), summarize_lean(_pool("test", w))
        rows.append({"weights": w, "train_n": tr["n"], "train_hit": tr["hit_rate"],
                     "test_n": te["n"], "test_hit": te["hit_rate"], "test_z": te["z"],
                     "test_avg_net_ret_pct": te["avg_net_ret_pct"]})
    out = pd.DataFrame(rows)
    # 학습 적중률 기준 선택(검증 성과로 고르면 안 됨 — 검증 데이터 오염)
    if out["train_hit"].notna().any():
        out["selected_on_train"] = out["train_hit"] == out["train_hit"].max()
    return out


# --------------------------------------------------------------------------
# 5.2 (참고용) 단순 예시 전략 WFO — 실제 추천 로직과 무관한 EMA크로스 뼈대.
#     새로 만든 backtest_wfo_real()을 실전 검증에 사용하세요. 이건 WFO 매커니즘
#     자체를 이해하기 위한 최소 예시로만 남겨둡니다.
# --------------------------------------------------------------------------

def backtest_wfo(df: pd.DataFrame, train_window: int = 500, test_window: int = 100,
                  param_grid: Optional[List[Dict]] = None) -> pd.DataFrame:
    """워크포워드 최적화 뼈대.
    - train_window 구간에서 파라미터(예: EMA fast/slow, ADX threshold)를 그리드서치로 최적화
    - 바로 다음 test_window 구간에서 '한 번도 본 적 없는 데이터'로 검증
    - 이 과정을 데이터 끝까지 롤링 반복 → 구간별 성과를 모아야 '진짜 엣지'인지 판단 가능

    ⚠️ 이 함수는 뼈대만 제공합니다. 실제 전략 로직(진입/청산 규칙)을 채워 넣고,
    거래비용·슬리피지·펀딩비를 반드시 반영해야 현실적인 결과가 나옵니다.
    과최적화 방지를 위해 파라미터 그리드는 최소한으로 유지하세요.
    """
    if param_grid is None:
        param_grid = [
            {"ema_fast": 20, "ema_slow": 50, "adx_th": 15},
            {"ema_fast": 50, "ema_slow": 200, "adx_th": 20},
        ]

    results = []
    start = 0
    while start + train_window + test_window <= len(df):
        train = df.iloc[start:start + train_window]
        test = df.iloc[start + train_window:start + train_window + test_window]

        best_param, best_score = None, -math.inf
        for params in param_grid:
            score = _evaluate_strategy(train, params)
            if score > best_score:
                best_score, best_param = score, params

        oos_score = _evaluate_strategy(test, best_param)
        results.append({
            "train_start": train["ts"].iloc[0], "train_end": train["ts"].iloc[-1],
            "test_start": test["ts"].iloc[0], "test_end": test["ts"].iloc[-1],
            "best_param": best_param, "in_sample_score": best_score, "out_of_sample_score": oos_score,
        })
        start += test_window  # 롤링

    return pd.DataFrame(results)


def _evaluate_strategy(df: pd.DataFrame, params: Dict,
                        fee_pct: float = 0.05, slippage_pct: float = 0.03,
                        funding_pct_per_8h: float = 0.01) -> float:
    """예시 전략 평가 함수: EMA 골든/데드크로스 + ADX 필터.
    ⚠️ 실전에서는 여기를 본인의 실제 전략 로직으로 교체하세요.

    비용을 반드시 반영합니다 (기본값은 대략적인 예시이며 실제 거래소 수수료로 교체하세요):
    - fee_pct: 편도 거래 수수료 (%) — 왕복이면 2번 발생
    - slippage_pct: 체결 슬리피지 (%) — 시장가 진입/청산 시 발생
    - funding_pct_per_8h: 무기한 선물 보유 시 8시간마다 발생하는 펀딩비 (%)
      → 포지션을 오래 들고 있을수록 비용이 누적되므로, 봉 하나(TIMEFRAME)당
        경과 시간에 비례해 비용을 차감합니다."""
    fast = ema(df["close"], params["ema_fast"])
    slow = ema(df["close"], params["ema_slow"])
    strength = adx(df) if len(df) > params["ema_slow"] else 0

    bars_per_8h = 2  # TIMEFRAME="4h" 기준 8시간 = 2봉. 다른 타임프레임 쓰면 이 값을 바꾸세요.
    per_bar_funding_cost = funding_pct_per_8h / bars_per_8h / 100

    position = 0
    pnl = 0.0
    for i in range(1, len(df)):
        if strength < params["adx_th"]:
            continue
        prev_position = position
        if fast.iloc[i] > slow.iloc[i] and position <= 0:
            position = 1
        elif fast.iloc[i] < slow.iloc[i] and position >= 0:
            position = -1

        # 포지션이 바뀌는 시점(=진입/청산 발생)에는 수수료+슬리피지를 왕복 비용으로 차감
        if position != prev_position:
            pnl -= (fee_pct + slippage_pct) / 100

        ret = (df["close"].iloc[i] / df["close"].iloc[i - 1] - 1) * position
        pnl += ret

        # 포지션을 들고 있는 매 봉마다 펀딩비 차감 (방향 무관하게 비용으로만 근사 반영;
        # 실제로는 펀딩비 부호가 시장 상황에 따라 바뀌므로 이건 보수적 근사치입니다)
        if position != 0:
            pnl -= per_bar_funding_cost

    return pnl


# --------------------------------------------------------------------------
# 6. 메인 파이프라인
# --------------------------------------------------------------------------

def main(risk_cfg: Optional[RiskConfig] = None):
    if risk_cfg is None:
        # 기본값: 계좌 예시 1,000 USDT, 트레이드당 1% 리스크, 동일방향 최대 3개
        # 실전에서는 반드시 본인 실제 잔고로 바꿔서 호출하세요: main(RiskConfig(account_balance=..., ...))
        risk_cfg = RiskConfig(account_balance=1000, risk_per_trade_pct=1.0, max_concurrent_setups=3)

    # 서킷브레이커: 오늘/이번 주 실현 손실이 한도를 넘었으면 신규 신호 자체를 생성하지 않음
    breaker_msg = circuit_breaker_triggered(risk_cfg)
    if breaker_msg:
        print("=" * 60)
        print(f"🛑 서킷브레이커 작동: {breaker_msg}")
        print("   손실을 만회하려는 추가 진입이 계좌를 가장 크게 망가뜨립니다.")
        print("   내일(또는 다음 주) 한도가 초기화될 때까지 신규 진입을 쉬세요.")
        print("=" * 60)
        return

    print("=" * 60)
    print("1) 거시 국면 판단 중...")
    regime = determine_overall_regime()
    print(f"  BTC 가격추세      : {regime.btc_trend}")
    print(f"  BTC.D 추세        : {regime.btc_d_trend}")
    print(f"  USDT.D 추세       : {regime.usdt_d_trend}")
    print(f"  TOTAL2 추세       : {regime.total2_trend}")
    print(f"  TOTAL3 추세       : {regime.total3_trend}")
    print(f"  ▶ 종합 국면       : {regime.overall.upper()}")
    print("=" * 60)

    print("2) 코인 스크리닝 중 (Binance/OKX/Bitget, 거래량 상위)...")
    setups = screen_market(regime.overall)

    if not setups:
        print("  조건을 만족하는 셋업이 없습니다. (거래량 급증 + RR 1.5 이상 기준)")
        return

    # 상관관계(BTC 동조화) 노출 제한 — 같은 방향 신호가 아무리 많아도 RR 상위 N개만 실전 후보로 남김
    setups = cap_correlated_exposure(setups, risk_cfg)

    print(f"  총 {len(setups)}개 후보 발견 (상관관계 제한 적용 후)\n")

    ready = [s for s in setups if not s.is_chase]
    waiting = [s for s in setups if s.is_chase]

    def _print_setup(s: CoinSetup):
        sizing = calculate_position_size(s.entry_price, s.sl, risk_cfg)
        poc_tag = " 🎯POC컨플루언스" if s.poc_confluence else ""
        print(f"[{s.exchange}] {s.symbol} | {s.bias}{poc_tag}")
        print(f"   근거     : {s.entry_note}")
        asym_str = f"{s.asymmetry:+.2f}" if s.asymmetry is not None else "표본부족"
        print(f"   상대강도 : {s.rs:+.2f}% (지수 대비)   비대칭점수: {asym_str} (상승↑/하락↓ 비대칭)")
        print(f"   현재가   : {s.current_price:.6f}   지정가 진입가: {s.entry_price:.6f}")
        print(f"   TP1/TP2  : {s.tp1:.6f} / {s.tp2:.6f}   SL: {s.sl:.6f}   RR: {s.rr_ratio:.2f}")
        print(f"   포지션   : 수량 {sizing['size']:.4f} (명목가치 {sizing['notional']:.2f} USDT, "
              f"리스크 {sizing['risk_amount']:.2f} USDT = 계좌의 {risk_cfg.risk_per_trade_pct}%)")
        print("-" * 50)

    def _rs_sort_key(s: CoinSetup):
        asym = s.asymmetry if s.asymmetry is not None else 0.0
        if s.bias in ("long", "wait_breakout_long", "range_fade_long"):
            return (s.rs, asym)
        return (-s.rs, -asym)

    print("=" * 60)
    print(f"✅ 되돌림 진입가 도달 — 지금 진입 검토 가능 ({len(ready)}개, 상대강도 순)")
    print("=" * 60)
    if not ready:
        print("  (현재 없음 — 전부 추격 구간이거나 신호 자체가 없음)")
    for s in sorted(ready, key=_rs_sort_key, reverse=True):
        _print_setup(s)

    print()
    print("=" * 60)
    print(f"⏳ 추격 구간 — 아직 진입 대기, 지정가만 걸어두고 관망 ({len(waiting)}개, 상대강도 순)")
    print("=" * 60)
    if not waiting:
        print("  (현재 없음)")
    for s in sorted(waiting, key=_rs_sort_key, reverse=True):
        _print_setup(s)


def validate_lean_auto(exchange_id: str = "binance", top_n: int = 25) -> None:
    """코인 목록을 직접 넣을 필요 없이, 거래량 상위 top_n개를 자동으로 골라
    횡보 기울기 점수의 예측력을 검증하고 결과를 해석까지 붙여 출력합니다.
    (추천용이 아니라 '이 기울기 점수를 믿어도 되는지' 점검용 — 한 번만 돌려보면 됩니다)"""
    symbols = [s for s in get_top_volume_symbols(exchange_id, top_n + 5)
               if s != f"BTC/{QUOTE}"][:top_n]
    print(f"검증 대상 {len(symbols)}개 코인 데이터 수집·분석 중... (수 분 걸릴 수 있음)")
    res = pooled_lean_evaluation(exchange_id, symbols)
    print(res.to_string())

    chosen = res[res.get("selected_on_train", False) == True]  # noqa: E712
    row = chosen.iloc[0] if len(chosen) else res.iloc[0]
    n, hit, z, net = row["test_n"], row["test_hit"], row["test_z"], row["test_avg_net_ret_pct"]
    print("\n[해석]")
    if not n or n < 100 or hit is None or z is None:
        print(f"- 검증 신호 {n}건으로 표본이 부족해 결론을 낼 수 없습니다. 기울기 점수를 방향 선택 근거로 쓰지 마세요.")
    elif hit >= 0.55 and z >= 2 and net > 0:
        print(f"- 적중률 {hit:.1%}, z={z:.1f}, 비용 후 평균 {net:+.2f}% → 쓸 만한 엣지가 있어 보입니다(과거 기준, 미래 보장 아님).")
    else:
        print(f"- 적중률 {hit:.1%}, z={z:.1f}, 비용 후 평균 {net:+.2f}% → 우연과 구분되는 엣지가 확인되지 않았습니다.")
        print("  횡보에서는 방향을 고르지 말고 양쪽 신호를 다 열어두는 쪽이 안전합니다.")


if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        validate_lean_auto()
    else:
        main()
