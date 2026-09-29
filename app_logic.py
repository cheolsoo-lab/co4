"""
app_logic.py — 화면(app.py)에서 쓰는 순수 로직 모음 (Streamlit 의존 없음 → 단독 테스트 가능)
카드 HTML 생성, 가격 포맷, 레버리지 상한 가이드, 주문 메모 텍스트 등.
"""
import html
import math
from typing import Dict, Optional, Tuple

REGIME_INFO = {
    "uptrend": {"emoji": "🟢", "title": "상승장", "color": "#16a34a", "bg": "rgba(22,163,74,.13)",
                "desc": "롱 위주. 급등 추격 말고 눌림목(되돌림 지정가)에서 진입"},
    "downtrend": {"emoji": "🔴", "title": "하락장", "color": "#dc2626", "bg": "rgba(220,38,38,.13)",
                  "desc": "숏 위주. 급락 추격 말고 반등 구간에서 진입"},
    "sideways": {"emoji": "🟡", "title": "횡보장", "color": "#d97706", "bg": "rgba(217,119,6,.13)",
                 "desc": "박스 경계에서만 진입. 방향은 기울기 점수로 판단하고 중간 구간은 관망"},
}
TREND_ICON = {"uptrend": "↑", "downtrend": "↓", "sideways": "→"}

# bias → (방향, 한글 방향, 설명)
BIAS_INFO = {
    "long": ("long", "롱", "추세 눌림목 진입"),
    "short": ("short", "숏", "추세 반등 매도"),
    "wait_breakout_long": ("long", "롱", "박스 상단 돌파 후 리테스트"),
    "wait_breakout_short": ("short", "숏", "박스 하단 이탈 후 리테스트"),
    "range_fade_long": ("long", "롱", "박스 하단 반등(평균회귀)"),
    "range_fade_short": ("short", "숏", "박스 상단 저항(평균회귀)"),
}

CSS = """
.block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:720px}
.regime{border-radius:16px;padding:14px 16px;margin:6px 0 10px 0;border:1px solid rgba(128,128,128,.25)}
.regime .t{font-size:1.35rem;font-weight:800;line-height:1.3}
.regime .d{font-size:.88rem;opacity:.9;margin-top:8px;line-height:1.5}
.gauge{margin-top:12px}
.gauge-track{position:relative;height:8px;border-radius:999px;
  background:linear-gradient(90deg,#dc2626,#9ca3af,#16a34a)}
.gauge-mark{position:absolute;top:-4px;width:16px;height:16px;border-radius:50%;
  background:#fff;border:3px solid #111827;transform:translateX(-50%);box-shadow:0 1px 3px rgba(0,0,0,.4)}
.gauge-lbl{position:absolute;top:10px;font-size:.68rem;opacity:.65}
.gauge-lbl.left{left:0}.gauge-lbl.right{right:0}
.brk{margin-top:14px;font-size:.82rem;background:rgba(128,128,128,.12);border-radius:10px;padding:7px 10px}
.unver{margin-top:8px;font-size:.74rem;opacity:.65;font-style:italic}
.conf{margin-top:10px;font-size:.78rem;opacity:.85}
.conf .hint{opacity:.7;font-size:.72rem}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.chip{font-size:.75rem;padding:3px 9px;border-radius:999px;background:rgba(128,128,128,.16)}
.cc{border:1px solid rgba(128,128,128,.28);border-left:5px solid var(--c);border-radius:14px;
    padding:12px 14px;margin:10px 0 4px 0;background:rgba(128,128,128,.06)}
.cc.long{--c:#16a34a}.cc.short{--c:#dc2626}
.cc-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pill{font-weight:700;font-size:.78rem;padding:2px 10px;border-radius:999px;color:#fff}
.pill.long{background:#16a34a}.pill.short{background:#dc2626}
.sym{font-size:1.15rem;font-weight:800}
.tag{font-size:.72rem;padding:1px 8px;border-radius:999px;background:rgba(245,158,11,.22)}
.sub{opacity:.75;font-size:.82rem;margin:3px 0 8px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.grid div{background:rgba(128,128,128,.11);border-radius:10px;padding:6px 8px}
.k{display:block;font-size:.68rem;opacity:.7}
.v{display:block;font-weight:650;font-size:.9rem;word-break:break-all}
.v.sl{color:#dc2626}.v.tp{color:#16a34a}
.foot{font-size:.8rem;opacity:.9;margin-top:8px;line-height:1.55}
.warn{margin-top:6px;font-size:.78rem;color:#d97706}
"""


def fmt_price(x: Optional[float]) -> str:
    """가격 크기에 맞춰 소수점 자리수를 자동 조절."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 10:
        return f"{x:,.3f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}"


def fmt_money(x: float) -> str:
    return f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"


def leverage_guide(entry: float, sl: float, liq_multiple: float = 3.0, cap: int = 10) -> int:
    """격리마진 기준 레버리지 상한 가이드.
    청산까지의 거리(≈1/레버리지)가 손절까지의 거리보다 liq_multiple배 이상 멀도록 잡고, 상한은 cap배.
    (수수료·유지증거금은 무시한 근사치입니다. 레버리지가 높을수록 손절 전에 청산될 위험이 커집니다.)"""
    if not entry:
        return 1
    dist = abs(entry - sl) / entry
    if dist <= 0:
        return 1
    return int(max(1, min(cap, math.floor(1 / (liq_multiple * dist)))))


def side_info(bias: str) -> Tuple[str, str, str]:
    return BIAS_INFO.get(bias, ("long", bias, ""))


def sort_key(setup) -> Tuple[float, float]:
    """롱은 상대강도가 클수록, 숏은 작을수록(더 약할수록) 위로. 동률이면 비대칭 점수."""
    asym = setup.asymmetry if setup.asymmetry is not None else 0.0
    if side_info(setup.bias)[0] == "long":
        return (setup.rs, asym)
    return (-setup.rs, -asym)


def order_memo(setup, sizing: Dict, lev: int) -> str:
    side, side_kr, sub = side_info(setup.bias)
    return "\n".join([
        f"{setup.symbol.replace('/', '')}  {side_kr} ({sub})",
        f"진입(지정가): {fmt_price(setup.entry_price)}",
        f"손절: {fmt_price(setup.sl)}",
        f"목표1: {fmt_price(setup.tp1)}  /  목표2: {fmt_price(setup.tp2)}",
        f"수량: {sizing['size']:.4f}  (명목 ${fmt_money(sizing['notional'])})",
        f"레버리지 상한 가이드: {lev}배 이하 (격리)",
    ])


def card_html(setup, sizing: Dict, risk_pct: float) -> str:
    side, side_kr, sub = side_info(setup.bias)
    lev = leverage_guide(setup.entry_price, setup.sl)
    e = html.escape
    tags = ""
    if setup.poc_confluence:
        tags += '<span class="tag">🎯 매물대 겹침</span>'
    stats = f"상대강도 {setup.rs:+.1f}%"
    if setup.asymmetry is not None:
        stats += f" · 비대칭 {setup.asymmetry:+.2f}"
    warn = ""
    if setup.is_chase and setup.entry_price:
        gap = (setup.current_price - setup.entry_price) / setup.entry_price * 100
        warn = (f'<div class="warn">⏳ 현재가가 진입가보다 {abs(gap):.1f}% '
                f'{"위" if gap > 0 else "아래"} — 지정가만 걸어두고 기다리세요 (지금 시장가 진입은 추격)</div>')

    def cell(k: str, v: str, cls: str = "") -> str:
        return f'<div><span class="k">{e(k)}</span><span class="v {cls}">{e(v)}</span></div>'

    grid = "".join([
        cell("현재가", fmt_price(setup.current_price)),
        cell("진입(지정가)", fmt_price(setup.entry_price)),
        cell("손절", fmt_price(setup.sl), "sl"),
        cell("목표1", fmt_price(setup.tp1), "tp"),
        cell("목표2", fmt_price(setup.tp2), "tp"),
        cell("손익비", f"{setup.rr_ratio:.2f}"),
    ])
    foot = (f"수량 <b>{sizing['size']:.4f}</b> · 명목 ${fmt_money(sizing['notional'])} · "
            f"손절 시 손실 ${fmt_money(sizing['risk_amount'])} (계좌의 {risk_pct:g}%)<br>"
            f"레버리지 상한 가이드 <b>{lev}배 이하</b> (격리마진)")
    return (f'<div class="cc {side}">'
            f'<div class="cc-head"><span class="pill {side}">{side.upper()} {e(side_kr)}</span>'
            f'<span class="sym">{e(setup.symbol)}</span>{tags}</div>'
            f'<div class="sub">{e(sub)} · {e(stats)}</div>'
            f'<div class="grid">{grid}</div>'
            f'<div class="foot">{foot}</div>{warn}</div>')


def regime_html(regime, macro_hours: float) -> str:
    info = REGIME_INFO[regime.overall]
    score = max(-1.0, min(1.0, regime.score))
    pos_pct = (score + 1) / 2 * 100  # 게이지 위 마커 위치(%) — -1=왼쪽 끝, +1=오른쪽 끝
    conf_color = {"높음": "#16a34a", "보통": "#d97706", "낮음": "#6b7280"}.get(regime.confidence_label, "#6b7280")

    breakout = ""
    if regime.overall == "sideways" and regime.breakout_up and regime.breakout_down:
        from crypto_market_regime import fmt_range  # 지연 임포트: app_logic은 cmr 없이도 단독 임포트 가능해야 함
        breakout = (f'<div class="brk">위로 <b>{html.escape(fmt_range(regime.breakout_up))}</b> 돌파 시 상승 전환 · '
                    f'아래로 <b>{html.escape(fmt_range(regime.breakout_down))}</b> 이탈 시 하락 전환</div>')

    unverified = ""
    if regime.lean_verified is False:
        unverified = '<div class="unver">ⓘ 이 방향 판단(횡보 기울기)은 아직 실측 검증 전입니다 — 참고용</div>'

    return (
        f'<div class="regime" style="background:{info["bg"]};border-left:5px solid {info["color"]}">'
        f'<div class="t">{info["emoji"]} {html.escape(regime.headline)}</div>'
        f'<div class="gauge"><div class="gauge-track">'
        f'<span class="gauge-lbl left">하락</span><span class="gauge-lbl right">상승</span>'
        f'<div class="gauge-mark" style="left:{pos_pct:.1f}%"></div></div></div>'
        f'<div class="d">{html.escape(regime.explanation)}</div>'
        f'{breakout}{unverified}'
        f'<div class="conf">판단 신뢰도: <b style="color:{conf_color}">{html.escape(regime.confidence_label)}</b>'
        f' <span class="hint">(근거 지표들이 서로 얼마나 같은 방향을 가리키는지)</span></div>'
        f'</div>'
    )


def regime_detail_html(regime) -> str:
    """지표별 상세 — 기본 카드에서는 숨기고 '자세히' 접기 안에서만 보여줌."""
    def chip(label: str, trend: str) -> str:
        return f'<span class="chip">{html.escape(label)} {TREND_ICON[trend]}</span>'

    chips = "".join([
        chip("BTC", regime.btc_trend), chip("BTC.D", regime.btc_d_trend),
        chip("USDT.D", regime.usdt_d_trend), chip("TOTAL2", regime.total2_trend),
        chip("TOTAL3", regime.total3_trend),
    ])
    return f'<div class="chips">{chips}</div>'
