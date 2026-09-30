"""
app.py — 코인 추천 웹 화면 (핸드폰 우선)
실행:  streamlit run app.py
배포:  README.md 참고 (Streamlit Community Cloud 또는 본인 PC/서버)

⚠️ 참고용 화면입니다. 자동 주문 기능은 없고, 신호는 4시간봉 기준 스윙입니다.
"""
import contextlib
import io
import threading
import traceback

import pandas as pd
import streamlit as st

import app_logic as L
import crypto_market_regime as cmr

st.set_page_config(page_title="코인 추천", page_icon="📈", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown(f"<style>{L.CSS}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_store() -> dict:
    """모든 접속(폰/PC)이 분석 결과를 공유 → 접속할 때마다 새로 스캔하지 않음."""
    return {"lock": threading.Lock(), "result": None, "log": "", "error": None, "scan_cfg": None}


store = get_store()

st.title("📈 코인 추천")
st.caption("Bitget 선물용 · 4시간봉 스윙 신호 · 참고용(자동 주문 아님)")

# ---------------------------------------------------------------- 설정
with st.expander("⚙️ 설정"):
    c1, c2 = st.columns(2)
    balance = c1.number_input("계좌 잔고 (USDT)", min_value=10.0, value=1000.0, step=100.0, key="balance")
    risk_pct = c2.number_input("트레이드당 리스크 (%)", min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                               key="risk_pct", help="손절가에 닿았을 때 잃는 금액이 계좌의 몇 %인지")
    max_n = st.slider("같은 방향 동시 추천 최대 개수", 1, 15, 10, key="max_n",
                      help="알트코인은 BTC와 같이 움직여서, 같은 방향을 많이 잡아도 분산이 잘 안 됩니다")
    c3, c4 = st.columns(2)
    top_n = c3.slider("거래소별 스캔 코인 수", 10, 150, 100, key="top_n", help="바꾼 뒤 '새로 분석'을 눌러야 반영")
    ttl_min = c4.selectbox("자동 갱신 주기(분)", [10, 15, 30, 60], index=1, key="ttl_min")
    bitget_only = st.checkbox("Bitget 선물 거래 가능한 코인만", value=True, key="bitget_only")

risk_cfg = cmr.RiskConfig(account_balance=balance, risk_per_trade_pct=risk_pct, max_concurrent_setups=max_n)
scan_cfg = (top_n, bitget_only)
ttl = ttl_min * 60


def result_age_sec() -> float:
    r = store["result"]
    return float("inf") if r is None else (pd.Timestamp.now() - r["asof"]).total_seconds()


def run_scan(force: bool) -> None:
    prog = st.progress(0.0, text="다른 분석이 진행 중이면 잠시 기다려요...")

    def cb(i: int, n: int, sym: str) -> None:
        prog.progress(min(i / max(n, 1), 1.0), text=f"분석 중 {i}/{n}  {sym}")

    with store["lock"]:
        # 기다리는 동안 다른 접속이 이미 갱신했다면 그 결과를 재사용
        if not force and result_age_sec() <= ttl:
            prog.empty()
            return
        cmr.TOP_N_BY_VOLUME = top_n
        cmr.BITGET_ONLY = bitget_only
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = cmr.run_analysis(risk_cfg, cb)
            store.update(result=res, log=buf.getvalue(), error=None, scan_cfg=scan_cfg)
        except Exception:
            store["error"] = traceback.format_exc()
            store["log"] = buf.getvalue()
    prog.empty()


top_l, top_r = st.columns([3, 2])
refresh = top_r.button("🔄 새로 분석", type="primary", use_container_width=True)
if refresh or result_age_sec() > ttl:
    run_scan(force=refresh)

res = store["result"]
if res is not None:
    mins = int(result_age_sec() // 60)
    top_l.caption(f"마지막 분석 {res['asof']:%H:%M} ({mins}분 전)")

if store["error"]:
    st.error("분석 중 오류가 발생했어요. 아래 내용을 그대로 복사해서 알려주세요.")
    st.code(store["error"], language=None)
if res is None:
    st.stop()
if store["scan_cfg"] != scan_cfg:
    st.info("스캔 설정이 바뀌었어요. '🔄 새로 분석'을 누르면 반영됩니다.")

# ---------------------------------------------------------------- 시장 국면
regime = res["regime"]
macro_hours = regime.snapshot.get("macro_span_hours", 0.0)
st.markdown(L.regime_html(regime, macro_hours), unsafe_allow_html=True)
if "btc_d" not in regime.snapshot:
    st.caption("ⓘ 이번엔 CoinGecko 응답이 지연/제한(429)되어 BTC.D 등 최신값을 못 받았어요. "
               "BTC 가격 추세로는 정상 판단했고, 다음 갱신 때 다시 시도합니다.")
with st.expander("📊 판단 근거 자세히"):
    st.markdown(L.regime_detail_html(regime), unsafe_allow_html=True)
    st.caption(regime.explanation)
if macro_hours < 24:
    st.caption(f"ⓘ BTC.D·USDT.D·TOTAL 추세는 기록이 {macro_hours:.0f}시간 쌓였어요. "
               "24시간 이상 쌓이면 국면 판단에 반영되고, 그 전에는 BTC 가격 추세로만 판단합니다.")

# ---------------------------------------------------------------- 서킷브레이커
breaker = cmr.circuit_breaker_triggered(risk_cfg)
if breaker:
    st.error(f"🛑 {breaker}\n\n손실 만회 목적의 추가 진입이 계좌를 가장 크게 망가뜨립니다. 오늘은 쉬세요.")
    st.stop()

# ---------------------------------------------------------------- 추천
with contextlib.redirect_stdout(io.StringIO()):
    setups = cmr.cap_correlated_exposure(res["all_setups"], risk_cfg) if res["all_setups"] else []
ready = sorted([s for s in setups if not s.is_chase], key=L.sort_key, reverse=True)
waiting = sorted([s for s in setups if s.is_chase], key=L.sort_key, reverse=True)


def render(items: list) -> None:
    for s in items:
        sizing = cmr.calculate_position_size(s.entry_price, s.sl, risk_cfg)
        st.markdown(L.card_html(s, sizing, risk_pct), unsafe_allow_html=True)
        with st.expander("📋 주문 메모 (복사)"):
            st.code(L.order_memo(s, sizing, L.leverage_guide(s.entry_price, s.sl)), language=None)


if not setups:
    st.info("**지금은 조건에 맞는 코인이 없어요.**\n\n"
            "거래량·상대강도·손익비·상위추세(일봉)·펀딩비·스프레드 조건을 모두 통과한 코인이 없다는 뜻입니다. "
            "억지로 진입하지 않는 것도 전략이에요. 다음 갱신 때 다시 확인하세요.")
else:
    tab1, tab2 = st.tabs([f"✅ 진입 검토 ({len(ready)})", f"⏳ 대기 ({len(waiting)})"])
    with tab1:
        st.caption("진입가에 도달했거나 가까운 코인. 상대강도 순.")
        if ready:
            render(ready)
        else:
            st.write("지금 진입가에 도달한 코인은 없어요. '대기' 탭의 지정가를 확인하세요.")
    with tab2:
        st.caption("아직 되돌림 전이라 지금 들어가면 추격인 코인. 지정가만 걸어두고 기다리세요.")
        if waiting:
            render(waiting)
        else:
            st.write("대기 중인 코인이 없어요.")

# ---------------------------------------------------------------- 부가 기능
with st.expander("📖 용어 / 사용법"):
    st.markdown(
        "- **진입(지정가)**: 이 가격에 지정가 주문을 걸어요. 시장가로 쫓아 들어가지 않는 게 핵심입니다.\n"
        "- **손절**: 이 가격에 닿으면 무조건 정리. 수량은 '손절 시 손실이 계좌의 리스크 %'가 되도록 계산돼 있어요.\n"
        "- **목표1/2**: 1차·2차 익절 구간. **손익비**는 (목표1까지 거리) ÷ (손절까지 거리).\n"
        "- **레버리지 상한 가이드**: 청산가가 손절가보다 훨씬 멀리 있도록 잡은 상한(최대 10배, 격리마진 기준). "
        "이 화면은 레버리지를 권하는 게 아니라 '이 이상 올리면 손절 전에 청산될 수 있다'는 한계선을 보여줘요.\n"
        "- **🎯 매물대 겹침**: 진입가가 거래량이 많이 쌓인 가격대(POC/Value Area)와 겹쳐 신뢰도가 높은 자리.\n"
        "- **박스 반등/돌파 리테스트**: 횡보장 전용 신호. 상위추세·구조·거래량으로 방향을 판단해 반대 방향은 걸러요."
    )

with st.expander("📝 매매 결과 기록 (서킷브레이커용)"):
    st.caption("하루 손실 5% / 주간 10%를 넘으면 신규 추천을 자동 중단합니다. "
               "※ Streamlit Cloud에서는 앱이 재시작되면 기록이 초기화될 수 있어요.")
    lc1, lc2 = st.columns(2)
    pnl = lc1.number_input("손익 (USDT, 손실은 음수)", value=0.0, step=1.0, key="pnl_in")
    sym_in = lc2.text_input("코인 (선택)", key="pnl_sym")
    if st.button("기록 저장"):
        cmr.log_trade_result(float(pnl), sym_in)
        st.success("저장했어요.")
        st.rerun()

if store["log"].strip():
    with st.expander("🔍 제외된 코인 / 분석 로그"):
        st.code(store["log"], language=None)

st.caption("⚠️ 참고용 정보이며 수익을 보장하지 않습니다. 손절은 반드시 지키고, 감당 가능한 금액만 사용하세요.")
