# =====================================================================
#  FOND-I-FOND: HRP MED OCH UTAN ML, KVARTALSVIS OMVIKTNING
#  Hämtar data via yfinance, kör en walk-forward-backtest av HRP och
#  av HRP där en random forest flyttar budget mellan tillgångsslagen,
#  och bygger en interaktiv dashboard direkt i Colab plus en HTML-fil.
#
#  Klistra in i en Colab-cell och kör. Skogen tränas om varje kvartal
#  på det som gick att veta då, så det tar en stund. Tålamod är också alfa.
# =====================================================================

!pip install -q --upgrade yfinance

import json
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")  # yfinance varnar för allt utom det som faktiskt är fel

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display, HTML

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


# ---------------------------------------------------------------------
# 1. KONFIGURATION
# ---------------------------------------------------------------------
FUNDS = {
    "0P0001H70O.ST": "SEB FRN",
    "0P0001788T.ST": "Atlant Opportunity",
    "0P0001788U.ST": "Atlant Protect",
    "0P0001IISR.ST": "Kvartil Calculus",
    "0P0000AAYU.F":  "SEB Asset Selection (EUR)",
    "0P0001SODO.ST": "Centaur Commodity",
    "0P0001N2HD.ST": "Carnegie Infrastructure",
    "0P000091OL.ST": "Atlant Sharp",
    "0P0001TF4H.ST": "REQ Nordic Compounders",
    "0P00017FUN.ST": "Atlant Green Tech Metals",
    "0P0000Z75N.F":  "Amundi Volatility World (EUR H)",
    "0P0001ECQR.ST": "Avanza Global",
    "0P0001H4TL.ST": "Avanza Emerging Markets",
    "JEPG.L":        "JPM Global Equity Premium Income (USD dist)",
}

# Om yfinance har fel om valutan, tvinga fram rätt här.
CURRENCY_OVERRIDE = {
    # "0P0000AAYU.F": "EUR",
    "JEPG.L": "USD",  # London-noterad men handlas i dollar. Utan detta kan yfinance få för sig att det är pence
}

# Proxies för tillgångsklasser i lokal valuta. Valutarisken får en egen proxy (USD/SEK).
PROXIES = {
    "ACWI":  "Globala aktier",
    "^OMX":  "Svenska aktier (OMXS30)",
    "IEF":   "Statsobligationer (UST 7-10y)",
    "HYG":   "High yield-kredit",
    "GLD":   "Guld",
    "DBC":   "Råvaror",
    "SEK=X": "USD/SEK",
}

FX_TICKERS = {
    "EUR": "EURSEK=X", "USD": "SEK=X", "NOK": "NOKSEK=X",
    "DKK": "DKKSEK=X", "GBP": "GBPSEK=X", "CHF": "CHFSEK=X",
}

START_DATE = "2014-01-01"
FREQ = "W-FRI"              # veckodata, så att svenska NAV och amerikanska ETF:er slutar bråka om tidszoner
PPY = 52
RF_ANNUAL = 0.02

# Avutjämning av tröga NAV (AR(1), Getmansky/Lo/Makarov i lightversion)
DESMOOTH = True
DESMOOTH_THRESHOLD = 0.10
DESMOOTH_MAX_RHO = 0.90

# Tillgångsslag per fond. ML-lagret flyttar budget mellan dessa, HRP bestämmer fördelningen inom dem.
# Kolla att indelningen stämmer med hur du ser på fonderna, modellen tror blint på den.
ASSET_CLASS = {
    "0P0001H70O.ST": "Räntor",
    "0P0001788T.ST": "Absolutavkastning",
    "0P0001788U.ST": "Krisskydd",
    "0P0001IISR.ST": "Absolutavkastning",
    "0P0000AAYU.F":  "Absolutavkastning",
    "0P0001SODO.ST": "Reala tillgångar",
    "0P0001N2HD.ST": "Reala tillgångar",
    "0P000091OL.ST": "Absolutavkastning",
    "0P0001TF4H.ST": "Aktier",
    "0P00017FUN.ST": "Reala tillgångar",
    "0P0000Z75N.F":  "Krisskydd",
    "0P0001ECQR.ST": "Aktier",
    "0P0001H4TL.ST": "Aktier",
    "JEPG.L":        "Aktier",
}
_unclassified = [n for t, n in FUNDS.items() if t not in ASSET_CLASS]
if _unclassified:
    print(f"[OBS] Saknar tillgångsslag, hamnar i 'Övrigt': {', '.join(_unclassified)}")

# ML-lagret: en random forest som försöker ranka tillgångsslagen inför nästa kvartal
ML_HORIZON = 13          # veckor framåt, alltså ett kvartal, samma som omviktningen
ML_MIN_TRAIN = 150       # färre träningsexempel än så och modellen får hålla tyst
ML_TREES = 300
ML_DEPTH = 3             # grunda träd, för med 40 kvartal data lär sig djupa träd bara slumpen utantill
ML_MIN_LEAF = 20
ML_STRENGTH = 0.30       # hur mycket en standardavvikelse i prognos flyttar ett tillgångsslags budget
ML_MAX_TILT = 0.50       # ett tillgångsslags budget kan som mest halveras eller växa 50 %
MIN_REGIME_OBS = 8

# Walk-forward
BT_LOOKBACK_WEEKS = 156
BT_MIN_WEEKS = 52
TC_BPS = 10
SEED = 42

STRATEGIES = ["HRP", "HRP + ML"]
DEFAULT_STRATEGY = "HRP"
BENCHMARK_NAME = "ACWI (SEK)"

FUND_PALETTE = ["#3E7CB1", "#B8323E", "#4F7A5A", "#C8963E", "#6B5B95", "#2A9D8F",
                "#8D6E63", "#5B6B78", "#D46A8C", "#1F4E79", "#9AAF5A", "#E07A3F",
                "#7A4E2D", "#4B8BBE"]

DASHBOARD_PATH = "/content/fof_dashboard.html"
DOWNLOAD_HTML = False  # True = Colab laddar ned HTML-filen automatiskt när den är klar

REGIMES = {
    "Aktieras":          lambda P: P["ACWI"] <= P["ACWI"].quantile(0.25),
    "Aktierally":        lambda P: P["ACWI"] >= P["ACWI"].quantile(0.75),
    "Räntechock":        lambda P: P["IEF"] <= P["IEF"].quantile(0.25),
    "Kreditstress":      lambda P: P["HYG"] <= P["HYG"].quantile(0.25),
    "Inflationspuls":    lambda P: P["DBC"] >= P["DBC"].quantile(0.75),
    "Stagflation-light": lambda P: (P["ACWI"] < 0) & (P["IEF"] < 0),
    "Svag krona":        lambda P: P["SEK=X"] >= P["SEK=X"].quantile(0.75),
}


def nm(t):
    """Ticker till namn. Ingen ska behöva kunna 0P-koder utantill, inte ens Morningstar."""
    return FUNDS.get(t, PROXIES.get(t, t))


def fund_class(t):
    return ASSET_CLASS.get(t, "Övrigt")


# ---------------------------------------------------------------------
# 2. DATAHÄMTNING
# ---------------------------------------------------------------------
def fetch_close(ticker, start):
    try:
        h = yf.Ticker(ticker).history(start=start, auto_adjust=True, actions=False)
    except Exception as e:
        print(f"  [FEL] {ticker}: {e}")
        return None
    if h is None or h.empty or "Close" not in h.columns:
        return None
    s = h["Close"].astype(float).copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")]
    s = s[(s > 0) & s.notna()]
    return s if len(s) > 0 else None


def detect_currency(ticker):
    if ticker in CURRENCY_OVERRIDE:
        return CURRENCY_OVERRIDE[ticker]
    try:
        cur = yf.Ticker(ticker).fast_info["currency"]
        if cur:
            return str(cur).upper()
    except Exception:
        pass
    if ticker.endswith(".ST"):
        return "SEK"
    if ticker.endswith(".F") or ticker.endswith(".DE"):
        return "EUR"
    return "USD"


def to_sek(s, ccy, fx_cache):
    if ccy == "SEK":
        return s
    if ccy not in FX_TICKERS:
        raise ValueError(f"Ingen FX-ticker för {ccy}, lägg till den i FX_TICKERS")
    if ccy not in fx_cache:
        fx_cache[ccy] = fetch_close(FX_TICKERS[ccy], START_DATE)
    fx = fx_cache[ccy]
    if fx is None:
        raise ValueError(f"Kunde inte hämta {FX_TICKERS[ccy]}")
    fx_aligned = fx.reindex(s.index.union(fx.index)).ffill().reindex(s.index)
    return (s * fx_aligned).dropna()


def to_weekly(s):
    return s.resample(FREQ).last()


def desmooth(r):
    """Gör tröga fonder lite ärligare. De kommer inte tacka dig."""
    x = r.dropna()
    if len(x) < 30:
        return r, np.nan
    rho = x.autocorr(lag=1)
    if (not DESMOOTH) or np.isnan(rho) or rho < DESMOOTH_THRESHOLD:
        return r, rho
    rho_c = min(rho, DESMOOTH_MAX_RHO)
    out = (x - rho_c * x.shift(1)) / (1 - rho_c)
    return out.reindex(r.index), rho


print("Hämtar fonddata ...")
fx_cache = {}
fund_px_sek, meta = {}, []
for t, name in FUNDS.items():
    s = fetch_close(t, START_DATE)
    if s is None or len(s) < 30:
        print(f"  [SAKNAS] {name} ({t}): ingen eller för lite data.")
        continue
    ccy = detect_currency(t)
    try:
        s_sek = to_sek(s, ccy, fx_cache)
    except Exception as e:
        print(f"  [FEL] {name}: valutakonvertering misslyckades ({e})")
        continue
    fund_px_sek[t] = s_sek
    meta.append({"Ticker": t, "Fond": name, "Valuta": ccy,
                 "Första datum": s.index.min().strftime("%Y-%m-%d"),
                 "Sista datum": s.index.max().strftime("%Y-%m-%d")})

if len(fund_px_sek) < 2:
    raise RuntimeError("Färre än två fonder med data. Det är inte en fond-i-fond, det är en fond.")

print("Hämtar proxies ...")
proxy_px = {}
for p in PROXIES:
    s = fetch_close(p, START_DATE)
    if s is None:
        print(f"  [SAKNAS] proxy {nm(p)} ({p})")
        continue
    proxy_px[p] = s

fund_px_w = pd.DataFrame({t: to_weekly(s) for t, s in fund_px_sek.items()}).ffill(limit=1)
fund_ret_raw = fund_px_w.pct_change(fill_method=None)
proxy_ret = pd.DataFrame({p: to_weekly(s) for p, s in proxy_px.items()}).ffill(limit=1).pct_change(fill_method=None)
# Proxies har historik från 2014, fonderna betydligt kortare. Utan detta pratar .iloc om helt olika veckor
# och backtesten jämför fonder från 2023 med aktiemarknaden 2015. Det blir tomt, av goda skäl.
proxy_ret = proxy_ret.reindex(fund_ret_raw.index)

# Helperiodens avutjämning används BARA i datakvalitetstabellen. Backtesten räknar om rho per fönster
# i weights_at(), annars vet 2019 års optimerare hur trög fonden kommer vara 2025. Det kallas fusk.
fund_ret_est = pd.DataFrame(index=fund_ret_raw.index)
rhos = {}
for t in fund_ret_raw.columns:
    fund_ret_est[t], rhos[t] = desmooth(fund_ret_raw[t])

acwi_sek = None
if "ACWI" in proxy_ret and "SEK=X" in proxy_ret:
    acwi_sek = (1 + proxy_ret["ACWI"]) * (1 + proxy_ret["SEK=X"]) - 1

dq = pd.DataFrame(meta).set_index("Ticker")
dq["Veckor"] = fund_ret_raw.notna().sum()
dq["Vol rå"] = fund_ret_raw.std() * np.sqrt(PPY)
dq["Vol justerad"] = fund_ret_est.std() * np.sqrt(PPY)
dq["AR(1)"] = pd.Series(rhos)
dq["Nollveckor"] = (fund_ret_raw == 0).sum() / fund_ret_raw.notna().sum()
dq["Flagga"] = np.where(dq["AR(1)"] > DESMOOTH_THRESHOLD, "Utjämnad NAV?", "")
dq.loc[dq["Nollveckor"] > 0.20, "Flagga"] = dq.loc[dq["Nollveckor"] > 0.20, "Flagga"] + " Stela priser"
print("\n=== DATAKVALITET ===")
display(dq)


# ---------------------------------------------------------------------
# 3. HRP OCH ML-LAGRET
# ---------------------------------------------------------------------
def regime_masks(P):
    masks = {}
    for name, rule in REGIMES.items():
        try:
            m = rule(P).fillna(False)
        except KeyError:
            continue  # proxyn saknas, regimen får vila
        if m.sum() >= MIN_REGIME_OBS:
            masks[name] = m
    return masks


def _cluster_var(S, idx):
    sub = S[np.ix_(idx, idx)]
    ivp = 1 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(S):
    n = S.shape[0]
    if n == 1:
        return np.array([1.0])
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[a:b] for c in clusters
                    for a, b in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]
            v0, v1 = _cluster_var(S, c0), _cluster_var(S, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w / w.sum()


def class_returns(ret_raw):
    """Likaviktad avkastning per tillgångsslag, av de fonder som fanns just den veckan."""
    classes = sorted({fund_class(t) for t in ret_raw.columns})
    return pd.DataFrame({c: ret_raw[[t for t in ret_raw.columns if fund_class(t) == c]].mean(axis=1, skipna=True)
                         for c in classes})


def cum_ret(r, n):
    return np.exp(np.log1p(r).rolling(n, min_periods=int(n * 0.8)).sum()) - 1


FEATURE_LABELS = {
    "mom13": "Momentum 3 mån", "mom26": "Momentum 6 mån", "mom52": "Momentum 12 mån",
    "vol26": "Volatilitet 6 mån", "dd52": "Drawdown 12 mån", "corr_acwi52": "Korrelation mot aktier",
    "acwi_mom26": "Globala aktier 6 mån", "ief_mom26": "Statsobligationer 6 mån", "hyg_mom26": "High yield 6 mån",
    "dbc_mom26": "Råvaror 6 mån", "gld_mom26": "Guld 6 mån", "usd_mom26": "USD/SEK 6 mån",
    "acwi_vol13": "Aktievolatilitet 3 mån",
}


def build_ml_panel(ret_raw, prox):
    """
    En rad per (vecka, tillgångsslag). Egenskaperna ser bara data t.o.m. veckan innan, målet är
    riskjusterad avkastning kommande kvartal minus snittet för alla tillgångsslag samma vecka.
    Modellen ska alltså ranka, inte spå åt vilket håll hela marknaden går. Det kan ingen.
    """
    idx = ret_raw.index.append(pd.DatetimeIndex([ret_raw.index[-1] + pd.Timedelta(weeks=1)]))  # en rad för "idag"
    cr = class_returns(ret_raw).reindex(idx)
    px = prox.reindex(idx)
    H = ML_HORIZON
    macro = {}
    for tk, lab in [("ACWI", "acwi"), ("IEF", "ief"), ("HYG", "hyg"), ("DBC", "dbc"), ("GLD", "gld"), ("SEK=X", "usd")]:
        if tk in px:
            macro[f"{lab}_mom26"] = cum_ret(px[tk], 26)
    if "ACWI" in px:
        macro["acwi_vol13"] = px["ACWI"].rolling(13, min_periods=10).std() * np.sqrt(PPY)
    macro = pd.DataFrame(macro, index=idx).shift(1)
    parts = []
    for c in cr.columns:
        r = cr[c]
        wealth = (1 + r.fillna(0.0)).cumprod()
        f = pd.DataFrame({
            "mom13": cum_ret(r, 13),
            "mom26": cum_ret(r, 26),
            "mom52": cum_ret(r, 52),
            "vol26": r.rolling(26, min_periods=20).std() * np.sqrt(PPY),
            "dd52": wealth / wealth.rolling(52, min_periods=26).max() - 1,
        }, index=idx)
        if "ACWI" in px:
            f["corr_acwi52"] = r.rolling(52, min_periods=26).corr(px["ACWI"])
        f = f.shift(1)  # allt ovan får bara veta vad som hänt t.o.m. förra veckan
        fwd = np.exp(np.log1p(r).rolling(H, min_periods=H).sum().shift(-(H - 1))) - 1
        f["target"] = fwd / (f["vol26"].clip(lower=0.01) * np.sqrt(H / PPY))
        f = pd.concat([f, macro], axis=1)
        f["class"] = c
        f["t"] = np.arange(len(idx))
        parts.append(f)
    panel = pd.concat(parts, ignore_index=True)
    feats = [c for c in panel.columns if c not in ("target", "class", "t")]
    panel.loc[panel[feats].isna().any(axis=1), "target"] = np.nan
    panel["target"] -= panel.groupby("t")["target"].transform("mean")
    return panel, feats


def ml_class_scores(panel, feats, i):
    """Tränar på allt vars utfall var känt före vecka i och rankar tillgångsslagen för vecka i."""
    train = panel[panel["t"] <= i - ML_HORIZON].dropna(subset=feats + ["target"])
    now = panel[panel["t"] == i].dropna(subset=feats)
    if len(train) < ML_MIN_TRAIN or len(now) < 2:
        return None  # för lite historik, HRP får köra ensam
    model = RandomForestRegressor(n_estimators=ML_TREES, max_depth=ML_DEPTH, min_samples_leaf=ML_MIN_LEAF,
                                  max_features=0.5, random_state=SEED, n_jobs=-1)
    model.fit(train[feats].values, train["target"].values)
    pred = pd.Series(model.predict(now[feats].values), index=now["class"].values)
    return pred, pd.Series(model.feature_importances_, index=feats)


def ml_tilt(w, tickers, pred):
    """HRP:s vikter, med budgeten per tillgångsslag skalad efter skogens rangordning. Inom slaget rör vi inget."""
    cls = np.array([fund_class(t) for t in tickers])
    p = pred[pred.index.isin(cls)]
    if len(p) < 2 or p.std() == 0:
        return w.copy()
    z = (p - p.mean()) / p.std()
    mult = np.clip(np.exp(ML_STRENGTH * z), 1 - ML_MAX_TILT, 1 + ML_MAX_TILT)
    out = w * np.array([mult.get(c, 1.0) for c in cls])
    return out / out.sum()


def weights_at(i, ret_raw, prox, panel, feats):
    """Vikter för beslut i vecka i, baserat enbart på data t.o.m. vecka i-1. Ingen tidsresa tillåten."""
    lo_i = max(0, i - BT_LOOKBACK_WEEKS)
    raw_f = ret_raw.iloc[lo_i:i]
    # Avutjämning med rho skattad på samma fönster. Nu är tidsresan på riktigt förbjuden, inte bara i docstringen.
    # Avutjämnat används bara för estimering. P&L räknas på riktiga priser, för det är de du får betalt i.
    est_f = pd.DataFrame({c: desmooth(raw_f[c])[0] for c in raw_f.columns}, index=raw_f.index)
    est_p = prox.reindex(ret_raw.index).iloc[lo_i:i]  # hängslen och livrem: samma datum oavsett vad som skickas in
    eligible = [c for c in est_f.columns
                if est_f[c].notna().sum() >= BT_MIN_WEEKS and pd.notna(ret_raw[c].iloc[i - 1])]
    if len(eligible) < 2:
        return None
    block = pd.concat([est_f[eligible], est_p], axis=1).dropna()
    if len(block) < BT_MIN_WEEKS:
        return None
    # Krympt kovarians, för rå kovarians på tre år är mest brus med självförtroende
    S = LedoitWolf().fit(block[eligible].values).covariance_ * PPY
    w_hrp = hrp_weights(S)
    ml = ml_class_scores(panel, feats, i)
    w_ml = ml_tilt(w_hrp, eligible, ml[0]) if ml is not None else w_hrp.copy()
    W = pd.DataFrame({"HRP": w_hrp, "HRP + ML": w_ml}, index=eligible)[STRATEGIES]
    return W, block, ml


# ---------------------------------------------------------------------
# 4. WALK-FORWARD MED KVARTALSVIS OMVIKTNING
#    Omviktning sker första veckan i varje nytt kalenderkvartal.
# ---------------------------------------------------------------------
def walk_forward(ret_raw, prox, panel, feats):
    dates = ret_raw.index
    q = dates.to_period("Q")
    rebal = [i for i in range(max(BT_MIN_WEEKS, 1), len(dates)) if q[i] != q[i - 1]]
    port = {s: pd.Series(np.nan, index=dates, dtype=float) for s in STRATEGIES}
    held = {s: pd.Series(dtype=float) for s in STRATEGIES}
    whist = {s: {} for s in STRATEGIES}
    log, ml_log = [], []
    print(f"Kör {len(rebal)} kvartal ...")
    for k, i in enumerate(rebal):
        j = rebal[k + 1] if k + 1 < len(rebal) else len(dates)
        res = weights_at(i, ret_raw, prox, panel, feats)
        Wq = res[0] if res is not None else None
        if res is not None and res[2] is not None:
            # Spara prognos och facit, så vi i efterhand kan se om skogen kunde något eller bara lät säker
            outcome = panel[panel["t"] == i].set_index("class")["target"]
            for c, pv in res[2][0].items():
                ml_log.append({"Datum": dates[i], "Tillgångsslag": c, "Prognos": pv, "Utfall": outcome.get(c, np.nan)})
        for s in STRATEGIES:
            if Wq is not None:
                w_new = pd.Series(Wq[s].values, index=Wq.index)
                whist[s][dates[i]] = w_new
            elif len(held[s]):
                w_new = held[s]  # ingen ny estimering möjlig, vi sitter still och låter vikterna driva
            else:
                continue
            names = list(w_new.index)
            hold = ret_raw.iloc[i:j][names].fillna(0.0)
            w_old = held[s]
            union = w_new.index.union(w_old.index)
            turnover = float((w_new.reindex(union, fill_value=0) - w_old.reindex(union, fill_value=0)).abs().sum()) \
                if Wq is not None else 0.0
            cost = turnover * TC_BPS / 1e4
            cur = w_new.values.copy()
            for n_week, (d, row) in enumerate(hold.iterrows()):
                gross = float(cur @ row.values)
                port[s].loc[d] = gross - (cost if n_week == 0 else 0.0)
                cur = cur * (1 + row.values) / (1 + gross)  # vikterna glider mellan omviktningar, som i verkliga livet
            held[s] = pd.Series(cur, index=names)
            log.append({"Datum": dates[i], "Strategi": s, "Omsättning": turnover})
        if (k + 1) % 4 == 0 or k + 1 == len(rebal):
            print(f"  {k + 1}/{len(rebal)} kvartal klara ({dates[i].date()})")
    return pd.DataFrame(port).dropna(how="all"), whist, pd.DataFrame(log), pd.DataFrame(ml_log)


print("Bygger ML-data ...")
ml_panel, ml_feats = build_ml_panel(fund_ret_raw, proxy_ret)
bt_ret, whist, bt_log, ml_hist = walk_forward(fund_ret_raw, proxy_ret, ml_panel, ml_feats)
if bt_ret.empty:
    raise RuntimeError("Ingen backtest möjlig: för lite överlappande historik. Sänk BT_MIN_WEEKS eller ta bort den yngsta fonden.")

if acwi_sek is not None:
    bt_ret[BENCHMARK_NAME] = acwi_sek.reindex(bt_ret.index)
SERIES = STRATEGIES + ([BENCHMARK_NAME] if BENCHMARK_NAME in bt_ret else [])

print("\nBeräknar vikter för en omviktning idag ...")
today = weights_at(len(fund_ret_raw.index), fund_ret_raw, proxy_ret, ml_panel, ml_feats)

# Träffsäkerhet: rangkorrelation mellan prognos och utfall per kvartal. Över 0,05 i snitt är bra,
# runt 0 betyder att skogen gissar, och då ska du lita på vanliga HRP i stället.
ml_ic = pd.Series(dtype=float)
if not ml_hist.empty:
    ml_ic = (ml_hist.dropna().groupby("Datum")
             .apply(lambda g: g["Prognos"].corr(g["Utfall"], method="spearman") if len(g) >= 3 else np.nan)
             .dropna())
ic_text = f"rank-IC {ml_ic.mean():.2f} över {len(ml_ic)} kvartal" if len(ml_ic) else "för lite historik för att utvärdera"
print(f"\n=== ML-LAGRET: {ic_text}" + (f", positiv {(ml_ic > 0).mean():.0%} av kvartalen ===" if len(ml_ic) else " ==="))


# ---------------------------------------------------------------------
# 5. STATISTIK
# ---------------------------------------------------------------------
def perf_stats(r, proxies=None):
    r = r.dropna()
    if len(r) < 10:
        return {}
    wealth = (1 + r).cumprod()
    cagr = wealth.iloc[-1] ** (PPY / len(r)) - 1
    vol = r.std() * np.sqrt(PPY)
    downside = np.sqrt((np.minimum(r - RF_ANNUAL / PPY, 0) ** 2).mean()) * np.sqrt(PPY)
    mdd = (wealth / wealth.cummax() - 1).min()
    var95 = r.quantile(0.05)
    qr = (1 + r).groupby(r.index.to_period("Q")).prod() - 1
    out = {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": (cagr - RF_ANNUAL) / vol if vol > 0 else np.nan,
        "Sortino": (cagr - RF_ANNUAL) / downside if downside > 0 else np.nan,
        "Max DD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Senaste 12m": (1 + r.iloc[-PPY:]).prod() - 1 if len(r) >= PPY else np.nan,
        "Bästa kvartal": qr.max(),
        "Sämsta kvartal": qr.min(),
        "Andel pos. kvartal": (qr > 0).mean(),
        "CVaR95 (vecka)": r[r <= var95].mean(),
        "Skevhet": r.skew(),
    }
    if proxies is not None:
        pj = proxies.reindex(r.index)
        for p in pj.columns:
            out[f"Korr {nm(p)}"] = r.corr(pj[p])
    return out


stats = {s: perf_stats(bt_ret[s], proxy_ret) for s in SERIES}
turn = bt_log.groupby("Strategi")["Omsättning"].mean() * 4
for s in STRATEGIES:
    stats[s]["Omsättning/år"] = float(turn.get(s, np.nan))

print("\n=== OUT-OF-SAMPLE, KVARTALSVIS OMVIKTNING ===")
display(pd.DataFrame(stats).T[["CAGR", "Vol", "Sharpe", "Max DD", "Calmar", "Senaste 12m", "Andel pos. kvartal"]])

filled = bt_ret[SERIES].fillna(0.0)
wealth = (1 + filled).cumprod()
drawdown = wealth / wealth.cummax() - 1
rolling12 = wealth / wealth.shift(PPY) - 1

qret = (1 + filled).groupby(filled.index.to_period("Q")).prod() - 1
q_labels = [f"{p.year} Q{p.quarter}" for p in qret.index]

yret = (1 + filled).groupby(filled.index.year).prod() - 1
first_d, last_d = filled.index[0], filled.index[-1]
y_labels = []
for y in yret.index:
    lab = str(y)
    if y == last_d.year and last_d.month < 12:
        lab += " hittills"
    elif y == first_d.year and first_d.month > 1:
        lab += " (del)"
    y_labels.append(lab)

masks_oos = regime_masks(proxy_ret.reindex(bt_ret.index))
reg_names = list(masks_oos.keys())
reg_vals = {s: [float(bt_ret[s][m].mean() * PPY) for m in masks_oos.values()] for s in SERIES}

fund_order = [t for t in FUNDS if t in fund_ret_raw.columns]
fund_colors = {nm(t): FUND_PALETTE[i % len(FUND_PALETTE)] for i, t in enumerate(fund_order)}

wh_payload = {}
for s in STRATEGIES:
    if not whist[s]:
        continue
    mat = pd.DataFrame(whist[s]).T.reindex(columns=fund_order).fillna(0.0)
    mat = mat.loc[:, mat.max() > 0.0005]
    wh_payload[s] = {
        "dates": [d.strftime("%Y-%m-%d") for d in mat.index],
        "series": {nm(t): mat[t].tolist() for t in mat.columns},
    }

today_weights, corr_payload = {}, {"labels": [], "z": []}
ml_payload = {"classes": [], "budget": {}, "importance": {}, "ic": ic_text}
if today is not None:
    W_today, block_today, ml_today = today
    for s in STRATEGIES:
        today_weights[s] = {nm(t): float(W_today.loc[t, s]) for t in W_today.index}
    c = block_today.corr()
    corr_payload = {"labels": [nm(x) for x in c.columns], "z": c.values.tolist()}
    print("\n=== VIKTER OM DU VIKTAR OM IDAG (%) ===")
    display((W_today.rename(index=nm) * 100).round(1))
    budget = (W_today.groupby([fund_class(t) for t in W_today.index]).sum() * 100).round(1)
    print("\n=== BUDGET PER TILLGÅNGSSLAG IDAG (%) ===")
    display(budget)
    ml_payload["classes"] = list(budget.index)
    ml_payload["budget"] = {s: (budget[s] / 100).tolist() for s in STRATEGIES}
    if ml_today is not None:
        imp = ml_today[1].sort_values(ascending=False)
        ml_payload["importance"] = {FEATURE_LABELS.get(k, k): float(v) for k, v in imp.items()}
        print("\n=== VAD SKOGEN TITTAR PÅ (feature importance) ===")
        display(imp.rename(index=lambda k: FEATURE_LABELS.get(k, k)).round(3).to_frame("Vikt"))
    W_today.rename(index=nm).to_csv("/content/fof_vikter_idag.csv", encoding="utf-8-sig")

dq_cols = ["Fond", "Valuta", "Första datum", "Veckor", "Vol rå", "Vol justerad", "AR(1)", "Nollveckor", "Flagga"]
dq_rows = dq[dq_cols].reset_index(drop=True).to_dict(orient="records")


# ---------------------------------------------------------------------
# 6. DASHBOARD
# ---------------------------------------------------------------------
def clean(o):
    """JSON tål inte NaN, så vi byter dem mot null. JSON är strängare än en compliance-avdelning."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


n_quarters = len(rebal_dates) if (rebal_dates := sorted({d for s in whist for d in whist[s]})) else 0
payload = clean({
    "title": "Mid risk, fond i fond",
    "facts": [
        f"Walk-forward {first_d.strftime('%Y-%m')} till {last_d.strftime('%Y-%m')}",
        f"{n_quarters} kvartalsvisa omviktningar",
        f"{TC_BPS} bps per omsatt krona",
        f"{BT_LOOKBACK_WEEKS} veckors estimeringsfönster",
        f"{len(fund_order)} fonder, allt i SEK",
        f"ML: {ic_text}",
    ],
    "dates": [d.strftime("%Y-%m-%d") for d in filled.index],
    "strategies": STRATEGIES,
    "default_strategy": DEFAULT_STRATEGY,
    "benchmark": BENCHMARK_NAME if BENCHMARK_NAME in SERIES else None,
    "fund_colors": fund_colors,
    "wealth": {s: wealth[s].tolist() for s in SERIES},
    "drawdown": {s: drawdown[s].tolist() for s in SERIES},
    "rolling": {s: rolling12[s].tolist() for s in SERIES},
    "stats": stats,
    "quarterly": {"labels": q_labels, "values": {s: qret[s].tolist() for s in SERIES}},
    "calendar": {"labels": y_labels, "values": {s: yret[s].tolist() for s in SERIES}},
    "regimes": {"labels": reg_names, "values": reg_vals},
    "weights_hist": wh_payload,
    "today_weights": today_weights,
    "today_date": fund_ret_raw.index[-1].strftime("%Y-%m-%d"),
    "corr": corr_payload,
    "ml": ml_payload,
    "dq": {"cols": dq_cols, "rows": dq_rows},
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
})

DASH_BODY = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700;800&display=swap');
.fof{--snow:#EEF2F4;--paper:#FFFFFF;--ink:#14212B;--slate:#5B6B78;--rule:#D5DDE2;--lingon:#B8323E;--glacier:#3E7CB1;--moss:#4F7A5A;
  font-family:'Schibsted Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink);background:var(--snow);
  padding:32px 32px 24px;border-radius:14px;font-variant-numeric:tabular-nums;line-height:1.45;box-sizing:border-box}
.fof *{box-sizing:border-box}
.fof .top{display:flex;justify-content:space-between;align-items:flex-end;gap:32px;padding-bottom:22px;border-bottom:2px solid var(--ink);margin-bottom:24px}
.fof h1{font-size:44px;line-height:1;font-weight:800;letter-spacing:-0.035em;margin:0}
.fof h1 small{display:block;font-size:15px;font-weight:500;letter-spacing:0;color:var(--slate);margin-top:10px}
.fof .facts{list-style:none;margin:0;padding:0;text-align:right;font-size:13px;color:var(--slate)}
.fof .facts li{padding:1px 0}
.fof .layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:28px;align-items:start}
.fof .rail{position:sticky;top:12px}
.fof .rail h2{font-size:14px;font-weight:600;margin:0 0 10px;color:var(--slate)}
.fof .strat{display:grid;grid-template-columns:22px 1fr auto;gap:2px 10px;width:100%;text-align:left;background:transparent;border:0;border-left:3px solid transparent;
  padding:10px 10px 10px 12px;cursor:pointer;font:inherit;color:inherit;border-radius:0 8px 8px 0}
.fof .strat:hover{background:rgba(20,33,43,.04)}
.fof .strat:focus-visible{outline:2px solid var(--glacier);outline-offset:2px}
.fof .strat.on{background:var(--paper);border-left-color:var(--lingon)}
.fof .strat .rk{grid-row:span 2;font-size:13px;color:var(--slate);padding-top:2px}
.fof .strat .nm{font-size:14px;font-weight:600}
.fof .strat .cg{font-size:14px;font-weight:700;text-align:right}
.fof .strat .sub{grid-column:2 / 4;font-size:12px;color:var(--slate)}
.fof .strat.bench{cursor:default;opacity:.8;border-top:1px dashed var(--rule);border-radius:0;margin-top:8px}
.fof .strat.bench:hover{background:transparent}
.fof .kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:var(--paper);border:1px solid var(--rule);border-radius:10px;margin-bottom:20px}
.fof .kpi{padding:16px 18px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.fof .kpi:nth-child(4n){border-right:0}
.fof .kpi:nth-last-child(-n+4){border-bottom:0}
.fof .kpi .v{font-size:30px;font-weight:700;letter-spacing:-0.02em;line-height:1.1}
.fof .kpi .l{font-size:13px;color:var(--slate);margin-top:4px}
.fof .kpi .d{font-size:12px;margin-top:6px;color:var(--slate)}
.fof .up{color:var(--moss)}
.fof .dn{color:var(--lingon)}
.fof .grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:20px}
.fof .panel{background:var(--paper);border:1px solid var(--rule);border-radius:10px;padding:16px 18px 10px}
.fof .panel h3{font-size:16px;font-weight:700;margin:0;letter-spacing:-0.01em}
.fof .panel p.note{font-size:12.5px;color:var(--slate);margin:2px 0 8px}
.fof .hero{padding:20px 22px 12px}
.fof .hero h3{font-size:20px}
.fof .s12{grid-column:span 12}.fof .s8{grid-column:span 8}.fof .s6{grid-column:span 6}.fof .s4{grid-column:span 4}
.fof .ch{height:320px}.fof .ch.tall{height:430px}.fof .ch.xl{height:520px}
.fof .tw{overflow-x:auto}
.fof table{width:100%;border-collapse:collapse;font-size:13px}
.fof th{text-align:right;font-weight:600;color:var(--slate);padding:8px 10px;border-bottom:2px solid var(--ink);white-space:nowrap;cursor:pointer;user-select:none}
.fof th:first-child,.fof td:first-child{text-align:left}
.fof td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
.fof tbody tr{cursor:pointer}
.fof tbody tr:hover td{background:#F6F8F9}
.fof tr.on td{background:#FBEFF0}
.fof tr.on td:first-child{box-shadow:inset 3px 0 0 var(--lingon)}
.fof .dq th{cursor:default}
.fof .dq tbody tr{cursor:default}
.fof footer{margin-top:24px;padding-top:14px;border-top:1px solid var(--rule);font-size:12px;color:var(--slate);max-width:80ch}
@media (max-width:1100px){
  .fof .layout{grid-template-columns:1fr}
  .fof .rail{position:static;display:flex;overflow-x:auto;gap:6px}
  .fof .rail h2{display:none}
  .fof .strat{min-width:220px;border-left:0;border-bottom:3px solid transparent;border-radius:8px 8px 0 0}
  .fof .strat.on{border-bottom-color:var(--lingon)}
  .fof .s8,.fof .s6,.fof .s4{grid-column:span 12}
  .fof .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
  .fof .kpi:nth-child(2n){border-right:0}
  .fof .top{flex-direction:column;align-items:flex-start}
  .fof .facts{text-align:left}
}
@media (prefers-reduced-motion:reduce){.fof *{transition:none!important}}
</style>

<div class="fof" id="fof-root">
  <div class="top">
    <h1 id="fof-title"></h1>
    <ul class="facts" id="fof-facts"></ul>
  </div>
  <div class="layout">
    <nav class="rail" aria-label="Strategier">
      <h2>Rangordnade efter Sharpe</h2>
      <div id="fof-rail"></div>
    </nav>
    <main>
      <section class="kpis" id="fof-kpis"></section>
      <div class="grid">
        <div class="panel hero s12">
          <h3>Tillväxt av 1 krona</h3>
          <p class="note">Logaritmisk skala, efter transaktionskostnader. Vald strategi i rött, jämförelseindex streckat, övriga i grått.</p>
          <div id="c-wealth" class="ch xl"></div>
        </div>
        <div class="panel s6"><h3>Drawdown</h3><p class="note">Fall från senaste toppen.</p><div id="c-dd" class="ch"></div></div>
        <div class="panel s6"><h3>Rullande 12 månader</h3><p class="note">Avkastning de senaste 52 veckorna.</p><div id="c-roll" class="ch"></div></div>
        <div class="panel s8"><h3>Avkastning per kvartal</h3><p class="note">Ett kvartal motsvarar en omviktningsperiod.</p><div id="c-q" class="ch"></div></div>
        <div class="panel s4"><h3>Vikter vid omviktning idag</h3><p class="note" id="fof-today-note"></p><div id="c-donut" class="ch"></div></div>
        <div class="panel s8"><h3>Budget per tillgångsslag idag</h3><p class="note" id="fof-ml-note"></p><div id="c-mlb" class="ch"></div></div>
        <div class="panel s4"><h3>Vad skogen tittar på</h3><p class="note">Hur mycket varje signal användes i dagens modell. Säger inget om riktning.</p><div id="c-mli" class="ch"></div></div>
        <div class="panel s12"><h3>Målvikter över tid</h3><p class="note">Vikterna som sattes vid varje kvartalsomviktning. Nya fonder dyker upp när de fått tillräckligt med historik.</p><div id="c-wh" class="ch tall"></div></div>
        <div class="panel s6"><h3>Kalenderår</h3><div id="c-cal" class="ch tall"></div></div>
        <div class="panel s6"><h3>Marknadsregimer</h3><p class="note">Annualiserad medelavkastning de veckor regimen gällde.</p><div id="c-reg" class="ch tall"></div></div>
        <div class="panel s8"><h3>Risk mot avkastning</h3><div id="c-scatter" class="ch tall"></div></div>
        <div class="panel s4"><h3>Korrelation, senaste fönstret</h3><p class="note">Fonder och tillgångsklasser, avutjämnade veckodata.</p><div id="c-corr" class="ch tall"></div></div>
        <div class="panel s12"><h3>Alla nyckeltal</h3><p class="note">Klicka på en kolumnrubrik för att sortera, eller på en rad för att välja strategi.</p><div class="tw"><table id="fof-tbl"></table></div></div>
        <div class="panel s12"><h3>Datakvalitet</h3><p class="note">Hög AR(1) betyder utjämnade priser, alltså att verklig risk sannolikt är högre än den ser ut.</p><div class="tw"><table class="dq" id="fof-dq"></table></div></div>
      </div>
    </main>
  </div>
  <footer id="fof-foot"></footer>
</div>

<script>
(function(){
const D = __DATA__;
const S = {active: D.default_strategy, sortKey: 'Sharpe', sortDir: -1};
const BENCH = D.benchmark;
const ALL = D.strategies.concat(BENCH ? [BENCH] : []);
const CFG = {displayModeBar:false, responsive:true};
const C = {ink:'#14212B', slate:'#5B6B78', rule:'#D5DDE2', grid:'#E8EDF0', lingon:'#B8323E', moss:'#4F7A5A', glacier:'#3E7CB1', fade:'#AEBAC3'};
const $ = id => document.getElementById(id);

function ok(v){ return v !== null && v !== undefined && !isNaN(v); }
function pct(v,d){ d = (d === undefined) ? 1 : d; return ok(v) ? (v*100).toFixed(d).replace('-', '−') + '%' : 'saknas'; }
function num(v,d){ d = (d === undefined) ? 2 : d; return ok(v) ? Number(v).toFixed(d).replace('-', '−') : 'saknas'; }
function fmt(v,f){ return f === 'pct' ? pct(v,1) : f === 'pct0' ? pct(v,0) : num(v,2); }
function rgba(hex,a){ const n = parseInt(hex.replace('#',''),16); return 'rgba(' + ((n>>16)&255) + ',' + ((n>>8)&255) + ',' + (n&255) + ',' + a + ')'; }
function st(s,k){ return (D.stats[s] || {})[k]; }

function L(extra){
  const base = {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:"'Schibsted Grotesk', system-ui, sans-serif", color:C.slate, size:12},
    margin:{l:52, r:16, t:6, b:36},
    xaxis:{gridcolor:C.grid, zeroline:false, linecolor:C.rule, ticks:''},
    yaxis:{gridcolor:C.grid, zeroline:false, linecolor:C.rule, ticks:''},
    legend:{orientation:'h', y:-0.16, font:{size:11}, bgcolor:'rgba(0,0,0,0)'},
    hoverlabel:{bgcolor:'#FFFFFF', bordercolor:C.rule, font:{family:"'Schibsted Grotesk', sans-serif", size:12, color:C.ink}},
    showlegend:false
  };
  Object.keys(extra || {}).forEach(k => {
    const v = extra[k];
    if (v && typeof v === 'object' && !Array.isArray(v) && base[k] && typeof base[k] === 'object') base[k] = Object.assign({}, base[k], v);
    else base[k] = v;
  });
  return base;
}

function drawHeader(){
  $('fof-title').innerHTML = D.title + '<small>HRP, med och utan ML-styrning mellan tillgångsslag, omviktat första veckan i varje kvartal</small>';
  $('fof-facts').innerHTML = D.facts.map(f => '<li>' + f + '</li>').join('');
  $('fof-foot').textContent = 'Genererad ' + D.generated + '. Backtesten visar vad strategierna hade gjort med den data som fanns vid varje tidpunkt. Historisk avkastning är ingen garanti för framtida avkastning, och fondavgifter utöver NAV, skatt och ränta på kassa ingår inte.';
}

function drawRail(){
  const ranked = D.strategies.slice().sort((a,b) => (st(b,'Sharpe') || -99) - (st(a,'Sharpe') || -99));
  let html = ranked.map((s,i) =>
    '<button class="strat' + (s === S.active ? ' on' : '') + '" data-s="' + s + '" aria-pressed="' + (s === S.active) + '">' +
    '<span class="rk">' + (i+1) + '</span><span class="nm">' + s + '</span>' +
    '<span class="cg ' + ((st(s,'CAGR') || 0) >= 0 ? '' : 'dn') + '">' + pct(st(s,'CAGR')) + '</span>' +
    '<span class="sub">Sharpe ' + num(st(s,'Sharpe')) + ', max DD ' + pct(st(s,'Max DD'),0) + '</span></button>').join('');
  if (BENCH) {
    html += '<div class="strat bench"><span class="rk"></span><span class="nm">' + BENCH + '</span>' +
      '<span class="cg">' + pct(st(BENCH,'CAGR')) + '</span><span class="sub">Jämförelse, inte en strategi</span></div>';
  }
  $('fof-rail').innerHTML = html;
  $('fof-rail').querySelectorAll('button.strat').forEach(b => b.onclick = () => { S.active = b.dataset.s; renderAll(); });
}

const KPIS = [
  {k:'CAGR', l:'Årlig avkastning', f:'pct', better:1},
  {k:'Vol', l:'Volatilitet', f:'pct', better:-1},
  {k:'Sharpe', l:'Sharpekvot', f:'num', better:1},
  {k:'Max DD', l:'Största fall', f:'pct', better:1},
  {k:'Senaste 12m', l:'Senaste 12 månaderna', f:'pct', better:1},
  {k:'Sämsta kvartal', l:'Sämsta kvartalet', f:'pct', better:1},
  {k:'Andel pos. kvartal', l:'Kvartal med plus', f:'pct0', better:1},
  {k:'Korr Globala aktier', l:'Korrelation mot globala aktier', f:'num', better:0}
];

function drawKPIs(){
  $('fof-kpis').innerHTML = KPIS.map(o => {
    const v = st(S.active, o.k), b = BENCH ? st(BENCH, o.k) : null;
    let delta = '';
    if (o.better !== 0 && ok(v) && ok(b)) {
      const d = v - b, good = d * o.better >= 0;
      const t = o.f === 'num' ? (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(2) : (d >= 0 ? '+' : '−') + Math.abs(d*100).toFixed(1) + ' procentenheter';
      delta = '<div class="d"><span class="' + (good ? 'up' : 'dn') + '">' + t + '</span> mot ' + BENCH + '</div>';
    } else if (ok(b)) {
      delta = '<div class="d">' + BENCH + ': ' + fmt(b, o.f) + '</div>';
    }
    const neg = (o.f !== 'num' && o.k !== 'Vol' && ok(v) && v < 0) ? ' dn' : '';
    return '<div class="kpi"><div class="v' + neg + '">' + fmt(v, o.f) + '</div><div class="l">' + o.l + '</div>' + delta + '</div>';
  }).join('');
}

function drawWealth(){
  const others = D.strategies.filter(s => s !== S.active).map(s => ({
    x:D.dates, y:D.wealth[s], name:s, type:'scatter', mode:'lines',
    line:{color:C.fade, width:1.1}, opacity:0.75, hovertemplate:'%{y:.3f}<extra>' + s + '</extra>'
  }));
  const tr = others;
  if (BENCH) tr.push({x:D.dates, y:D.wealth[BENCH], name:BENCH, type:'scatter', mode:'lines',
    line:{color:C.ink, width:1.6, dash:'dash'}, hovertemplate:'%{y:.3f}<extra>' + BENCH + '</extra>'});
  tr.push({x:D.dates, y:D.wealth[S.active], name:S.active, type:'scatter', mode:'lines',
    line:{color:C.lingon, width:3}, hovertemplate:'%{y:.3f}<extra>' + S.active + '</extra>'});
  Plotly.react('c-wealth', tr, L({hovermode:'x', yaxis:{type:'log', tickformat:'.2f'}}), CFG);
}

function drawDD(){
  const tr = [];
  if (BENCH) tr.push({x:D.dates, y:D.drawdown[BENCH], name:BENCH, type:'scatter', mode:'lines', line:{color:C.ink, width:1.2, dash:'dash'}, hovertemplate:'%{y:.1%}<extra>' + BENCH + '</extra>'});
  tr.push({x:D.dates, y:D.drawdown[S.active], name:S.active, type:'scatter', mode:'lines', fill:'tozeroy',
    line:{color:C.lingon, width:1.8}, fillcolor:rgba(C.lingon, 0.16), hovertemplate:'%{y:.1%}<extra>' + S.active + '</extra>'});
  Plotly.react('c-dd', tr, L({hovermode:'x unified', showlegend:true, yaxis:{tickformat:'.0%'}}), CFG);
}

function drawRoll(){
  const tr = [];
  if (BENCH) tr.push({x:D.dates, y:D.rolling[BENCH], name:BENCH, type:'scatter', mode:'lines', line:{color:C.ink, width:1.2, dash:'dash'}, hovertemplate:'%{y:.1%}<extra>' + BENCH + '</extra>'});
  tr.push({x:D.dates, y:D.rolling[S.active], name:S.active, type:'scatter', mode:'lines', line:{color:C.lingon, width:2.2}, hovertemplate:'%{y:.1%}<extra>' + S.active + '</extra>'});
  Plotly.react('c-roll', tr, L({hovermode:'x unified', showlegend:true, yaxis:{tickformat:'.0%', zeroline:true, zerolinecolor:C.slate, zerolinewidth:1}}), CFG);
}

function drawQ(){
  const q = D.quarterly, v = q.values[S.active] || [];
  const tr = [{x:q.labels, y:v, type:'bar', name:S.active,
    marker:{color:v.map(x => (x || 0) >= 0 ? C.moss : C.lingon)},
    hovertemplate:'%{x}: %{y:.1%}<extra>' + S.active + '</extra>'}];
  if (BENCH) tr.push({x:q.labels, y:q.values[BENCH], type:'scatter', mode:'markers', name:BENCH,
    marker:{color:C.ink, size:6, symbol:'line-ew-open', line:{width:2, color:C.ink}}, hovertemplate:'%{y:.1%}<extra>' + BENCH + '</extra>'});
  Plotly.react('c-q', tr, L({bargap:0.3, showlegend:true, xaxis:{type:'category', tickangle:-45, tickfont:{size:10}, gridcolor:'rgba(0,0,0,0)'}, yaxis:{tickformat:'.0%', zeroline:true, zerolinecolor:C.slate}}), CFG);
}

function drawDonut(){
  const w = D.today_weights[S.active] || {};
  const e = Object.entries(w).filter(x => x[1] > 0.0005).sort((a,b) => b[1] - a[1]);
  $('fof-today-note').textContent = e.length ? 'Beräknat på data till och med ' + D.today_date + '.' : 'Inte tillräckligt med data för en omviktning idag.';
  Plotly.react('c-donut', [{type:'pie', hole:0.6, sort:false, direction:'clockwise',
    labels:e.map(x => x[0]), values:e.map(x => x[1]),
    marker:{colors:e.map(x => D.fund_colors[x[0]]), line:{color:'#FFFFFF', width:2}},
    textinfo:'percent', textposition:'inside', insidetextfont:{color:'#FFFFFF', size:11},
    hovertemplate:'%{label}<br>%{value:.1%}<extra></extra>'}],
    L({showlegend:true, legend:{orientation:'h', y:-0.05, font:{size:10}}, margin:{l:0, r:0, t:0, b:0}}), CFG);
}

function drawML(){
  const m = D.ml;
  $('fof-ml-note').textContent = 'HRP bestämmer vikterna inom varje tillgångsslag, ML flyttar budget mellan dem. Träffsäkerhet i backtesten: ' + m.ic + '.';
  if (!m.classes.length) { Plotly.purge('c-mlb'); Plotly.purge('c-mli'); return; }
  const tr = D.strategies.map(s => ({type:'bar', name:s, x:m.classes, y:m.budget[s],
    marker:{color:s === S.active ? C.lingon : C.fade}, hovertemplate:'%{x}: %{y:.1%}<extra>' + s + '</extra>'}));
  Plotly.react('c-mlb', tr, L({barmode:'group', bargap:0.3, showlegend:true, xaxis:{type:'category', gridcolor:'rgba(0,0,0,0)'}, yaxis:{tickformat:'.0%'}}), CFG);
  const imp = Object.entries(m.importance).reverse();
  if (!imp.length) { Plotly.purge('c-mli'); return; }
  Plotly.react('c-mli', [{type:'bar', orientation:'h', x:imp.map(x => x[1]), y:imp.map(x => x[0]),
    marker:{color:C.glacier}, hovertemplate:'%{y}: %{x:.1%}<extra></extra>'}],
    L({margin:{l:150, r:12, t:6, b:30}, xaxis:{tickformat:'.0%'}, yaxis:{type:'category', tickfont:{size:10}}}), CFG);
}

function drawWH(){
  const h = D.weights_hist[S.active];
  if (!h) { Plotly.purge('c-wh'); return; }
  const tr = Object.keys(h.series).map(f => ({x:h.dates, y:h.series[f], name:f, type:'scatter', mode:'lines', stackgroup:'one',
    line:{width:0.6, shape:'hv', color:D.fund_colors[f]}, fillcolor:rgba(D.fund_colors[f], 0.82),
    hovertemplate:'%{y:.1%}<extra>' + f + '</extra>'}));
  Plotly.react('c-wh', tr, L({hovermode:'x unified', showlegend:true, legend:{orientation:'h', y:-0.12, font:{size:11}}, yaxis:{tickformat:'.0%', range:[0,1]}}), CFG);
}

function heat(id, cols, valuesByRow){
  const rows = ALL.filter(s => valuesByRow[s]);
  const y = rows.map(r => r === S.active ? '<b>' + r + '</b>' : r);
  Plotly.react(id, [{type:'heatmap', z:rows.map(r => valuesByRow[r]), x:cols, y:y, zmid:0,
    colorscale:[[0,'#8E1F2B'],[0.5,'#FFFFFF'],[1,'#2F6B45']], showscale:false, xgap:2, ygap:2,
    texttemplate:'%{z:.0%}', textfont:{size:10, color:C.ink},
    hovertemplate:'%{y}<br>%{x}: %{z:.1%}<extra></extra>'}],
    L({margin:{l:180, r:8, t:6, b:70}, xaxis:{tickangle:-35, gridcolor:'rgba(0,0,0,0)', linecolor:'rgba(0,0,0,0)'}, yaxis:{autorange:'reversed', gridcolor:'rgba(0,0,0,0)', linecolor:'rgba(0,0,0,0)'}}), CFG);
}

function drawScatter(){
  const pts = ALL.filter(s => ok(st(s,'Vol')) && ok(st(s,'CAGR')));
  const tr = pts.map(s => {
    const on = s === S.active, b = s === BENCH;
    return {x:[st(s,'Vol')], y:[st(s,'CAGR')], name:s, type:'scatter', mode:'markers+text',
      text:[s], textposition:'top center', textfont:{size:11, color:on ? C.lingon : C.slate},
      marker:{size:on ? 18 : 11, color:on ? C.lingon : (b ? '#FFFFFF' : C.fade), symbol:b ? 'diamond' : 'circle', line:{width:b ? 2 : 1, color:b ? C.ink : '#FFFFFF'}},
      hovertemplate:'<b>' + s + '</b><br>Volatilitet %{x:.1%}<br>Årlig avkastning %{y:.1%}<br>Sharpe ' + num(st(s,'Sharpe')) + '<extra></extra>'};
  });
  Plotly.react('c-scatter', tr, L({hovermode:'closest', margin:{l:56, r:24, t:20, b:48},
    xaxis:{title:{text:'Volatilitet per år', font:{size:12}}, tickformat:'.0%'},
    yaxis:{title:{text:'Årlig avkastning', font:{size:12}}, tickformat:'.0%'}}), CFG);
}

function drawCorr(){
  if (!D.corr.labels.length) return;
  Plotly.newPlot('c-corr', [{type:'heatmap', z:D.corr.z, x:D.corr.labels, y:D.corr.labels, zmin:-1, zmax:1,
    colorscale:[[0,'#3E7CB1'],[0.5,'#FFFFFF'],[1,'#B8323E']], showscale:false, xgap:1, ygap:1,
    hovertemplate:'%{y}<br>%{x}<br>%{z:.2f}<extra></extra>'}],
    L({margin:{l:150, r:4, t:4, b:150}, xaxis:{tickangle:-60, tickfont:{size:9}, gridcolor:'rgba(0,0,0,0)'}, yaxis:{autorange:'reversed', tickfont:{size:9}, gridcolor:'rgba(0,0,0,0)'}}), CFG);
}

const COLS = [
  ['Strategi', null], ['CAGR','pct'], ['Vol','pct'], ['Sharpe','num'], ['Sortino','num'], ['Max DD','pct'], ['Calmar','num'],
  ['Senaste 12m','pct'], ['Bästa kvartal','pct'], ['Sämsta kvartal','pct'], ['Andel pos. kvartal','pct0'],
  ['CVaR95 (vecka)','pct'], ['Korr Globala aktier','num'], ['Korr Statsobligationer (UST 7-10y)','num'], ['Omsättning/år','pct0']
];
const HEAD = {'Korr Globala aktier':'Korr. aktier', 'Korr Statsobligationer (UST 7-10y)':'Korr. statsobl.', 'Andel pos. kvartal':'Pos. kvartal', 'CVaR95 (vecka)':'CVaR 95 %, vecka'};
const SIGNED = ['CAGR','Senaste 12m','Bästa kvartal','Sämsta kvartal','Max DD','CVaR95 (vecka)'];

function drawTable(){
  const rows = ALL.filter(s => D.stats[s]).slice();
  rows.sort((a,b) => {
    if (S.sortKey === 'Strategi') return S.sortDir * a.localeCompare(b, 'sv');
    const x = st(a, S.sortKey), y = st(b, S.sortKey);
    return S.sortDir * ((ok(x) ? x : -1e9) - (ok(y) ? y : -1e9));
  });
  const head = '<tr>' + COLS.map(c => '<th data-k="' + c[0] + '" aria-sort="' + (S.sortKey === c[0] ? (S.sortDir > 0 ? 'ascending' : 'descending') : 'none') + '">' +
    (HEAD[c[0]] || c[0]) + (S.sortKey === c[0] ? (S.sortDir > 0 ? ' ↑' : ' ↓') : '') + '</th>').join('') + '</tr>';
  const body = rows.map(s => '<tr class="' + (s === S.active ? 'on' : '') + '" data-s="' + s + '">' + COLS.map(c => {
    if (!c[1]) return '<td>' + s + (s === BENCH ? ' <span style="color:' + C.slate + '">(index)</span>' : '') + '</td>';
    const v = st(s, c[0]);
    const cls = SIGNED.includes(c[0]) && ok(v) ? (v >= 0 ? 'up' : 'dn') : '';
    return '<td class="' + cls + '">' + fmt(v, c[1]) + '</td>';
  }).join('') + '</tr>').join('');
  $('fof-tbl').innerHTML = '<thead>' + head + '</thead><tbody>' + body + '</tbody>';
  $('fof-tbl').querySelectorAll('th').forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (S.sortKey === k) S.sortDir *= -1; else { S.sortKey = k; S.sortDir = -1; }
    drawTable();
  });
  $('fof-tbl').querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => {
    if (D.strategies.includes(tr.dataset.s)) { S.active = tr.dataset.s; renderAll(); }
  });
}

function drawDQ(){
  const cols = D.dq.cols;
  const f = (k,v) => {
    if (v === null || v === undefined) return '';
    if (k === 'Vol rå' || k === 'Vol justerad' || k === 'Nollveckor') return pct(v,1);
    if (k === 'AR(1)') return num(v,2);
    return v;
  };
  $('fof-dq').innerHTML = '<thead><tr>' + cols.map(c => '<th>' + c + '</th>').join('') + '</tr></thead><tbody>' +
    D.dq.rows.map(r => '<tr>' + cols.map(c => '<td class="' + (c === 'Flagga' && r[c] ? 'dn' : '') + '">' + f(c, r[c]) + '</td>').join('') + '</tr>').join('') + '</tbody>';
}

function renderAll(){
  drawRail(); drawKPIs(); drawWealth(); drawDD(); drawRoll(); drawQ(); drawDonut(); drawML(); drawWH();
  heat('c-cal', D.calendar.labels, D.calendar.values);
  heat('c-reg', D.regimes.labels, D.regimes.values);
  drawScatter(); drawTable();
}

function boot(){ drawHeader(); drawCorr(); drawDQ(); renderAll(); }

if (window.Plotly) boot();
else {
  const s = document.createElement('script');
  s.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
  s.onload = boot;
  s.onerror = () => { $('fof-kpis').innerHTML = '<div class="kpi">Plotly kunde inte laddas. Kontrollera internetanslutningen och kör cellen igen.</div>'; };
  document.head.appendChild(s);
}
})();
</script>
"""

data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")  # så att ingen fondnamn råkar stänga script-taggen
body = DASH_BODY.replace("__DATA__", data_json)

full_html = (
    "<!doctype html><html lang='sv'><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<title>Mid risk, fond i fond</title></head>"
    "<body style='margin:0;padding:16px;background:#E3E9EC'>" + body + "</body></html>"
)
with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"\nDashboard sparad: {DASHBOARD_PATH}")

if DOWNLOAD_HTML:
    try:
        from google.colab import files
        files.download(DASHBOARD_PATH)
    except Exception as e:
        print(f"Automatisk nedladdning fungerade inte ({e}). Hämta filen från filpanelen till vänster.")

display(HTML(body))
