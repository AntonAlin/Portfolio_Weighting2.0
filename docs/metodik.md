# Metodik

Det här dokumentet beskriver hur `fof_strategilabb.py` räknar, steg för steg. Hur skriptet körs och konfigureras står i [README](../README.md).

## 1. Data

**Källa.** Dagskurser hämtas från Yahoo Finance via `yfinance` med `auto_adjust=True`, så att utdelningar räknas in där Yahoo har dem. Datan börjar tidigast `START_DATE` (2014-01-01).

**Valuta.** Varje fonds valuta läses från Yahoo, eller från `CURRENCY_OVERRIDE` om den är satt. Går det inte gissas valutan från tickerns suffix. Fonder i annan valuta än SEK räknas om med dagliga växelkurser (`EURSEK=X`, `SEK=X` för USD med flera), och fondernas avkastning är därmed alltid i SEK.

**Frekvens.** Allt görs om till veckodata med fredag som slutdag (`W-FRI`). Det undviker tidszonsproblem mellan svenska fondkurser och amerikanska ETF:er. Saknas en vecka fylls den med föregående kurs, högst en vecka i taget.

**Proxies.** Sju tillgångsproxies används i lokal valuta: ACWI (globala aktier), OMXS30, IEF (amerikanska statsobligationer 7–10 år), HYG (high yield), GLD (guld), DBC (råvaror) och USD/SEK. De används till marknadsregimer, korrelationer, beta och ML-signaler. Jämförelseindexet är ACWI omräknat till SEK.

## 2. Justering av tröga kurser

Fonder som prissätts sällan eller med eftersläpning får utjämnade kurser. Det gör att veckoavkastningen blir autokorrelerad och att risken ser lägre ut än den är.

För varje fond skattas första ordningens autokorrelation ρ. Är ρ minst `DESMOOTH_THRESHOLD` (0,10) justeras avkastningen enligt en förenklad Getmansky–Lo–Makarov-modell:

```
r*_t = (r_t − ρ · r_{t−1}) / (1 − ρ)
```

ρ begränsas uppåt till `DESMOOTH_MAX_RHO` (0,90).

Justeringen används **bara vid estimering**, alltså för kovarians och korrelationer. Backtestens avkastning räknas alltid på faktiska kurser, eftersom det är dem man handlar till. I backtesten skattas ρ om i varje estimeringsfönster, så att ett beslut aldrig bygger på en autokorrelation som skattats med framtida data.

## 3. HRP

Vikterna tas fram med Hierarchical Risk Parity (López de Prado, 2016):

1. **Kovarians.** Justerade veckoavkastningar för de senaste `BT_LOOKBACK_WEEKS` (156) veckorna. Matrisen krymps med Ledoit–Wolf, eftersom en rå kovariansmatris på tre års data är mycket brusig. Den räknas om till årstakt.
2. **Klustring.** Korrelationen görs om till avståndet `d = √((1 − ρ) / 2)`, och fonderna klustras hierarkiskt med single linkage.
3. **Ordning.** Fonderna sorteras så att lika fonder hamnar bredvid varandra.
4. **Rekursiv uppdelning.** Listan delas i två halvor gång på gång. Varje halva får vikt i omvänd proportion till sin klustervarians, där klustervariansen räknas med invers-varians-vikter inom klustret.

HRP har inga golv eller tak. Det gör att fonder med hög volatilitet kan få mycket små vikter.

**Vilka fonder som tas med.** En fond får vikt vid en omviktning bara om den har minst `BT_MIN_WEEKS` (52) veckor giltig historik i fönstret och en kurs veckan före beslutet.

## 4. ML-lagret

ML-lagret ändrar inte vilka fonder som ingår eller hur de fördelas inom ett tillgångsslag. Det skalar bara om **budgeten per tillgångsslag**.

### Tillgångsslag

Varje fond tilldelas ett tillgångsslag i `ASSET_CLASS`: Räntor, Absolutavkastning, Krisskydd, Reala tillgångar eller Aktier. Avkastningen för ett tillgångsslag är det likaviktade snittet av de fonder i slaget som har en kurs den veckan.

### Träningsdata

Datan byggs som en panel med en rad per vecka och tillgångsslag.

**Signaler per tillgångsslag:**

| Signal | Definition |
|---|---|
| Momentum 3, 6 och 12 mån | Ackumulerad avkastning över 13, 26 och 52 veckor |
| Volatilitet 6 mån | Standardavvikelse över 26 veckor, i årstakt |
| Drawdown 12 mån | Fall från högsta värdet de senaste 52 veckorna |
| Korrelation mot aktier | Rullande 52 veckors korrelation mot ACWI |

**Marknadssignaler, samma för alla tillgångsslag:**

| Signal | Definition |
|---|---|
| Globala aktier, statsobligationer, high yield, råvaror, guld och USD/SEK, 6 mån | 26 veckors avkastning för respektive proxy |
| Aktievolatilitet 3 mån | 13 veckors volatilitet i ACWI, i årstakt |

Alla signaler förskjuts en vecka. En rad för vecka *t* innehåller alltså bara information fram till och med vecka *t − 1*.

**Målvariabel.** Riskjusterad avkastning de kommande `ML_HORIZON` (13) veckorna:

```
mål_t = avkastning(t … t+12) / (vol26_t · √(13/52))
```

Volatiliteten har ett golv på 1 %. Målvariabeln dras sedan av med snittet för alla tillgångsslag samma vecka. Modellen ska alltså bara ranka tillgångsslagen mot varandra, inte förutsäga om hela marknaden går upp eller ned.

### Modell

`RandomForestRegressor` med 300 träd, maxdjup 3, minst 20 observationer per löv och 50 % av signalerna per delning. Grunda träd och stora löv begränsar hur mycket modellen kan anpassa sig till brus. Det behövs eftersom det bara finns några tiotal kvartal att lära sig av.

**Ingen framåtblick.** Vid ett beslut i vecka *i* tränas modellen enbart på rader där utfallet redan var känt, alltså där *t ≤ i − 13*. Prognosen görs för raden *t = i*. Det krävs minst `ML_MIN_TRAIN` (150) träningsrader, annars används ren HRP.

### Från prognos till vikter

1. Prognoserna för de tillgångsslag som finns i portföljen standardiseras till z-värden.
2. Varje tillgångsslag får multiplikatorn `m = exp(ML_STRENGTH · z)`, begränsad till intervallet `[1 − ML_MAX_TILT, 1 + ML_MAX_TILT]`. Med standardvärdena blir det `exp(0,30 · z)` inom [0,5; 1,5].
3. Varje fonds HRP-vikt multipliceras med sitt tillgångsslags multiplikator, och vikterna normaliseras så att de summerar till 1.

Alla fonder i samma tillgångsslag skalas lika mycket, så HRP:s fördelning inom slaget behålls.

### Utvärdering

Vid varje omviktning i backtesten sparas modellens prognos och det faktiska utfallet per tillgångsslag. Efteråt räknas **rank-IC**, alltså Spearmans rangkorrelation mellan prognos och utfall, för varje kvartal med minst tre tillgångsslag. Snittet och andelen kvartal med positiv IC redovisas. En IC runt noll betyder att modellen inte kan förutsäga något.

## 5. Walk-forward-backtest

- **Omviktning.** Första veckan i varje nytt kalenderkvartal, med början när det finns 52 veckors data.
- **Beslutsdata.** Vikterna för vecka *i* räknas bara på data till och med vecka *i − 1*.
- **Mellan omviktningar** låts vikterna följa marknaden, precis som i en riktig portfölj.
- **Transaktionskostnad.** `TC_BPS` (10 baspunkter) gånger omsättningen dras av den första veckan efter varje omviktning. Omsättningen räknas som summan av absoluta viktförändringar.
- **Om vikter inte går att räkna ut** vid en omviktning, till exempel på grund av för lite data, behålls föregående portfölj.

## 6. Nyckeltal

Nyckeltalen räknas på backtestens veckoavkastning efter kostnader. Riskfri ränta `RF_ANNUAL` är 2 %.

| Nyckeltal | Definition |
|---|---|
| CAGR | Genomsnittlig årlig avkastning |
| Vol | Veckovolatilitet · √52 |
| Sharpe | (CAGR − riskfri ränta) / Vol |
| Sortino | (CAGR − riskfri ränta) / nedsidesvolatilitet |
| Max DD | Största fall från en tidigare topp |
| Calmar | CAGR / \|Max DD\| |
| CVaR 95 % (vecka) | Snittet av de 5 % sämsta veckorna |
| Andel positiva kvartal | Andel kalenderkvartal med positiv avkastning |
| Omsättning per år | Snittomsättning per omviktning · 4 |
| Korrelationer | Mot varje proxy, på veckodata |

**Marknadsregimer.** Veckorna delas in efter proxyernas utveckling: aktieras och aktierally (ACWI i nedersta eller översta kvartilen), räntechock (IEF i nedersta kvartilen), kreditstress (HYG i nedersta kvartilen), inflationspuls (DBC i översta kvartilen), stagflation light (både ACWI och IEF negativa) och svag krona (USD/SEK i översta kvartilen). För varje regim redovisas snittavkastningen i årstakt. En regim visas bara om den inträffat minst `MIN_REGIME_OBS` (8) veckor.

## 7. Innehavslistan

Innehavslistan beskriver en omviktning idag, räknad på samma sätt som i backtesten.

**Belopp.** Varje fonds vikt gånger `INVESTED_CAPITAL` avrundas nedåt till hela kronor. De kronor som blir över fördelas till fonderna med störst avrundningsrest (största restmetoden). Summan blir då exakt lika med kapitalet.

**Riskmått per fond.** De räknas med samma Ledoit–Wolf-kovarians Σ som HRP använde. Med vikterna w och portföljvolatiliteten σ_p = √(wᵀΣw):

| Mått | Formel |
|---|---|
| Riskbidrag | w_i · (Σw)_i / σ_p², summerar till 100 % |
| Volatilitet | √Σ_ii |
| Korrelation mot portföljen | (Σw)_i / (σ_i · σ_p) |
| Korrelation mot aktier | Korrelation mot ACWI i estimeringsfönstret |

**Historisk avkastning per fond** räknas på veckokurser i SEK: 13 veckor, sedan årsskiftet, 52 veckor och upp till 156 veckor i årstakt. Max DD och Sharpe gäller samma treårsfönster. Fonder med kortare historik visar värden sedan start.

**Portföljmått:**

| Mått | Formel |
|---|---|
| Förväntad volatilitet | σ_p |
| VaR 95 %, ett år | 1,645 · σ_p · kapital |
| VaR 95 %, en månad | 1,645 · σ_p · √(1/12) · kapital |
| Effektivt antal fonder | 1 / Σ w_i² |
| Diversifieringskvot | Σ w_i σ_i / σ_p |
| Beta mot globala aktier | Σ w_i · cov(r_i, ACWI) / var(ACWI) |
| Omsättning | ½ · Σ \|w_i − w_i,förra\| mot förra kvartalets målvikter |

VaR är parametrisk med medelavkastning noll och normalfördelning. Verkliga förluster har tjockare svansar, så ett riktigt dåligt år kan bli klart sämre än siffran.

**Köp/sälj-lista.** Är `CURRENT_HOLDINGS` ifylld räknas handeln per fond som målbelopp minus nuvarande innehav. Fonder man äger men som saknas i universumet får målbeloppet 0 och markeras "Utgår".

## Referenser

- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out of Sample.* Journal of Portfolio Management, 42(4).
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* Journal of Multivariate Analysis, 88(2).
- Getmansky, M., Lo, A. W. & Makarov, I. (2004). *An econometric model of serial correlation and illiquidity in hedge fund returns.* Journal of Financial Economics, 74(3).
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1).
