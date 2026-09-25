# Portfolio Weighting 2.0

Kvartalsvis viktning av en fond-i-fond-portfölj med låg risk. Skriptet hämtar kurser, räknar fram vikter med **Hierarchical Risk Parity (HRP)** och låter sedan en **random forest** flytta budget mellan tillgångsslagen. Resultatet blir en innehavslista i kronor och en interaktiv dashboard.

Allt ligger i en fil, [`fof_strategilabb.py`](fof_strategilabb.py), som är skriven för att klistras in i en cell i Google Colab eller Microsoft Fabric och köras rakt av.

## Innehåll

- [Vad skriptet gör](#vad-skriptet-gör)
- [Komma igång](#komma-igång)
- [Konfiguration](#konfiguration)
- [Utdata](#utdata)
- [Läsa resultatet](#läsa-resultatet)
- [Begränsningar](#begränsningar)
- [Metodbeskrivning](docs/metodik.md)

## Vad skriptet gör

1. **Hämtar data.** Dagskurser för fonderna och för sju tillgångsproxies (globala aktier, svenska aktier, statsobligationer, high yield, guld, råvaror, USD/SEK) hämtas via `yfinance`. Allt räknas om till SEK och veckodata.
2. **Justerar tröga kurser.** Fonder vars kurser är utjämnade (hög autokorrelation) justeras, så att deras risk inte underskattas.
3. **Viktar med HRP.** Vikterna bygger på en krympt kovariansmatris (Ledoit–Wolf) över de senaste tre åren.
4. **Lägger på ML-lagret.** En random forest rankar tillgångsslagen inför nästa kvartal. Ett tillgångsslags andel kan skalas mellan 0,5× och 1,5×. Fördelningen mellan fonderna inom ett tillgångsslag är fortfarande HRP:s.
5. **Kör en walk-forward-backtest.** Portföljen viktas om första veckan i varje kvartal. Varje beslut fattas bara med data som fanns vid det tillfället, och transaktionskostnader dras av.
6. **Bygger innehavslistan och dashboarden** för en omviktning idag.

Två strategier jämförs sida vid sida:

| Strategi | Beskrivning |
|---|---|
| `HRP + ML` | Förval. HRP-vikter där ML-lagret flyttar budget mellan tillgångsslag. |
| `HRP` | Ren HRP utan ML-lagret, som jämförelse. |

Jämförelseindex är ACWI omräknat till SEK.

## Komma igång

### Google Colab

1. Skapa en ny notebook.
2. Klistra in hela `fof_strategilabb.py` i en cell.
3. Ändra `INVESTED_CAPITAL` till ditt belopp (se [Konfiguration](#konfiguration)).
4. Kör cellen. Första körningen tar några minuter, eftersom modellen tränas om för varje kvartal i backtesten.

Beroenden som inte redan finns i Colab installeras av första raden (`!pip install -q --upgrade yfinance`). Resten används som de är: `numpy`, `pandas`, `scipy`, `scikit-learn`, `jinja2` (för tabellformatering) och `openpyxl` (för Excel-exporten).

### Microsoft Fabric

Skriptet fungerar i en Fabric-notebook med två ändringar:

- Byt `!pip install -q --upgrade yfinance` mot `%pip install -q --upgrade yfinance` och lägg den i en egen cell.
- Ändra sökvägarna som börjar på `/content/` till en plats som finns i Fabric, till exempel `/lakehouse/default/Files/`. Det gäller `DASHBOARD_PATH` och de tre exportfilerna i avsnitt 5 och 5b.

Fabric behöver nå Yahoo Finance utåt för att hämta kurser.

## Konfiguration

Alla inställningar ligger i avsnitt 1 högst upp i skriptet.

### Det du normalt ändrar

| Inställning | Vad den gör |
|---|---|
| `INVESTED_CAPITAL` | Beloppet i kronor som ska fördelas. Är satt till ett exempelvärde i repot, ange ditt eget lokalt. |
| `CURRENT_HOLDINGS` | Valfritt. Det du äger idag, i kronor per ticker. Fylls den i blir innehavslistan en köp/sälj-lista. Fonder du äger som inte längre finns i `FUNDS` markeras "Utgår" och säljs. |
| `FUNDS` | Fonduniversumet, med Yahoo-ticker som nyckel och visningsnamn som värde. |
| `ASSET_CLASS` | Vilket tillgångsslag varje fond tillhör. ML-lagret flyttar budget mellan dessa, så indelningen spelar roll. Fonder som saknas här hamnar i "Övrigt". |
| `CURRENCY_OVERRIDE` | Tvingar fram rätt valuta om `yfinance` gissar fel. Används bland annat för `JEPG.L`, som handlas i USD men är noterad i London. |

### Lägga till en fond

1. Leta upp fondens Yahoo-ticker. Svenska fonder har ofta formatet `0P000XXXXX.ST`.
2. Lägg till den i `FUNDS` och i `ASSET_CLASS`.
3. Kör skriptet och kontrollera att fonden får rätt valuta och tillräckligt med historik i datakvalitetstabellen.

En fond tas med i omviktningen först när den har minst 52 veckors historik (`BT_MIN_WEEKS`).

### Modellinställningar

| Inställning | Standard | Vad den gör |
|---|---|---|
| `BT_LOOKBACK_WEEKS` | 156 | Estimeringsfönster i veckor, cirka tre år. |
| `BT_MIN_WEEKS` | 52 | Minsta historik för att en fond ska få vikt. |
| `TC_BPS` | 10 | Transaktionskostnad i baspunkter per omsatt krona. |
| `ML_STRENGTH` | 0,30 | Hur hårt ML-prognosen styr budgeten mellan tillgångsslag. |
| `ML_MAX_TILT` | 0,50 | Högsta skalning av ett tillgångsslags budget (±50 %). |
| `ML_HORIZON` | 13 | Prognoshorisont i veckor, ett kvartal. |
| `ML_MIN_TRAIN` | 150 | Minsta antal träningsexempel innan ML-lagret får påverka vikterna. |
| `ML_TREES`, `ML_DEPTH`, `ML_MIN_LEAF` | 300, 3, 20 | Skogens storlek. Träden hålls grunda med flit, eftersom datamängden är liten. |
| `DESMOOTH_THRESHOLD` | 0,10 | Autokorrelation över detta räknas som utjämnade kurser och justeras. |
| `DEFAULT_STRATEGY` | `HRP + ML` | Strategin som innehavslistan och dashboarden öppnar med. |

## Utdata

### I notebooken

- **Datakvalitet:** valuta, historiklängd, rå och justerad volatilitet, autokorrelation och flaggor för utjämnade eller stela kurser per fond.
- **ML-lagrets träffsäkerhet:** snittlig rank-IC och andel kvartal med positiv IC.
- **Nyckeltal** för båda strategierna och jämförelseindexet, beräknade på backtesten.
- **Vikter idag** per fond och per tillgångsslag.
- **Vad modellen tittar på:** hur mycket varje signal används (feature importance).
- **Innehavslistan** för förvald strategi.

### Filer

| Fil | Innehåll |
|---|---|
| `fof_dashboard.html` | Fristående dashboard som går att öppna i en webbläsare. En internetanslutning behövs för Plotly och typsnittet. |
| `fof_innehav.csv` | Innehavslistan med semikolon och decimalkomma, så att svensk Excel läser den rätt. |
| `fof_innehav.xlsx` | Samma lista som Excel-fil. Skapas bara om `openpyxl` finns. |
| `fof_vikter_idag.csv` | Dagens vikter per fond för båda strategierna. |

Sätt `DOWNLOAD_HTML = True` om Colab ska ladda ned dashboarden automatiskt.

### Dashboarden

Överst finns strategiväljaren och nyckeltal jämförda med ACWI. Därefter kommer:

- **Innehavslista:** belopp per fond, grupperat per tillgångsslag, med sammanfattning på portföljnivå och en knapp för CSV-nedladdning.
- **Tillväxt av en krona, drawdown och rullande 12 månader.**
- **Avkastning per kvartal och kalenderår.**
- **Vikter idag** och **budget per tillgångsslag** med och utan ML.
- **Målvikter över tid:** hur vikterna har ändrats vid varje omviktning.
- **Marknadsregimer:** hur strategierna har klarat aktieras, räntechock, kreditstress med mera.
- **Risk mot avkastning, korrelationsmatris, alla nyckeltal och datakvalitet.**

## Läsa resultatet

### Innehavslistan

| Kolumn | Betydelse |
|---|---|
| Vikt / Belopp | Målvikt och belopp i hela kronor. Beloppen summerar exakt till `INVESTED_CAPITAL`. |
| Förändring | Procentenheter mot målvikten vid förra kvartalsomviktningen. "Ny" betyder att fonden inte hade vikt då. |
| ML-justering | Hur många procentenheter ML-lagret flyttat fonden jämfört med ren HRP. |
| Riskbidrag | Fondens andel av portföljens totala risk. En liten vikt kan ge ett stort riskbidrag. |
| Volatilitet, korr. portfölj, korr. aktier | Beräknat på samma justerade data och fönster som HRP använder. |
| 3 mån, i år, 1 år, 3 år per år | Historisk avkastning i SEK. Fonder med kortare historik visar avkastning sedan start. |
| Max DD 3 år, Sharpe 3 år | Största fall och riskjusterad avkastning de senaste tre åren. |

På portföljnivå visas:

- **Förväntad volatilitet:** portföljens risk enligt kovariansmatrisen.
- **VaR 95 %:** en förlust i kronor som bara ska överskridas vart tjugonde år, eller var tjugonde månad för månadssiffran. Den är parametrisk och antar normalfördelning.
- **Effektivt antal fonder:** `1 / Σw²`. Det är antalet lika stora innehav som skulle ge samma koncentration.
- **Diversifieringskvot:** viktad fondvolatilitet delat med portföljvolatilitet. Högre betyder mer diversifiering.
- **Beta mot globala aktier:** hur mycket portföljen brukar röra sig när ACWI rör sig.

### Är ML-lagret värt att använda?

Titta på **rank-IC**, som skrivs ut efter backtesten och står i dashboardens sidhuvud. Den mäter hur väl modellens rangordning av tillgångsslagen stämde med utfallet, kvartal för kvartal.

- **Runt 0:** modellen gissar. Använd vanliga `HRP`.
- **Stadigt över cirka 0,05 och positiv de flesta kvartal:** det finns en signal värd att ta på allvar.

Jämför också `HRP + ML` med `HRP` i backtesten. Om ML-varianten inte är bättre efter kostnader tillför den inget.

## Begränsningar

- **Kort historik.** Flera fonder har bara några års data. Det ger ett fåtal kvartal att träna och utvärdera på, och ML-modeller hittar lätt mönster som inte finns.
- **Tröga kurser underskattar risken.** Backtestens avkastning räknas på faktiska kurser. För fonder med utjämnade kurser blir den uppmätta volatiliteten därför lägre än den verkliga. Den förväntade volatiliteten i innehavslistan räknas på justerad data och är den ärligare siffran.
- **Backtesten täcker ingen riktig krasch.** Ett litet största fall säger mer om perioden än om portföljen.
- **Koncentration till en förvaltare syns inte i volatiliteten.** Flera fonder kommer från samma förvaltare, och den risken fångas inte av modellen.
- **Yahoo Finance kan ha luckor eller fel**, särskilt för fonder. Kontrollera alltid datakvalitetstabellen.
- **Kostnader som inte ingår:** fondavgifter utöver det som redan ligger i NAV, skatt och ränta på kassa.

Historisk avkastning är ingen garanti för framtida avkastning. Skriptet är ett analysverktyg och ingen investeringsrådgivning.
