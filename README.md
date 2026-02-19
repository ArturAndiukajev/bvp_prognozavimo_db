# bvp_prognozavimo_db
Reikia sukurti .env faila savo projekto direktorijoje, ir tada gauti ALFRED duomenu API'u sioje nuorodoje https://fredaccount.stlouisfed.org/apikeys
tada tenai irasyti FRED_API_KEY=
po = irasyti savo gauta API.
run_ingest. py iskarto praleidzia ALFRED, Eurostat ir google trends duomenu atsiuntyma.

Šis projektas skirtas makroekonominių duomenų saugojimui ir realaus laiko (nowcasting / vintages) logikos palaikymui.
Sistema kaupia:
- **series** (laiko eilučių aprašus),
- **observations** (reikšmes su `period_date` ir `observed_at`/vintage),
- **releases** (kada atsisiųsta/įkelta ir koks tai snapshot/vintage),
- **ingestion_log** (įkėlimo žurnalą).

## Reikalavimai
- Python 3.10+ (arba 3.11)
- PostgreSQL (rekomenduojama per Docker)
- `.env` failas su reikalingais raktai (pvz. `FRED_API_KEY` ALFRED/FRED šaltiniams)

## Konfigūracija
- `datasets.yaml` – nurodo, kokius šaltinius ir kokius rodiklius/datasets įkelti.
  Pvz.:
  - `alfred`: serijų sąrašas (FRED serijų ID)
  - `eurostat`: dataset kodai + filtrai
  - `google_trends`: keywords + geo + timeframe
  - `yahoo_finance`: tickers + period (ir t.t.)

- `.env` – aplinkos kintamieji (pvz. `FRED_API_KEY`).

##Pagrindiniai valdymo skriptai
-docker-compose.yml - paleidžia PostgreSQL. Naudojamas lokaliam testavimui.
-create_schema.py- sukuria DB lenteles ir indeksus (schema). Naudoti, kai pirmą kartą reikia sukurti DB, arba po reset_db.py/DB išvalymo.
-reset_db.py- išvalo DB. Naudoti, kai reikia atstatyti sistemą nuo pat pradžios.
-full_reload.py - Atlieka pilną inicializaciją - sukuria schemą (arba tikrina, kad ji yra), paleidžia visus loaders, sukelia duomenis į DB. Naudoti integraciniam testui ir pirmajam pakrovimui.
-run_updates.py - paleidžia visų šaltinių update režimą - įkelia tik naujausius duomenis / naujus vintages, sukuria naujus release įrašus, neperrašo senų istorinių įrašų.
-run_ingest.py - paleidžia įkėlimą.

##Duomenų šaltinių loader’iai
-load_alfred.py - įkelia duomenis iš ALFRED (FRED real-time vintages). Svarbu: observed_at = realus vintage laikas, tame pačiame period_date gali būti keli skirtingi observed_at → revisions. Reikalinga: .env su FRED_API_KEY.
-fredmd.py - įkelia FRED-MD panelę (iš current.csv). Tai snapshot tipo šaltinis: observed_at = snapshot laikas (kada atsisiųsta), revisions realizuojami kaip naujas snapshot (naujas observed_at).
-load_eurostat.py - įkelia pasirinktus Eurostat datasets pagal datasets.yaml. vienas dataset → daug series (skirtingi dimensijų deriniai), observed_at = snapshot laikas.
-load_google_trends.py - įkelia Google Trends pagal keywords, geo, timeframe. observed_at = snapshot laikas, duomenys gali „kisti“ laikui bėgant, todėl snapshot/hashed release yra svarbus.
-load_financials.py - įkelia finansinius duomenis iš Yahoo Finance (pvz. indeksai, akcijos). update režime paima tik „uodegą“ (pvz. paskutines N dienų).

##Patikros / diagnostikos skriptai
-test_connection.py - greitas DB prisijungimo testas.
-check_db.py - patikrina, ar lentelės egzistuoja, kiek yra series/observations/releases ir pan.
-check_quality.py - kokybės patikra: dublių paieška, ar nėra keistų tuščių reikšmių, ar logika tvarkinga.
-get_vintage.py - pagalbinis skriptas darbui su vintages (pvz. gauti vintage informaciją ar testuoti vintage logiką).
