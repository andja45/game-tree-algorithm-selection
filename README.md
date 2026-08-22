# Selekcija algoritma pretrage stabla igre

Seminarski rad iz Istraživanja podataka 2 (Matematički fakultet, Univerzitet u Beogradu).

## Opis projekta

Cilj projekta je predikcija, iz strukture sintetičkog stabla igre, koji od dva algoritma
pretrage, Alpha-Beta (AB) ili Monte Carlo Tree Search (MCTS), daje bolji rezultat na tom
stablu pod ograničenim budžetom broja posećenih čvorova.

Problem je instanca problema selekcije algoritma (Rice, 1976). Za svako stablo se pokreće
iscrpni minimax (bez ograničenja budžeta) i dva ograničena pretraživača, AB i MCTS, sa istim budžetom čvorova. Iz strukturnih atributa stabla i atributa ponašanja algoritama
trenira se klasifikator koji unapred predviđa čiji je izabrani potez bolji, bez potrebe da se ijedan od algoritama zaista pokrene.

U okviru projekta izvršene su sledeće faze:

- generisanje sintetičkog skupa stabala igre,
- eksplorativna analiza podataka i vizualizacija u 2D prostoru (PCA, t-SNE),
- pretprocesiranje: izvedeni atributi, čišćenje, kodiranje, standardizacija, izbor podskupa atributa,
- klasifikacija sa sedam algoritama i poređenje na tri skupa atributa (svi, strukturni, redukovani),
- modeli za više ciljnih promenljivih (`which_better`, `which_algo`, `ab_correct`, `mcts_correct`, `margin_category`, `tree_difficulty`),
- analiza na neviđenim konfiguracijama stabla.

Ceo tekst zadatka, opis podataka, obrade i rezultata nalazi se u
[`docs/izvestaj.tex`](docs/izvestaj.tex) (kompajlirana verzija:
[`docs/izvestaj.pdf`](docs/izvestaj.pdf)).

---

## Skup podataka

Skup je generisan skriptom `generate.py`. Sadrži:

- **16 978 stabala** (od 20 000 generisanih, ostatak odbačen kao degenerisan),
- **91 atribut**, od čega 81 prediktorski i 10 ciljnih/izvedenih,
- čvorove tipa MAX, MIN, CHANCE i LEAF, sa vrednostima listova u intervalu $[-1, 1]$,
- dva uporediva cilja klasifikacije: `which_algo` (da li je algoritam pronašao globalno
  optimalan potez) i `which_better` (čiji je od dva izabrana poteza vrednosno bolji).

## Pretprocesiranje podataka

Nad originalnim skupom izvršeni su sledeći koraci:

- konstrukcija izvedenih atributa (`ab_coverage`, `mcts_unanimous`, `depth_chance_inter`, `is_tie`, `tree_difficulty`),
- provera nedostajućih vrednosti i analiza autlajera (IQR i $3\sigma$ metoda),
- kodiranje kategoričkih ciljnih promenljivih (`OrdinalEncoder`, `LabelEncoder`),
- standardizacija numeričkih atributa (`StandardScaler`),
- izbor podskupa atributa ansambl glasanjem tri nezavisne grupe metoda: filter (korelacija, ANOVA), wrapper (RFE) i embedded (Random Forest).

Finalni pretprocesirani skup ima 33 atributa i čuva se u `data/trees_preprocessed.csv`.

## Korišćeni modeli

Za klasifikaciju su testirani sledeći algoritmi:

- Stablo odlučivanja
- Slučajne šume
- Logistička regresija
- XGBoost
- Neuronska mreža (MLP)
- Metoda potpornih vektora (SVM)
- Slaganje modela (Stacking)

Modeli su upoređeni na tri skupa atributa (svi originalni, samo strukturni, finalni
redukovani), unakrsnom validacijom (`StratifiedKFold`, k=10), sa macro-F1 kao primarnom
metrikom zbog neravnomerne raspodele klasa. Dodatno su izračunati preciznost, odziv, ROC AUC
i SHAP vrednosti važnosti atributa.

## Vizualizacije

U projektu su prikazane:

- raspodele klasa i ključnih atributa,
- korelaciona analiza sa ciljnim promenljivama,
- PCA i t-SNE projekcija stabala u 2D prostoru,
- matrice konfuzije za sve modele,
- dijagram faza (dominantna klasa u prostoru budžeta i gustine CHANCE čvorova),
- predikcija na neviđenim vrednostima faktora grananja.

---

## Struktura projekta

```
generate.py                  generator sintetičkog skupa
search/
    tree.py                  reprezentacija stabla (Node, NodeType, Tree)
    exact_solver.py          iscrpni Minimax
    alphabeta.py             AlphaBetaBudget, iterativno produbljivanje sa budžetom čvorova
    mcts.py                  MCTSBudget, UCT sa budžetom čvorova
data/
    trees10.csv              originalni skup (16 978 stabala, 91 atribut)
    trees_preprocessed.csv   pretprocesirani skup (33 atributa + ciljne kolone)
notebooks/
    01_eda.ipynb             eksplorativna analiza
    02_preprocessing.ipynb   pretprocesiranje i izbor atributa
    03_classification.ipynb  klasifikacija
    04_visualizations.ipynb  
models/
    scaler.pkl               naučeni StandardScaler
plots/                       sve figure koje notebook-ovi generišu
docs/
    izvestaj.tex, izvestaj.pdf   dokumentacija seminarskog rada
```

---

## Pokretanje projekta

### 1. Kreiranje virtuelnog okruženja

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalacija zavisnosti

```bash
pip install -r requirements.txt
```

### 3. Generisanje skupa podataka

```bash
python generate.py
```

Skripta generiše skup od 20 000 stabala i
čuva ga u `data/trees10.csv`. Broj stabala se menja preko `NUMBER_OF_TREES` na vrhu fajla.

### 4. Pokretanje notebook-ova

```bash
jupyter notebook
```

Notebook-ovi u `notebooks/` se pokreću redom, počevši od:

```
01_eda.ipynb
```

Svaki naredni notebook čita izlaz prethodnog: `02_preprocessing.ipynb` čita `trees10.csv` i
piše `trees_preprocessed.csv` i `models/scaler.pkl`, dok `03_classification.ipynb` i
`04_visualizations.ipynb` čitaju oba CSV fajla.

---

## Glavni nalazi

Budžet pretrage (`node_budget`) i margina između prva dva poteza iz korena
(`branch_top2_gap`) su najjači prediktori toga čiji je potez bolji; oba atributa se računaju
bez pokretanja ijednog algoritma. Klasifikacija čiji je potez bolji (`which_better`) je predvidljivija i bolje
generalizuje na neviđene konfiguracije od klasifikacije ko je pogodio tačan potez (`which_algo`), jer
ne meša kvalitet algoritma sa opštom težinom stabla.

Detaljni rezultati, tabele i figure nalaze se u [`docs/izvestaj.tex`](docs/izvestaj.tex).

## Korišćene biblioteke

- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- Jupyter

Tačne verzije su navedene u [`requirements.txt`](requirements.txt).
