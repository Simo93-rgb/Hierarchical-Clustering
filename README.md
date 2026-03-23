# Hierarchical Clustering Ibrido (K-Means + Agglomerativo)

Relazione tecnica del progetto di clustering non supervisionato con pipeline ibrida.

Il repository implementa un approccio a due stadi:
- pre-clustering con K-Means per ridurre la cardinalita del dataset;
- clustering gerarchico agglomerativo custom sui centroidi ottenuti.

Obiettivo: rendere trattabili e leggibili dataset estesi (in particolare Anuran Calls / Frogs MFCCs), mantenendo un output interpretabile tramite dendrogramma, silhouette e metriche di coerenza dei cluster.

## 1. Obiettivi del lavoro

Gli obiettivi principali sono:
- implementare da zero il clustering agglomerativo con piu strategie di linkage;
- integrare un pre-processing K-Means per ridurre complessita computazionale;
- confrontare configurazioni diverse di linkage e riduzione K-Means;
- salvare risultati quantitativi e grafici per analisi sperimentale.

## 2. Cenni teorici

### 2.1 Clustering agglomerativo

Nel clustering gerarchico agglomerativo ogni osservazione parte come cluster singolo. A ogni iterazione vengono uniti i due cluster piu vicini fino a ottenere un solo cluster (o un numero fissato tramite taglio del dendrogramma).

### 2.2 Distanza tra cluster: idea generale

Il punto chiave non e solo la distanza tra singoli punti, ma la distanza tra *insiemi* (cluster).
Data una metrica di base $d(x_i, x_j)$, il linkage definisce una funzione $D(A, B)$ tra cluster $A$ e $B$.

Nel progetto la metrica base usata nelle run principali e euclidea.

### 2.3 Differenze tra i linkage analizzati

Di seguito le varianti usate nelle analisi:

1. Single linkage
	- Definizione: $$D(A,B)=\min_{x\in A, y\in B} d(x,y)$$
	- Effetto tipico: tende a creare catene (chaining), sensibile a punti ponte.

2. Complete linkage
	- Definizione: $$D(A,B)=\max_{x\in A, y\in B} d(x,y)$$
	- Effetto tipico: cluster piu compatti, maggiore separazione, piu sensibile agli outlier lontani.

3. Average linkage
	- Definizione: $$D(A,B)=\frac{1}{|A||B|}\sum_{x\in A}\sum_{y\in B} d(x,y)$$
	- Effetto tipico: compromesso tra single e complete, spesso piu stabile.

4. Centroid linkage
	- Definizione: $$D(A,B)=\|\mu_A-\mu_B\|_2$$
	- con $\mu_A, \mu_B$ centroidi dei cluster.
	- Effetto tipico: favorisce unioni guidate dalla posizione media; puo avere comportamenti diversi in presenza di cluster allungati.

5. Ward linkage
	- Definizione (incremento di varianza intra-cluster):
	$$D(A,B)=\frac{|A||B|}{|A|+|B|}\|\mu_A-\mu_B\|_2^2$$
	- Effetto tipico: tende a produrre cluster compatti e bilanciati minimizzando l'aumento della somma dei quadrati intra-cluster.

## 3. Pipeline ibrida del progetto

1. Caricamento dataset e preprocessing (normalizzazione e opzioni di riduzione feature).
2. Pre-clustering K-Means con un numero $k$ configurabile (`k_means_reduction`).
3. Clustering gerarchico agglomerativo custom sui centroidi K-Means.
4. Taglio del dendrogramma e/o selezione automatica del numero di cluster via silhouette.
5. Salvataggio di:
	- metriche in CSV;
	- dendrogrammi;
	- silhouette plot.

## 4. Struttura del progetto

- `main.py`: entrypoint CLI con argparse.
- `src/funzioni.py`: orchestrazione della pipeline e utility operative.
- `src/hierarchical_clustering.py`: implementazione custom dell'agglomerativo.
- `src/data.py`: caricamento/preprocessing dataset.
- `src/evaluation.py`: metriche e selezione cluster.
- `src/plot.py`: generazione grafici.
- `assets/Dataset/`: dataset sorgente.
- `assets/<dataset>/Results/`: risultati numerici.
- `assets/<dataset>/Plot/`: grafici.

## 5. Esecuzione

Setup ambiente:

```bash
uv venv
source .venv/bin/activate
uv pip install numpy pandas scipy scikit-learn matplotlib ucimlrepo
```

Help completo CLI:

```bash
uv run main.py --help
```

Esempio run singola:

```bash
uv run main.py --mode single --dataset Frogs_MFCCs --linkage average --distance euclidean --kmeans-reduction 25 --optimal-k -1
```

Esempio run multipla:

```bash
uv run main.py --mode multi --dataset iris_dataset --k-min 5 --k-max 20 --max-clusters 8
```

## 6. Risultati e discussione

I risultati vengono salvati nelle sottocartelle del dataset selezionato, separando metriche e grafici per configurazione.

La lettura consigliata e:
- confrontare prima metriche aggregate (precision/recall/F1/rand index);
- poi verificare coerenza del taglio con silhouette e dendrogramma;
- infine confrontare la stabilita rispetto a `k_means_reduction`.

## 7. Limiti attuali

Il progetto e orientato alla chiarezza didattica e sperimentale, non alla massima ottimizzazione computazionale. Per note operative, TODO e decisioni implementative future vedere [implementazioni.md](implementazioni.md).