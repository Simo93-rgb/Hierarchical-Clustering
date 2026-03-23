# Hierarchical Clustering Ibrido (K-Means + Agglomerativo)

Relazione tecnica del progetto di clustering non supervisionato con pipeline ibrida.

Il repository implementa un approccio a due stadi:

  - pre-clustering con K-Means per ridurre la cardinalità del dataset;
  - clustering gerarchico agglomerativo custom sui centroidi ottenuti.

Obiettivo: rendere trattabili e leggibili dataset estesi (in particolare Anuran Calls / Frogs MFCCs), mantenendo un output interpretabile tramite dendrogramma, silhouette e metriche di coerenza dei cluster.

## 1\. Obiettivi del lavoro

Gli obiettivi principali sono:

  - implementare da zero il clustering agglomerativo con più strategie di linkage;
  - integrare un pre-processing K-Means per ridurre complessità computazionale;
  - confrontare configurazioni diverse di linkage e riduzione K-Means;
  - salvare risultati quantitativi e grafici per analisi sperimentale.

## 2\. Cenni teorici

### 2.1 Clustering agglomerativo

Nel clustering gerarchico agglomerativo ogni osservazione parte come cluster singolo. A ogni iterazione vengono uniti i due cluster più vicini fino a ottenere un solo cluster (o un numero fissato tramite taglio del dendrogramma).

### 2.2 Distanza tra cluster: idea generale

Il punto chiave non è solo la distanza tra singoli punti, ma la distanza tra *insiemi* (cluster).
Data una metrica di base $d(x_i, x_j)$, il linkage definisce una funzione $D(A, B)$ tra cluster $A$ e $B$.

Nel progetto la metrica base usata nelle run principali è euclidea.

### 2.3 Differenze tra i linkage analizzati

Di seguito le varianti usate nelle analisi:

1.  **Single linkage**

      - Definizione: $$D(A,B)=\min_{x\in A, y\in B} d(x,y)$$
      - Effetto tipico: tende a creare catene (chaining), sensibile a punti ponte.

2.  **Complete linkage**

      - Definizione: $$D(A,B)=\max_{x\in A, y\in B} d(x,y)$$
      - Effetto tipico: cluster più compatti, maggiore separazione, più sensibile agli outlier lontani.

3.  **Average linkage**

      - Definizione: $$D(A,B)=\frac{1}{|A||B|}\sum_{x\in A}\sum_{y\in B} d(x,y)$$
      - Effetto tipico: compromesso tra single e complete, spesso più stabile.

4.  **Centroid linkage**

      - Definizione: $$D(A,B)=\|\mu_A-\mu_B\|_2$$
      - con $\mu_A, \mu_B$ centroidi dei cluster.
      - Effetto tipico: favorisce unioni guidate dalla posizione media; può avere comportamenti diversi in presenza di cluster allungati.

5.  **Ward linkage**

      - Definizione (incremento di varianza intra-cluster):
        $$D(A,B)=\frac{|A||B|}{|A|+|B|}\|\mu_A-\mu_B\|_2^2$$
      - Effetto tipico: tende a produrre cluster compatti e bilanciati minimizzando l'aumento della somma dei quadrati intra-cluster.

## 3\. Pipeline ibrida del progetto

1.  Caricamento dataset e preprocessing (normalizzazione e opzioni di riduzione feature).
2.  Pre-clustering K-Means con un numero $k$ configurabile (`k_means_reduction`).
3.  Clustering gerarchico agglomerativo custom sui centroidi K-Means.
4.  Taglio del dendrogramma e/o selezione automatica del numero di cluster via silhouette.
5.  Salvataggio di:
      - metriche in CSV;
      - dendrogrammi;
      - silhouette plot.

## 4\. Scelta dei Parametri in Ambito Non Supervisionato

In assenza di una *ground truth* nota (es. un dataset sul vino senza classi), la giustificazione dei parametri dell'algoritmo non può basarsi su metriche esterne come Precision o Recall, ma deve affidarsi alla struttura intrinseca dei dati e a metriche di validazione interna.

### 4.1 Numero di micro-cluster per il pre-processing ($k$)

Nel nostro approccio ibrido, il K-Means non ha lo scopo di trovare i cluster finali, ma di generare dei "micro-cluster" (o prototipi) che sintetizzino la distribuzione dei dati originali, abbattendo i tempi di calcolo dell'agglomerativo. Come si giustifica il valore di $k$?

  * **Regola pratica (Rule of Thumb):** Il parametro $k$ deve essere sufficientemente piccolo da garantire un vantaggio computazionale, ma molto più grande del numero atteso di cluster finali. Una regola empirica diffusa suggerisce $k \approx \sqrt{N/2}$, dove $N$ è il numero di campioni totali.
  * **Stabilità delle metriche:** Si può giustificare un valore di $k$ dimostrando empiricamente che la sua variazione non degrada la qualità finale. Se calcolando la Silhouette finale passando da $k=100$ a $k=50$ il punteggio resta stabile, significa che 50 micro-cluster sono sufficienti a mantenere inalterata la topologia originale del dataset senza perdita di informazione critica.

### 4.2 Determinazione del numero di cluster finali ("a priori")

Quando non si conoscono le famiglie o le classi in anticipo, il numero di cluster finali non è imposto, ma viene "scoperto" analizzando il comportamento dell'algoritmo:

  * **Analisi visiva del Dendrogramma:** È il pregio maggiore del clustering gerarchico. Osservando il dendrogramma, si cerca il "salto" verticale più lungo (la maggiore distanza sull'asse Y) che non viene intersecato da unioni orizzontali. Quel salto indica una forte e naturale distanza tra gruppi di dati. Tracciando una linea di taglio orizzontale in quel punto, l'albero restituisce il numero intrinseco di cluster. \* **Massimizzazione del Silhouette Score:** Si calcola il Coefficiente di Silhouette medio dell'intero modello al variare dei possibili tagli (es. tagliando l'albero per avere da 2 a 10 cluster). Si giustifica la scelta finale eleggendo il numero di cluster che massimizza lo score medio (più vicino a +1) e che presenta sagome omogenee e senza picchi negativi evidenti nel plot. \* **Metodo del Gomito (Elbow Method):** Tracciando la devianza o inerzia intra-cluster (WCSS) al variare dei cluster, si seleziona il punto in cui la curva subisce una brusca frenata nella sua discesa, formando un "gomito". Quel punto rappresenta il compromesso ideale tra un numero ridotto di cluster e un'alta coesione interna.

## 5\. Struttura del progetto

  - `main.py`: entrypoint CLI con argparse.
  - `src/funzioni.py`: orchestrazione della pipeline e utility operative.
  - `src/hierarchical_clustering.py`: implementazione custom dell'agglomerativo.
  - `src/data.py`: caricamento/preprocessing dataset.
  - `src/evaluation.py`: metriche e selezione cluster.
  - `src/plot.py`: generazione grafici.
  - `assets/Dataset/`: dataset sorgente.
  - `assets/<dataset>/Results/`: risultati numerici.
  - `assets/<dataset>/Plot/`: grafici.

## 6\. Esecuzione

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

## 7\. Risultati e discussione

I risultati vengono salvati nelle sottocartelle del dataset selezionato, separando metriche e grafici per configurazione.

La lettura consigliata è:

  - confrontare prima metriche aggregate (precision/recall/F1/rand index);
  - poi verificare coerenza del taglio con silhouette e dendrogramma;
  - infine confrontare la stabilità rispetto a `k_means_reduction`.

