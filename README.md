
## Lab 1 (MEGA) — Python + Data Analysis Stack (NumPy → Pandas → EDA → prep)

**Opis:** szybki start w środowisku (Colab/Jupyter), NumPy jako narzędzie numeryczne, Pandas do składania danych, podstawowa EDA i przygotowanie do modelowania.
**Zadania:**

1. (Setup) Sprawdź wersje bibliotek, ustaw `seed`, zrób mini-test deterministyczności.
2. NumPy: broadcasting + wektoryzacja (pętla vs wektor) na MSE / odległościach.
3. Stabilność numeryczna: `float32` vs `float64` na prostej operacji (np. sumowanie, softmax-stable demo).
4. Pandas: wczytaj dataset (np. seaborn/titanic) i policz podstawowe statystyki + missingness.
5. EDA: 3 wykresy (rozkład targetu, 2 cechy vs target, heatmap korelacji).
6. (Home) Data assembly: 2–3 tabele syntetyczne → merge/join → feature table.
7. (Home) Czyszczenie: typy, duplikaty, imputacja, outliery (IQR) + krótka lista decyzji.
8. (Home) Split bez leakage + baseline model (sklearn) + 2 wnioski “co poprawić”.

---

## Lab 2 — Perceptron & XOR: separowalność liniowa + “naprawy”

**Opis:** perceptron jako klasyfikator liniowy, geometria separowalności, XOR jako przykład porażki, porównanie z nieliniowymi modelami i prosty feature map / małe MLP.
**Zadania:**

1. Wygeneruj AND/OR/XOR na {0,1}² (z małym szumem) i narysuj scatter.
2. Zaimplementuj perceptron (fit/predict) lub użyj sklearn i sprawdź accuracy dla AND/OR/XOR.
3. Narysuj granicę decyzyjną (meshgrid) dla perceptronu na XOR.
4. Zademonstruj “naprawę” XOR przez feature map: dodaj cechę (z=x_1x_2).
5. Porównaj z SVM RBF i małym MLP (1 hidden layer) — jedna figura na model.
6. Pytanie: “co zmieniło się — model czy przestrzeń cech?” + krótki wniosek.

---

## Lab 3 (OPT-1) — GD + Armijo + 2D trajectories (conditioning & non-convex)

**Opis:** GD jako metoda numeryczna, wpływ kroku na stabilność, line search Armijo, trajektorie w 2D i rola uwarunkowania.
**Zadania:**

1. Zdefiniuj f1,f2,f3 (1D convex + non-convex) i ich gradienty.
2. Zaimplementuj `gradient_descent()` zwracający historię (x_k).
3. Eksperyment lr: “za mały / dobry / za duży” + wykresy (x_k) i (f(x_k)).
4. Zaimplementuj Armijo backtracking i porównaj fixed-lr vs Armijo na f3.
5. Przejdź do 2D: ill-conditioned quadratic (a≠b) + contour + trajektoria.
6. Non-convex 2D (Himmelblau lub saddle): porównaj dwie inicjalizacje.

---

## Lab 4 (OPT-2) — SGD noise + batch size + momentum/Nesterov + SGLD

**Opis:** SGD jako optymalizacja “empirical risk”, mini-batch noise, momentum/Nesterov jako przyspieszenie, SGLD jako intuicja SDE (temperatura i eksploracja).
**Zadania:**

1. Zbuduj synthetic regression (A,b) i funkcję `full_loss(w)` + `full_grad(w)`.
2. Zaimplementuj SGD z mini-batch (history + loss history).
3. Porównaj batch_size: 1 vs 16 vs full; pokaż “szum” na loss/trajectory.
4. Dodaj momentum i porównaj do SGD (czas/krzywe).
5. Dodaj Nesterov i porównaj do momentum.
6. Zaimplementuj SGLD na 2D non-convex (Himmelblau/double-well) i porównaj trajektorie dla różnych T.

---

## Lab 5 — Keras MLP: pierwsza sieć + diagnostyka uczenia

**Opis:** budowa i trening MLP w Keras, metryki, krzywe uczenia, overfitting, podstawy dobrej praktyki.
**Zadania:**

1. Wczytaj MNIST/Fashion-MNIST, normalizacja, split train/val.
2. Zbuduj prosty MLP (Flatten → Dense → Dense).
3. Trening + wykres loss/accuracy dla train i val.
4. Dodaj EarlyStopping i porównaj wyniki.
5. Confusion matrix + analiza dwóch typowych pomyłek modelu.
6. Mini-eksperyment: zmień liczbę neuronów/warstw i porównaj overfitting.

---

## Lab 6 — Regularization & generalization (praktycznie)

**Opis:** jak ograniczać overfitting i poprawiać generalizację: L2, dropout, data augmentation (light), kalibracja/pewność (opcjonalnie).
**Zadania:**

1. Weź model z Lab 5 jako baseline.
2. Dodaj L2 i porównaj krzywe uczenia.
3. Dodaj dropout i porównaj.
4. (Opcjonalnie) Prosta augmentacja (np. `tf.image` lub Keras preprocessing).
5. Porównaj metryki: accuracy + (jeśli czas) calibration curve / max-softmax confidence histogram.
6. Podsumuj: które techniki działają i dlaczego (intuicja bias–variance).

---

## Lab 7 — Numerical issues in NN: vanishing/exploding + init + clipping + log-sum-exp

**Opis:** skąd biorą się NaNy, zanikające/wybuchające gradienty; jak temu przeciwdziałać.
**Zadania:**

1. Zbuduj głębsze MLP z tanh/sigmoid i monitoruj normy gradientów (callback / custom loop light).
2. Zmień aktywacje na ReLU i porównaj.
3. Porównaj inicjalizacje Xavier vs He (na tej samej architekturze).
4. Dodaj gradient clipping i zobacz wpływ na stabilność.
5. Zademonstruj log-sum-exp trick na softmax (overflow w “naiwnej” wersji).
6. Checklist “debug NaN/Inf”.

---

## Lab 8 — Autoencoder for anomaly detection (cyber/monitoring)

**Opis:** AE jako model rekonstrukcji; anomaly score i dobór progu; metryki ROC/PR i ryzyka fałszywych alarmów.
**Zadania:**

1. Przygotuj dataset: “normal” vs “anomaly” (np. MNIST: digit=0 normal, reszta anomaly) albo syntetyk.
2. Zbuduj prosty autoencoder (encoder–bottleneck–decoder).
3. Naucz na “normal” i policz reconstruction error dla test.
4. Zrób histogram błędów dla normal vs anomaly.
5. Ustal próg: np. percentile / optymalizacja F1; policz ROC/PR.
6. Omów: trade-off false positives vs false negatives (cyber/monitoring).

---

## Lab 9 — GAN (light DCGAN): generowanie + failure modes

**Opis:** generatywne modelowanie przez grę generator–dyskryminator; stabilność i mode collapse.
**Zadania:**

1. Przygotuj prosty DCGAN na MNIST (mały, szybki).
2. Trening przez kilka epok + zapis próbek co N kroków.
3. Zidentyfikuj symptomy: niestabilność, collapse, “brzydkie próbki”.
4. Zastosuj 1–2 stabilizacje (label smoothing / noise / tuning LR).
5. Porównaj jakościowo (grid próbek) przed vs po.
6. Krótkie podsumowanie: dlaczego GAN jest trudniejszy niż AE.

---

## Lab 10 — LLM Lab (practical foundations): tokenization, embeddings, mini-RAG, safety

**Opis:** praktyczna “alfabetyzacja LLM”: tokeny i kontekst, embeddings i wyszukiwanie semantyczne, mini-RAG i ryzyka (hallucinations, prompt injection).
**Zadania:**

1. Tokenization demo: jak rośnie liczba tokenów vs długość tekstu; limit kontekstu (intuicja).
2. Embeddings: policz embeddingi 20 zdań i zrób semantic search (cosine similarity).
3. Zbuduj mini-korpus (10–20 krótkich dokumentów) i retrieval top-k.
4. Mini-RAG: generuj odpowiedź “only from retrieved context” + cytowanie źródeł.
5. Hallucination test: pytanie poza kontekstem → jak to wykryć (refusal / “I don’t know”).
6. Prompt injection mini-demo: dokument zawiera złośliwą instrukcję → jak bronić (system prompt / filtering / policy).

*(Może być z API lub bez, zależnie od Twoich ograniczeń — workflow zostaje ten sam.)*

---

## Lab 11 — Interpretability + ethics + monitoring (use cases)

**Opis:** interpretowalność (SHAP/LIME), pułapki interpretacji, dataset shift, monitoring w produkcji i odpowiedzialne AI.
**Zadania:**

1. Trenuj prosty model (np. XGBoost/RandomForest/LogReg na tabular).
2. SHAP: top features global + wyjaśnienie jednej predykcji.
3. Pokaż pułapkę: korelacja cechy z targetem przez leakage / proxy.
4. Dataset shift: zasymuluj drift (zmiana rozkładu jednej cechy) i sprawdź spadek jakości.
5. Monitoring: policz PSI/KL na cechach i zaproponuj alert threshold.
6. Mini-debata: 3 ryzyka + 3 mitigacje (privacy, fairness, transparency).

---

## Lab 12 — PINN “hello world”: ODE/PDE residual + BC/IC

**Opis:** PINN: loss = data/BC/IC + residual PDE/ODE; collocation points; wrażliwość na ważenie składników.
**Zadania:**

1. Wybierz prosty problem: 1D Poisson lub ODE (u'(t)= -u) z warunkiem początkowym.
2. Zbuduj sieć (u_\theta(x)) i policz residual (autodiff).
3. Dodaj BC/IC do loss i trenuj.
4. Zmień liczbę collocation points i porównaj wynik.
5. Zmień wagi składowych loss i zobacz wpływ.
6. Podsumuj: “co było najtrudniejsze i dlaczego”.

---

## Lab 13 — Model identification (ODE/SDE-flavored) with NN / gray-box

**Opis:** identyfikacja parametrów modelu dynamicznego: “data + physics”.
**Zadania (wariant ODE):**

1. Wygeneruj dane z logistycznego wzrostu lub SIR (z szumem).
2. Dopasuj parametry klasycznie (least squares) jako baseline.
3. Zbuduj NN do aproksymacji rozwiązania + ucz parametry (hybrid loss).
4. Porównaj: baseline vs hybrid (błąd parametrów i predykcji).
5. Test generalizacji: inny zakres czasu / inne IC.
6. Wnioski: identyfikowalność i wpływ szumu.

*(Wariant SDE: OU process – estymacja drift/diffusion, jeśli chcesz bardziej “stochastycznie”.)*

---

## Lab 14 — RL: Tic-Tac-Toe (tabular) + exploration + OU intuition

**Opis:** RL w czystej formie na kółko-krzyżyk: stany, akcje, nagrody, Q-learning/SARSA; eksploracja; krótko o OU jako “smooth noise” w continuous control.
**Zadania:**

1. Zaimplementuj środowisko Tic-Tac-Toe (state, legal moves, terminal).
2. Zaimplementuj epsilon-greedy policy.
3. Zaimplementuj Q-learning albo SARSA i trenuj vs prosty przeciwnik.
4. Wykres: win/draw rate w czasie + wpływ epsilon schedule.
5. Reward shaping: porównaj 2 warianty nagrody (ostrożnie!).
6. (Mini) OU intuition: wygeneruj OU noise i porównaj z Gaussian (wykresy); opcjonalnie Gym Pendulum z “noise injection”.
