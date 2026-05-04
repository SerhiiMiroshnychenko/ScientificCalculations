# Технічний журнал змін у статті за зауваженнями Reviewer 3 (Round 2)

**Файли:**
- Оригінал: `D:\WINDSURF\ARTICLEs\BJMC\new-04.05.2026\Стаття_0.md` (305 рядків)
- Поточна версія: `D:\WINDSURF\ARTICLEs\BJMC\new-04.05.2026\Стаття_.md` (327 рядків)
- Відповіді рецензентам: `D:\WINDSURF\ARTICLEs\BJMC\new-04.05.2026\Відповіді.md`

**Призначення документа:** внутрішній робочий журнал «Було / Стало / Зауваження №» для відстеження походження кожної редакційної зміни.

---

## 1. Abstract

**Зауваження:** R3 Comment 1 (contribution must be explicitly described) + R3 Comment 2 (methodology/approach vs method).

**Було** (~154 слова; оригінал `Стаття_0.md` L15):
> «…This study develops a multi-algorithmic **methodology** to identify and rank factors… The **approach** integrates five feature significance methods… The unique contribution lies in the comprehensive multi-method **approach**, which provides robust feature ranking that compensates for individual method limitations, enabling automated prediction adaptable to various enterprise resource planning systems for transaction management optimization and enterprise decision-making.»

**Стало** (149 слів; поточний `Стаття_.md` L15):
> «…This study develops a multi-algorithmic **method** to identify and rank factors… It integrates five feature-significance methods… **The contribution to the body of scientific knowledge is twofold: (i) a method-neutral ranking rule that compensates for individual method limitations, and (ii) an empirical taxonomy of B2B order success determinants for ERP-based decision-support modules.**»

**Характер правок:** (а) `methodology/approach` → `method`; (б) фінальне речення про «unique contribution» замінено на явне twofold-contribution у форматі (i)+(ii); (в) загальна довжина вкладається в формальний ліміт BJMC ≤150 слів.

---

## 2. Introduction (Section 1)

**Зауваження:** R3 Comment 3 (explicit R&D problems + contribution) + R3 Comment 4 (Related Work content leaked into Introduction).

**Було** (оригінал L19–L30; 12 абзаців): мотивація → огляд ML/AI → divided opinions → B2B success-factor studies → rank-aggregation literature → попередні роботи авторів → research aim → findings preview.

**Стало** (поточний L19–L34): мотивація → короткий місток до Related Work → **блок Problem statement (P1, P2, P3)** → попередні роботи авторів → research aim → **блок Contributions (C1, C2, C3, C4)** → findings preview.

**Нові елементи:**
- **P1** — відсутність method-neutral aggregation для B2B success factors.
- **P2** — відсутність формально специфікованого правила консолідації розбіжних ранжувань.
- **P3** — відсутність відтворюваного pipeline-опису для ERP-based settings.
- **C1** — multi-algorithmic feature-significance method (п'ять методів у єдиному пайплайні).
- **C2** — modified Borda rank-aggregation rule (mean-of-ranks замість sum-of-positions).
- **C3** — empirical taxonomy of B2B order success determinants.
- **C4** — reproducible processing pipeline.

**Винесено в нову секцію 2:** абзаци про ML/AI у sales forecasting, divided opinions, B2B success-factor studies, rank-aggregation literature.

---

## 3. Related Work (нова Section 2)

**Зауваження:** R3 Comment 4.

**Було:** секції не існувало. Уся літературна база була частиною Introduction.

**Стало** (поточний L36–L41): окрема секція з чотирма абзацами:
1. ML/AI у sales forecasting (Hrischev, Lin, Zoltners, Bauskar, Rohaan).
2. Divided-opinions debate (Rohaan, Miroshnychenko, Bauskar, Lin).
3. Прямі B2B success-factor studies (Eid, Zoltners, Wilson, Günther, Høgevold, Rodriguez).
4. Rank-aggregation в сусідніх доменах (Dwork — web search; Wald, Sarkar — bioinformatics; Vambol — environmental decision-making) з містком до власної модифікації Borda.

**Наслідок:** перенумерація подальших секцій (`Materials and Methods` 2→3, `Results` 3→4, `Discussion` 4→5, `Conclusions` 5→6).

---

## 4. Materials and Methods (Section 3) — opener

**Зауваження:** R3 Comment 2 (термінологія).

**Було** (`Стаття_0.md` L34):
> «The **methodology** for a comprehensive analysis… integrates methods from different areas… To implement the proposed **approach**, we first perform data preparation…»

**Стало** (`Стаття_.md` L45):
> «The proposed **method** for a comprehensive analysis… integrates feature-significance methods from different areas…, consolidated through a single rank-aggregation rule. The method comprises three sequential stages: type-specific data preprocessing (Fig. 6), per-method feature-significance assessment, and modified Borda rank aggregation (Fig. 7). To implement the proposed **method**, we first perform data preparation…»

---

## 5. Stage-by-stage опис під Fig. 6

**Зауваження:** R3 Comment 5 (preprocessing scheme too generic).

**Було:** у `Стаття_0.md` опис pre-processing був у формі довгих прозових абзаців (L52–L58), без чіткого розділення на стадії та без конкретних параметрів/алгоритмів.

**Стало** (`Стаття_.md` L79): структурований блок після caption Fig. 6, із стадіями **I / II / III-a / III-b / III-c**:

### Stage I (Feature engineering and removal of irrelevant variables)
- Пайплайн `build_b2b_from_pgadmin_export.py` трансформує сирий PostgreSQL-експорт у `cleanest_data.csv` (25 ознак).
- Видалено 3 категорії:
  - *технічні ID*: `id`, `order_id`, `partner_id`, `customer_id`;
  - *низькопредиктивні B2B-атрибути*: `sales_team`, `customer_category`, `customer_country`, `product_categories`, `payment_term`, `delivery_method`;
  - *post-event target-leakage*: `processing_time_hours`.
- Видалено службові `_*`-змінні.
- **Temporal leakage prevention** для `partner_*` (віднімання внеску поточного замовлення з кумулятивних сум).
- **Near-zero-variance filter** з порогом 95% однакових значень.

### Stage II (Missing values)
- Медіанна імпутація для числових, модальна — для категоріальних.

### Stage III-a (Temporal)
- Вилучення `day_of_week`, `month`, `quarter`, `hour_of_day`, `create_date_months`.
- Циклічне sin/cos-кодування через `feature_engine.creation.CyclicalFeatures`.

### Stage III-b (Categorical)
- OHE Extended Compact, варіант **High Distribution First (HDF)** (Ul Haq et al., 2019).
- Застосовано до `salesperson` (31 значення) і `source` (66 значень).
- `d+1` слот для unseen-значень.

### Stage III-c (Numerical)
- Перелічено 17 числових ознак.
- `RobustScaler` (median + IQR).
- Детекція викидів за IQR-правилом (Dekking et al., 2005).

---

## 6. Stage-by-stage опис під Fig. 7

**Зауваження:** R3 Comment 5 (similar question about Fig. 7).

**Було:** у `Стаття_0.md` опис після Fig. 7 (≈L94) був коротким прозовим абзацом «First, for each significance evaluation method, we form an ordered list…», без імплементаційних деталей.

**Стало** (`Стаття_.md` L111): структурований опис зі стадіями **I / II / III / Output** і явними викликами бібліотек:

| Метод | Виклик | Параметри |
|---|---|---|
| AUC | `sklearn.metrics.roc_auc_score` | univariate проти `is_successful` |
| MI | `sklearn.feature_selection.mutual_info_classif` | Kraskov–Stögbauer–Grassberger NN-оцінювач, фіксований `random_state` |
| dCor | `dcor.distance_correlation` | — |
| LogReg | `sklearn.linear_model.LogisticRegression` | L2, стандартизовані \|coef\| |
| DecTree | `sklearn.tree.DecisionTreeClassifier` | Gini, CART-зростання |

Додано явну прив'язку: Stage III → Eq. (1) + Eq. (3) (на контрасті з Eq. (2)); вихід → Table 1 + Fig. 8, що використовуються в Sections 4–5.

---

## 7. Рівняння (1)–(3)

**Зауваження:** R3 Comment 6 (equations not explicitly used with referencing).

**Було** (`Стаття_0.md` L71, L77, L82): рівняння підписані номерами, але текст не містив фраз типу «Eq. (1)», «Eq. (2)», «Eq. (3)».

**Стало:**

| Рядок | Було | Стало |
|---|---|---|
| L88 (Methods) | «…is defined as:» | «…is defined **by Eq. (1)**:» |
| L94 | «…gives each element a score:» | «…gives each element a score, **given by Eq. (2)**:» |
| L99 | «…applies arithmetic mean of ranks:» | «…applies the arithmetic mean of method-specific ranks, **given by Eq. (3)**:» |
| L104 (нова synthesis-фраза) | — | *«The consolidated ordering reported in Table 1 and Fig. 8 is obtained by applying Eq. (1) to each (feature, method) pair and then aggregating the resulting per-method ranks via Eq. (3); Eq. (2) is reproduced here only as the classical reference formulation…»* |
| L111 (Fig. 7 опис) | — | «…transformed into ranks using **Eq. (1)**; aggregated… using **Eq. (3)**, in contrast to the classical sum-of-positions form given by **Eq. (2)**.» |
| L196 (Discussion) | — | «…the modified Borda mean rank of **Eq. (3)**…» |
| L203 (Conclusions) | — | «…modified Borda rank-aggregation rule based on the arithmetic mean of method-specific ranks (**Eq. (3)**)…» |

---

## 8. Інтерпретація Figs. 9–13 (Section 4, Results)

**Зауваження:** R3 Comment 7 (no proper interpretation to derive novel knowledge).

**Було** (`Стаття_0.md` L156): один загальний абзац на ≈5 речень після Fig. 13:
> «The presented visualizations serve as marginal (single-feature) illustrations that complement the multi-method feature ranking… Communication intensity and order flexibility exhibit clear monotonic positive associations… The nonlinear, inverted U-shaped dependence of `order_amount` explains the divergent importance scores… Discount and temporal variables show negligible or sample-limited effects… These empirical patterns reinforce the quantitative findings…»

**Стало** (`Стаття_.md` L177–L183): по одному окремому абзацу на кожен рисунок у форматі чотирьох явних шарів:

| Рисунок | Observation | Mechanism | Novelty | Implication |
|---|---|---|---|---|
| Fig. 9 `order_messages` | saturating S від ≈8% (1–3 msg) до ≈90% (17+) | sustained dialogue як проксі спільного розв'язання задач | перша робота з такою S-формою, відтвореною 5 методами | дискретні тригери на порозі 9 повідомлень |
| Fig. 10 `order_changes` | 46% → 85% монотонно | ітеративна ревізія = реляційний капітал | інверсія «changes=warning» у B2B | re-calibration alert-rules |
| Fig. 11 `order_amount` | перевернута U, пік 300–1000 USD | B2B sweet spot vs bureaucratic friction | механістичне пояснення розбіжності LogReg/MI vs AUC/DecTree | не-монотонна трансформація |
| Fig. 12 `discount_total` | вузька підвибірка, слабкий позитивний тренд | discounts медійовані реляційними факторами | B2C-логіка знижок не переноситься | interaction features замість агрегованого discount |
| Fig. 13 `day_of_week` | 61–72%, без патерну | ERP-driven scheduling поглинає day-level ефект | negligible всупереч (Guo et al., 2021) | де-пріоритизувати fine-grained temporal |

---

## 9. Discussion (Section 5)

**Зауваження:** R3 Comment 8 (discuss novelty, provide theoretical and practical implications).

**Було** (`Стаття_0.md` L158–L169): факторно-фактна перевірка відповідності з попередніми роботами (5 абзаців) + cross-methodological triangulation (1 абзац) + limitations (1 абзац). Відсутні явні параграфи про новизну, теоретичні та практичні implications.

**Стало** (`Стаття_.md` L186–L199): зберігаю перші 7 абзаців (factor-by-factor confirmation + triangulation) без змін, додаю три нові bold-led параграфи між triangulation і limitations:

### 9.1 «Distinguishing confirmatory from novel contributions» (новий)
- Явне відокремлення confirmatory (factor-by-factor agreement) від novel:
  - (a) modified Borda mean rank з explicitly argued properties;
  - (b) cross-method stability як reliability criterion;
  - (c) negative finding: discount + temporal дом'ятно підпорядковані relational variables.

### 9.2 «Theoretical implications» (новий)
- Relational view of B2B transactions.
- Cross-method stability як quantifiable reliability criterion, що доповнює predictive accuracy (Rohaan, Bauskar).

### 9.3 «Practical implications» (новий)
- Integration pattern в Odoo `sale.order`.
- Новий KPI: *cross-method stability index* (SD рангу між методами).
- Перехід від generic activity counts до thresholded relational events.

---

## 10. Conclusions (Section 6)

**Зауваження:** R3 Comment 1 + R3 Comment 2.

**Було** (`Стаття_0.md` L172):
> «This study identified key factors… using a multi-algorithmic **approach** that combines five feature significance evaluation methods (AUC, MI, dCor, LogReg, DecTree) and aggregates results via rank reconciliation… The proposed **framework** is intended as a generalizable **methodological approach**… The proposed **methodology** can be implemented…»

**Стало** (`Стаття_.md` L203):
- `approach`/`methodology` → `method` / `framework` (згідно з R3#2);
- Додано явне речення про внесок: «Beyond these factor-level findings, the principal scientific contributions of the work are (i) a method-neutral ranking method whose properties… are explicitly argued for, and (ii) a *cross-method stability* perspective on feature importance, complementing the predictive-accuracy criterion…».
- Посилання на `Eq. (3)` у формулюванні rank-aggregation rule.

---

## 11. Термінологічна уніфікація (наскрізно)

**Зауваження:** R3 Comment 2.

Введено єдину конвенцію:

| Термін | Значення | Застосування |
|---|---|---|
| **method** | формалізована процедура з input/output | наш внесок + 5 методів значущості + рank-aggregation rule |
| **algorithm** | конкретна обчислювальна реалізація | CART, Kraskov–Stögbauer–Grassberger |
| **technique** | допоміжна процедура | RobustScaler, OHE-EC, modified Borda |
| **methodology** | більше не вживається як синонім method | залишено тільки в описовому сенсі «methodological limitations/advantages» |
| **approach** | тільки як high-level strategy | поодинокі вживання у прозовому сенсі |

**Ключові заміни:**
- `comprehensive approach to analyzing` → `comprehensive coverage of` (L75).
- `using different methodological approaches` → `using different methods` (L193, Discussion).
- `the methodology/the approach` → `the method` (opener Methods, Conclusions).
- `evaluation techniques` → `evaluation methods` (Conclusions).
- `cross-validation system` → лишено в лапках як термін.

---

## 12. Дрібні технічні правки

| Де | Було | Стало | Причина |
|---|---|---|---|
| Fig. 7 опис | «(Sékely et al., 2007)» | «(Székely et al., 2007)» | помилка прізвища |
| Fig. 7 опис | «modelling» | «modeling» | US English |
| Bibliography references | без змін | без змін | нових позицій не додано |

---

## 13. Зведення відповідності Comment → розділи, що змінено

| Comment | Основні розділи з правками |
|---|---|
| 1 (contribution) | Abstract, Section 1 (Contributions block), Section 6 |
| 2 (terminology) | усі розділи (наскрізна уніфікація) |
| 3 (R&D problems) | Section 1 (Problem statement block) |
| 4 (Related Work) | Section 1 → Section 2 (виділення), перенумерація 3–6 |
| 5 (Fig. 6 / Fig. 7 detail) | Section 3 (параграфи після caption Fig. 6 і Fig. 7) |
| 6 (Eqs. referencing) | Section 3, Section 5, Section 6 |
| 7 (Figs interpretation) | Section 4 (параграфи після Fig. 13) |
| 8 (novelty/implications) | Section 5 (три нові bold-led параграфи) |

---

## 14. Статус внутрішніх TODO

| Пункт | Статус |
|---|---|
| Stage I — конкретний опис на основі `Stage_I_Preprocessing_Report.md` | ✅ виконано (див. §5) |
| Abstract ≤150 слів зі збереженням contribution statement | ✅ виконано (149 слів, див. §1) |
| OHE-EC HDF-варіант | ✅ підтверджено (§5, Stage III-b) |
| Формат response letter | ✅ залишено як у раніше прийнятих Відповідях 1–2 |
| Внутрішня узгодженість section/fig/eq/table посилань | ✅ перевірено наскрізно |

---

**Останнє оновлення документа:** 2026-05-04.
