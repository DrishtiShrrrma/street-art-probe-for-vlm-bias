# Continent L1 – Key Insights (InternVL3, Prompts 1–4)

The notebook (i.e. `internvl3_continentL1_json_parsing_prompts1-4.ipynb`) evaluates InternVL3 on continent-level classification across four prompt styles.
Below are the **main findings** and the **confusion matrix** illustrating common error patterns.

---

## 📌 Summary of Results

| Prompt       | Accuracy   | Macro-F1   | Notes                                                                        |
| ------------ | ---------- | ---------- | ---------------------------------------------------------------------------- |
| **Prompt 1** | **0.6012** | **0.5912** | Best overall performance; simple numbered format seems effective.            |
| Prompt 2     | 0.5528     | 0.5609     | More structured, but lower recall in several classes.                        |
| Prompt 3     | 0.4869     | 0.5003     | Worst performing; debate format adds noise instead of improving consistency. |
| **Prompt 4** | 0.5944     | 0.5716     | Second-best; probabilistic outputs help but still below prompt 1.            |

### **Key Insight**

**Prompt 1 and Prompt 4 outperform the others**, suggesting that

* simple formats or
* explicit probability distributions

are more effective than verbose or debate-style prompts.

---

## 🌍 Class-Level Observations

### **Africa**

* Often confused with *Asia* and *South America* in Prompt 2
* High precision in P2 (0.88) but lower recall (0.48) → conservative predictions

### **Asia**

* Strong diagonal (good recall), but confused with:

  * Europe
  * North America
* Prompt 4 boosts performance significantly (F1: ~0.75)

### **Europe**

* Frequently misclassified as **North America**
* Prompt 1 achieves highest recall (0.81)

### **North America**

* Consistently the **most confused** class
* Often mispredicted as:

  * Europe
  * South America
  * Asia
* Recall stays low across prompts (0.36–0.60)

### **South America**

* Moderate performance but still confused with:

  * North America
  * Africa
  * Asia

---

## 🔥 Prompt-Level Patterns

### **Prompt 1**

* Clean structured text → fewer extraction errors
* High recall for Europe and South America

### **Prompt 2**

* Structured output but uneven performance
* Africa & North America show big recall drops

### **Prompt 3**

* Multi-role debate format backfires
* Large inconsistencies → lowest accuracy

### **Prompt 4**

* Probabilities help recover some misclassifications
* Particularly good for Asia and Europe
* But second-choice continent rarely rescues mistakes

---

## 📊 Confusion Matrix (Prompt 2)

This visual highlights which continents the model confuses most often.

<img width="550" height="442" alt="image" src="https://github.com/user-attachments/assets/92ddd8f1-0c5f-4b13-9486-b9031cbad3d9" />


Common confusion patterns:

* Africa ↔ Asia / South America
* Europe ↔ North America
* South America ↔ North America

---

## 📝 High-Level Takeaways

* **Prompt engineering matters** — simple prompts outperform verbose formats.
* **North America is the hardest class** across all prompts.
* **Asia and Europe** are generally the strongest classes.
* **Prompt 3 (debate)** introduces noise rather than resolving ambiguity.
* **Probabilistic prompting (Prompt 4)** helps but doesn’t consistently exceed Prompt 1.


Just tell me!
