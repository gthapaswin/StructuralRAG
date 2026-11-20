# Grounded answer generation

Generator `qwen3:8b`, claim checker `llama3.1`, faithfulness judged by `cross-encoder/nli-deberta-v3-small` (NLI, independent of both). 150 LLM-written questions, top-3 sections, 900-word context budget; 100 unanswerable questions (asked against a different report).

| setup | variant | abstained | faithfulness | citation_hit | citation_hit_branch | citation_precision | gold_in_context | mean_sentences | removed_per_answer |
|---|---|---|---|---|---|---|---|---|---|
| A: BM25 + plain | no_check | 0.100 | 0.589 | 0.815 | 0.859 | 0.779 | 0.807 | 2.696 | 0.000 |
| A: BM25 + plain | checked | 0.107 | 0.603 | 0.821 | 0.866 | 0.789 | 0.807 | 2.604 | 0.100 |
| B: structure + plain | no_check | 0.067 | 0.575 | 0.893 | 0.900 | 0.829 | 0.907 | 2.721 | 0.000 |
| B: structure + plain | checked | 0.067 | 0.582 | 0.893 | 0.900 | 0.831 | 0.907 | 2.657 | 0.060 |
| C: structure + hierarchical | no_check | 0.067 | 0.598 | 0.893 | 0.900 | 0.842 | 0.907 | 2.629 | 0.000 |
| C: structure + hierarchical | checked | 0.067 | 0.611 | 0.893 | 0.900 | 0.845 | 0.907 | 2.579 | 0.047 |
| C on unanswerable questions | no_check | 0.960 | 0.875 | nan | nan | nan | nan | 1.500 | 0.000 |
| C on unanswerable questions | checked | 0.960 | 0.875 | nan | nan | nan | nan | 1.500 | 0.000 |


## Paired comparisons

| comparison | metric | n_pairs | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|---|
| structure-aware vs BM25 retrieval (plain ctx) | faithfulness | 133 | -0.011 | 0.674 | 0.698 |
| structure-aware vs BM25 retrieval (plain ctx) | citation_hit | 133 | 0.075 | 0.040 | 0.132 |
| structure-aware vs BM25 retrieval (plain ctx) | abstained | 150 | -0.033 | 0.175 | 0.442 |
| hierarchical vs plain context | faithfulness | 135 | 0.025 | 0.399 | 0.358 |
| hierarchical vs plain context | citation_hit | 135 | -0.007 | 1.000 | 0.867 |
| hierarchical vs plain context | abstained | 150 | 0.000 | 1.000 | 1.000 |
| claim check vs none (config C) | faithfulness | 140 | 0.014 | 0.062 | 0.416 |
| claim check vs none (config C) | citation_hit | 140 | 0.000 | 1.000 | 1.000 |
| claim check vs none (config C) | abstained | 150 | 0.000 | 1.000 | 1.000 |
| full pipeline vs BM25 baseline | faithfulness | 131 | 0.030 | 0.323 | 0.199 |
| full pipeline vs BM25 baseline | citation_hit | 131 | 0.076 | 0.042 | 0.130 |
| full pipeline vs BM25 baseline | abstained | 150 | -0.033 | 0.273 | 0.453 |


## Example answers (config C, checked)

**Q:** How does the VA address challenges in providing timely in-person care to veterans?  
**Gold section:** 1.1  
**A:** Telehealth allows veterans to access health care services through information and communication technology without needing to visit a service provider in person. [1.1.1]
  
*Removed by claim check:* The VA addresses challenges in providing timely in-person care to veterans by expanding the use of telehealth services. [1.6.1]


**Q:** How do states typically handle best interest determinations and data sharing for foster care youth in school enrollment?  
**Gold section:** 8.1  
**A:** States typically handle best interest determinations by considering all factors relating to the child’s best interest, including the appropriateness of the current educational setting and proximity to the school in which the child is enrolled at the time of placement. [1.1] Data sharing for foster care youth in school enrollment is facilitated through sample memoranda of understanding or agreements between school districts and local child welfare agencies. [8.1]


**Q:** What historical context led to the U.S. establishing nuclear security programs for Russia and former Soviet states?  
**Gold section:** 1  
**A:** The U.S. established nuclear security programs for Russia and former Soviet states following the collapse of the Soviet Union in 1991, due to significant concerns about Russia's ability to maintain adequate security over its nuclear weapons and weapons-usable nuclear materials. [1] In 1995, the Department of Energy (DOE) established the Material Protection, Control, and Accounting (MPC&A) program to equip Russia and other countries with modern nuclear material security systems and promote effective nuclear material security practices. [1]


**Q:** Who were the key individuals involved in the development of this report beyond the primary contact?  
**Gold section:** 10.2  
**A:** The key individuals involved in the development of this report beyond the primary contact include Scott Spicer, Amy Moran Lowe, Grace Cho, and Michael Walton. [3.2]
  
*Removed by claim check:* Other contributors to the report were Joel Aldape, Aditi Archer, Susan Aschoff, James E. Bennett, Deirdre Gleeson Brown, Alicia Cackley, Sarah Cornetto, Elizabeth Curda, Chris Currie, Kelly DeMots, Erin Guinn-Villareal, Camille Henley, Denton Herring, Sara Schibanoff Kelly, James Lawson, Matthew T. Lowney, Sheila R. McCoy, Jean McSween, Amanda R. Parker, Sara Pelton, Brenda Rabinowitz, Michelle Sager, Brian Schwartz, Almeta Spencer, Manuel Valverde, Jr., and Su Jin Yon. [3.2]


**Q:** How do federal agencies and sector stakeholders collaborate to enhance the resilience of water and wastewater infrastructure against natural disasters and climate change?  
**Gold section:** 1.2  
**A:** For example, utilities use HUD Community Development Block Grant-Disaster Recovery funding and FEMA’s Public Assistance grant program to increase resilience to floods. [4.1] Similarly, utilities like Houston Water and San Diego Public Utilities Department use FEMA and EPA programs to enhance resilience against floods and droughts. [4.1] Representatives of several utilities reported using selected federal financial assistance programs to help fund projects that enhance climate resilience. [4.1]
  
*Removed by claim check:* Federal agencies and sector stakeholders collaborate by using financial assistance programs to fund projects that enhance climate resilience. [4.1]


**Q:** How does NASA handle cost and schedule changes when a program exceeds its baseline estimates?  
**Gold section:** 1.1  
**A:** NASA handles cost and schedule changes by initiating a replan process when development cost growth reaches 15 percent or more of the estimate in the baseline report or a major milestone is delayed by 6 months or more from the baseline date. [1.1]
  
*Removed by claim check:* If development costs are expected to increase by more than 30 percent, NASA must submit a report to the Committee on Science, Space, and Technology of the House of Representatives and rebase the program. [1.1]

