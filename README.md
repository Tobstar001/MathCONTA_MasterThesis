# MathCONTA_MasterThesis

**RQ1**: For every contamination detection method there exists one notebook which will lead you through the experimental Setup.

├── code
│   ├── CDM_eval
│   │   ├── CDMs_functions_old.py
│   │   ├── CDMs_functions_v1.py
│   │   ├── CD_pipeline_CV_CDD.ipynb (RQ1)
│   │   ├── CD_pipeline_CV_ContaTraces.ipynb (RQ1)
│   │   ├── CD_pipeline_CV_minK.ipynb (RQ1)
│   │   ├── CD_pipeline_CV_ngram_acc.ipynb (RQ1)
│   │   ├── CD_pipeline_CV_ngram_cdd.ipynb (RQ1)
│   │   ├── CD_pipeline_CV_ngram_loglike.ipynb (RQ1)
│   │   ├── cdm_accuracy_log_analysis.ipynb (RQ1)
│   │   └── ensemble_and_boxplot_eval.ipynb (RQ1)
│   └── LLM_eval
│       ├── LLM_Eval_Visuals.ipynb (RQ2)
│       └── LLM_Evaluations.ipynb (RQ2)

## Abstract of the Thesis:
Large language models (LLMs) have demonstrated impressive capabilities in mathematical reasoning tasks. However, concerns persist around data contamination, where benchmark problems used for evaluation have appeared in the model's pretraining data. Such contamination can artificially inflate performance metrics, particularly in domains where genuine reasoning must be distinguished from memorization. This thesis introduces \emph{MathCONTA}, a novel mathematical dataset for contamination detection. It spans multiple domains—including algebra, number theory, combinatorics, and integration—and covers various difficulty levels, from simple word problems to advanced math contest problems.

\emph{MathCONTA} consists of two balanced subsets: one containing problems known to have been seen during pretraining (contaminated) and another with entirely novel problems (uncontaminated). In contrast to previous studies that simulate contamination via finetuning, \emph{MathCONTA} reflects how contamination arises naturally during pretraining, offering a more realistic basis for evaluation. Using this dataset, we evaluate several representative detection methods, including min$K$\%, $n$-gram accuracy, and Contamination Detection via output Distribution (CDD), spanning confidence-based and memorization-based approaches. Our findings show, perhaps surprisingly, that none of the tested techniques reliably distinguish between contaminated and uncontaminated items. Moreover, combining these methods does not significantly improve detection performance. We hypothesize that this is because \emph{MathCONTA} requires detecting subtle, incidental mathematical contamination arising naturally during large-scale pretraining, rather than the more obvious contamination introduced through fine-tuning.


Finally, we analyze the downstream impact of contamination on model accuracy and find that while it can lead to modest gains at specific difficulty levels, it is unlikely to be the primary factor behind recent advances in LLM-based mathematical reasoning. To support transparency and future research, \emph{MathCONTA} and all accompanying code for experiments will be made openly available.

<img width="1051" alt="Bildschirmfoto 2025-05-05 um 21 51 46" src="https://github.com/user-attachments/assets/5f0a0a57-a825-4b7c-ba44-d679717a913a" />



