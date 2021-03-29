# Chat Topic Analysis Comparison
This collection of programs were developed for the Library at the University of Northern Iowa to study the performance of unsupervised and semi-supervised topic modeling techniques on its library's chat reference data. It integrates 4 unsupervised and 2 semi-supervised topic models and calculates TC-PMI, TC-LCP, and TC-NZ topic coherence scores as a reference for users to evaluate the performance of the models.

Unsupervised Models available:
* TF-IDF & LSA  (using gensim)
* TF-IDF & pLSA (using gensim)
* LDA (using sklearn)
* PyMallet LDA (https://github.com/mimno/PyMallet)

Semi-supervised Models available:
* GuidedLDA (using https://github.com/vi3k6i5/GuidedLDA)
* CorEx (Correlation Explanation (using https://github.com/gregversteeg/corex_topic)

## Getting Started

### Files

There are 3 main components of the program:
* Phase 1 P1_preprocessing_data.py
* Phase 2 P2_unsupervised_topic_modeling.py
* Phase 3 P3_semi_supervised_topic_modeling.py

