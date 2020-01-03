# mRNA-half-life-predictor
Python scripts to build ML models to predict mRNA half-lives in using sequence [and structure] based features

# Author  : Sanjan TP Gupta (sgupta78@wisc.edu)

Aim: Analyze mRNA half-life dataset based on counts of sequence motifs 
      in UTRs (minKeyLen to maxKeyLen, say 3-8 letters)

# I/P:
e_file - tab delimited file containing seq_id, tvals & [stdev]
f_file - tab delimited file containing seq_id, 5' utr, cds, 3' utr

# O/P:
 g_file  - tab delimited file containing seq_id, counts of diff seq motifs, 
           tvals (optional)
 g1_file - log file to store pred vs true and performance metrics for CV and ho
 g2_file - log file to store feature importance scores based on final model

# Global variables:
 minKeyLen = 3 & maxKeyLen = 8 ==> searches for sequence motifs that are b/w 3 to
                                   8 chars long
 disc_threshold = 0.1 ==> a sequence motif (or more broadly, feature) would be
                           considered discriminative iff the difference b/w +ve and
                           -ve classes is at least 10%

 Edits:
 v6 - classify a motif as useful (stable/unstable) based on its distribution in train-set
 v5 - grid search for param tuning (max_features & min_samples_split)
 v4 - CVfolds fixed and scatter plot
 v3 - data-set as X and y with wflag
 v2 - recursive motif generation, modular and well commented
 v1 - 3-letter motifs only
