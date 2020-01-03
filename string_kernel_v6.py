# Project : mRNA-half-life-predictor
# Author  : Sanjan TP Gupta (sgupta78@wisc.edu)
#
# Aim: Analyze mRNA half-life dataset based on counts of sequence motifs 
#      in UTRs (minKeyLen to maxKeyLen, say 3-8 letters)
#
# I/P:
# e_file - tab delimited file containing seq_id, tvals & [stdev]
# f_file - tab delimited file containing seq_id, 5' utr, cds, 3' utr
#
# O/P:
# g_file  - tab delimited file containing seq_id, counts of diff seq motifs, 
#           tvals (optional)
# g1_file - log file to store pred vs true and performance metrics for CV and ho
# g2_file - log file to store feature importance scores based on final model
#
# Global variables:
# minKeyLen = 3 & maxKeyLen = 8 ==> searches for sequence motifs that are b/w 3 to
#                                   8 chars long
# disc_threshold = 0.1 ==> a sequence motif (or more broadly, feature) would be
#                           considered discriminative iff the difference b/w +ve and
#                           -ve classes is at least 10%
#
# Edits:
# v6 - classify a motif as useful (stable/unstable) based on its distribution in train-set
# v5 - grid search for param tuning (max_features & min_samples_split)
# v4 - CVfolds fixed and scatter plot
# v3 - data-set as X and y with wflag
# v2 - recursive motif generation, modular and well commented
# v1 - 3-letter motifs only

import random, scipy, time
import numpy as np

# ML related libraries
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error

# plotting libraries
import matplotlib
matplotlib.use('Agg')                   # for using over ssh -X
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from matplotlib import transforms as mtransforms

e  = open('PCC7002_tvals_stdev.dat','r')
f  = open('PCC7002_fasta.dat','r')
g1 = open('PCC7002_kernel_log.dat','w')
g2 = open('PCC7002_kernel_franking.dat','w')

random.seed(0)

# Global variables
minKeyLen = 3
maxKeyLen = 4
disc_threshold = 0.1              # 10% difference as a threshold for classifying
                                  # as a discriminative feature


def prune_data(X, fvector):
    # A custom function to perform column-wise pruning efficiently
    # (As python is primarily meant for string based operations
    #  the following idiomic expression is an efficient way to accomplish
    #  column-wise pruning when working with high-dimensional datasets)
    return X[:,[i for i in range(len(fvector)) if fvector[i]]]

def reg_report_log(y_true, y_pred, g, wflag = 1, mean_center = []):
    # Writes the performance metric of the ML models to file
    #
    # Inputs: g     - name of log file
    #         wflag - set wflag = 0 to suppress writing the list of true_val vs
    #                  pred_val to file (esp during long runs)
    
    # Compare true and pred vals
    if wflag:
        g.write('\n True_val \t Pred_val \n')
        
        if mean_center:
            y_true_rsc = mean_center.inverse_transform(y_true)
            y_pred_rsc = mean_center.inverse_transform(y_pred)
            for i in range(len(y_pred)):
                g.write(' ' + str(y_true_rsc[i]) + ' \t ' + str(y_pred_rsc[i]) + '\n')
        else:
            for i in range(len(y_pred)):
                g.write(' ' + str(y_true[i]) + ' \t ' + str(y_pred[i]) + '\n')
            
    g.write('\n Performance metrics:')
    R2_score = r2_score(y_true, y_pred)
    g.write('\n R2_score                = ' + str(R2_score))
                
    rmse = (mean_squared_error(y_true,y_pred)**0.5)
    nrmse = rmse/(max(y_true) - min(y_true))
    g.write('\n RMSE                    = '+str(rmse))
    g.write('\n NRMSE                   = '+str(nrmse))
    
    log_vec = [str(round(val,3)) for val in [R2_score,rmse,nrmse]]
    
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    if pred_std and true_std:    
        r, pval = scipy.stats.pearsonr(y_true, y_pred)
        g.write('\n Pearson corr. coeff : r = ' + str(r) + '; pval = ' + str(pval))
        rs, sp_pval = scipy.stats.spearmanr(y_true, y_pred)
        g.write('\n Spearman corr. coeff: r = ' + str(rs) + '; pval = ' + str(sp_pval) + '\n\n')
        log_vec.extend([str(val) for val in [round(r,3),'%.3E'%pval,round(rs,3),'%.3E'%sp_pval]])
        
    else:
        g.write('\n Std dev of y_true = ' + str(true_std) + ' and y_pred = ' + str(pred_std))
        g.write('\n Hence, failed to compute corr. coeffs\n\n')
        log_vec.extend(['Nan']*4)
        
    return log_vec      # R2_score, RMSE, NRMSE, PCC, pval, SCC, pval

def parse_tvals():
    # Read and parse the experimentally measured half-lives &
    # the corresponding stdev (optional)
    
    tvals_dic = {}      # key = seq_id, val = half-life
    tval_wts_dic = {}   # key = seq_id, val = 1 - stdev/max(stdev)
    elines = e.read().split('\n')[1:]
    max_stdev = 0
    for line in elines:
        vals = line.split('\t')
        seq_id = vals[0]
        tvals_dic[seq_id] = round(float(vals[1]),3)
        
        if len(vals) > 2:
            stdev = round(float(vals[2]),3)
            tval_wts_dic[seq_id] = stdev
            if stdev > max_stdev:
                max_stdev = stdev
    
    if max_stdev:
        # Transform stdev into wts
        for key in tval_wts_dic:
            tval_wts_dic[key] = 1 - (tval_wts_dic[key]/max_stdev)
    
    return tvals_dic, tval_wts_dic

def keylist_elongator(keylist):
    # Function to generate all possible minKeyLen (=3) to maxKeyLen (=8) letter
    # sequence motifs
    #
    # Inputs: 
    # keylist - list of all possible keys of 'length = minKeyLen' to seed the 
    #           keylist_elongator function
    
    count = len(keylist[0])    
    for key in keylist:
        # Verify that keylist contains keys of homogenous len
        if len(key) > count:
            print('Pls. supply keylist of constant sized keys')
            exit()
    
    curr_keylist = keylist      # contains a set of keys of 'same' len
    master_keylist = keylist    # to store keys of 'incremental' len
    print('\nKey_size \t len(keylist) \t expected_len')
    print(count,' \t ',len(curr_keylist),' \t ',4**count)
    master_count = 4**count
    while count < maxKeyLen:
        new_keylist = []    
        for c in ['A','C','G','T']:
            for key in curr_keylist:
                new_key = c + key
                new_keylist.append(new_key)
        master_keylist.extend(new_keylist)
        count += 1
        curr_keylist = new_keylist
        print(count,' \t ',len(curr_keylist),' \t ',4**count)
        master_count += (4**count)
        
    print('\nMaster_keylist \t ',len(master_keylist),' \t ',master_count)
    return master_keylist
    
def feature_vec_gen(seq, keylist = []):
    # Count the 'overlapping' frequency of different sequence motifs (keylist)
    # in the given sequence (seq)
    # For example: Counts of motif 'AAA' in the sequence 'AAAAXSAAWAAA' = 3
    #              (Notice that the first 4-bits will yield 2 hits)

    subs_freq = {}
    for subs_len in range(minKeyLen,maxKeyLen+1):
        for i in range(len(seq)-subs_len+1):
            motif = seq[i:i+subs_len]
            if motif in subs_freq:
                subs_freq[motif] += 1
            else:
                subs_freq[motif] = 1

    if not len(keylist):
        keylist = sorted(subs_freq)

    fvec = []
    for key in keylist:
        if key in subs_freq:
            fvec.append(str(subs_freq[key]))
        else:
            fvec.append('0')

    return fvec, keylist

def main():
    start_time = time.time()
    
    # Read seq_id, tvals and wts (if any)
    tvals_dic, tval_wts_dic = parse_tvals()

    n_trees = input('Enter the no. of trees for RF analysis (default = 100) : ')
    try:
        n_trees = int(n_trees)
    except ValueError:
        n_trees = 100
        
    nfolds = input('Enter the number of folds for cross validation analysis (default = 10) : ')
    try:
        nfolds = int(nfolds)
    except ValueError:
        nfolds = 10                         # default value
    
    tsize = input('Enter the fraction of dataset to be held-out (default = 0.2 ==> 20% of data) : ')
    try:
        tsize = float(tsize)
    except ValueError:
        tsize = 0.2                                             # default value

    g_wflag = input('Do you wish to store the feature vector generated for later use? Pls. enter 0 or 1 (default = 0) : ')
    try:
        g_wflag = int(g_wflag)                      # writes dataset to g-file if set to 1
    except ValueError:
        g_wflag = 0

        
    ## Test set for design phase analysis
    test_fname = ''
    X_new = []
    if test_fname:
        # Add module to read test_fname and store into Xnew ???
        print('Skipping test_fname for now')
    
    ## Generate list of all 3 letter keys and then recursively generate upto
    ## 8 letter keys
    keylist = []
    for i in ['A','C','G','T']:
        for j in ['A','C','G','T']:
            for k in ['A','C','G','T']:
                s = i+j+k
                keylist.append(s)
    
    # Use keylist_elongator to recursively elongate keylist upto maxKeyLen
    final_keylist = keylist_elongator(keylist)
    print('Key length varies from ',len(min(final_keylist)),' to ',
          len(max(final_keylist)))
    
    ## Generate feature vector list and data matrices X & y
    original_flist = []
    X = []
    y = []    
    
    s = f.readline()                # scan header
    
    for key in final_keylist:
        # adding 5' header
        motif_name = "5'-" + key
        original_flist.append(motif_name)
        
    for key in final_keylist:
        # adding 3' header
        motif_name = "3'-" + key
        original_flist.append(motif_name)
    
    if g_wflag:
        # write data-set to file
        g = open('PCC7002_UTR_kernels.dat','w', 1)
        s_op = 'seq_id\t' + '\t'.join(original_flist) + '\thalf-life\n'
        g.write(s_op)
    
    print('\nGenerating feature vector and compiling X & y ...')
    s = f.readline()
    while s:
        l1 = s.strip().split('\t')
        seq_id = l1[0]
        
        #data_row = []
        if seq_id in tvals_dic:
            fvec_5utr, dummy = feature_vec_gen(l1[1],final_keylist)
            fvec_3utr, dummy = feature_vec_gen(l1[3],final_keylist)
            
            X.append([int(val) for val in fvec_5utr + fvec_3utr])
            y.append(round(float(tvals_dic[seq_id]),3))
            
            if g_wflag:
                # write data-set to file
                s_op = seq_id + '\t' + '\t'.join(fvec_5utr) + '\t'
                s_op += '\t'.join(fvec_3utr) 
                s_op = s_op + '\t' + str(tvals_dic[seq_id]) + '\n'
                
                g.write(s_op)
    
        s = f.readline()

    g1.write('Input params: ')
    g1.write('\n minKeyLen = ' + str(minKeyLen) + ' maxKeyLen = ' + str(maxKeyLen))
    g1.write('\n nfolds    = ' + str(nfolds)    + ' tsize     = ' + str(tsize))
    g1.write('\n nTrees    = ' + str(n_trees))
    g1.write('\n\nOriginally,    Dim(X) = ' + str(len(X)) + ' x ' + str(len(X[0])))

    # remove features having no variation
    fvector = [1]*len(X[0])
    print('\n Removing features with negligible variation (i.e. max - min < 0.01) ...\n')
    for c in range(len(X[0])):
        l1 = [X[i][c] for i in range(len(X))]
        
        if min(l1) == max(l1):                          # i.e. no variance
        #if (max(l1) - min(l1)) < 0.01:                 # i.e. variance ~ 0.0001
            fvector[c] = 0

    np_X = np.array(X)
    np_X_new = np.array(X_new)
    
    if len(fvector) != sum(fvector):
        X_pruned = prune_data(np_X,fvector)    
        g1.write('\nAfter pruning, Dim(X) = ' + str(len(X_pruned)) + ' x ' + str(len(X_pruned[0])))

        X_new_pr = []
        if test_fname:
            X_new_pr = prune_data(np_X_new,fvector)
            g1.write('\nSize of X_new    :' + str(len(X_new_pr)) + ' x ' + str(len(X_new_pr[0])))
        np_X, np_X_new = X_pruned, X_new_pr
        flist = [original_flist[i] for i in range(len(X[0])) if fvector[i]]
        
    else:
        # if fvector is full of ones
        flist = original_flist

    tot_elements = len(np_X)*len(np_X[0])
    nnz_elements = np.count_nonzero(np_X)
    g1.write('\nNo. of elements          = ' + str(tot_elements) )
    g1.write('\nNo. of non-zero elements = ' + str(nnz_elements))
    per_sparsity = round((1 - (nnz_elements/tot_elements))*100,3)
    g1.write('\n% sparsity               = ' + str(per_sparsity) + '\n')
    
    # converting the data to numpy array
    np_y = np.array(y)
 
    #np_X, np_y = np.array(X), np.array(y)
    print('\nNo. of features = ',len(flist))
    print('Dim(X)       = ',np_X.shape)
    print('Dim(y)       = ',np_y.shape)
    
    # Part of the data is held out for testing the best model built
    X_train, X_ho, y_train, y_ho = train_test_split(np_X, np_y, test_size=tsize
                                                    ,random_state=0)

    print('Dim(X_train) = ',X_train.shape)    
    print('Dim(X_ho)    = ',X_ho.shape)
    
    print('\nBuilding RF based on ',n_trees,' decision trees ...')
    print('\nCaution using n_jobs = -1 (==> all cores being used) ...')
    
    # Possible ways to speed up RF
    # Decrease n_estimators, set max_features to sqrt or log2, max_depth to 5 or 10
    #
    # Total no. of features ~ 160,000 after removing non-varying features
    # Ideal values: n_estimators >= 10, max_features > 10, max_depth ~ 10 (as binary split trees 
    #               ==> 2^10 - 1 = 1023 features as decision nodes)
    
    # ToDo: Include max_depth as a parameter within the grid search
    #       & possibly change cross_val_score to return SCC for heat map ???
    
    #param_grid = {'max_features': [0.1,0.2,0.3,0.5,0.7,0.9,1], 'min_samples_split': [2,5,10,50,100]}       
    param_grid = {'max_features': [0.1,0.2,0.3,0.5,0.7,0.9], 'min_samples_split': [2,5,10,50,100,200,500,1000]}   
    #param_grid = {'max_features': [0.1], 'min_samples_split': [50]}   

    '''
    mdl = RandomForestRegressor(n_estimators = n_trees, random_state = 0, n_jobs = -1) 
                                                                    # max_features = 'sqrt',  max_depth = None
    mdl_type = 'RandomForestRegressor'
    '''

    mdl = ExtraTreesRegressor(n_estimators = n_trees, bootstrap = True,
                              random_state = 0, n_jobs = -1)    # max_depth = 'None', max_features = 'sqrt'
    mdl_type = 'ExtraTreesRegressor'                              
    ''
    
    # Notes on GridSearchCV: 
    # - error_score = 0 to set score = 0  for failed runs instead of premature termination of grid search
    # - by default, pre_dispatch = '2*n_jobs' and each dispatch makes a new copy of dataset (memory 
    #   intensive) 
    gridCV_mdl = GridSearchCV(mdl, param_grid, cv = nfolds, n_jobs = 5, pre_dispatch = 'n_jobs', verbose = 1, error_score = 0) 
    dummy_y = gridCV_mdl.fit(X_train,y_train)
    best_mdl = gridCV_mdl.best_estimator_                           # By default, gridCV refits a final model
                                                                    # using entire dataset & best set of
                                                                    # params found during grid search
    print(gridCV_mdl.best_score_)
    print(gridCV_mdl.best_params_)

    print('Type(best_mdl) = ',type(best_mdl))
    print('Type(dummy_y)  = ',type(dummy_y))
    y_CV_pred = best_mdl.fit(X_train, y_train).predict(X_train) 
    print('Size(y_CV_pred) = ',len(y_CV_pred))
    tr_log_vec = reg_report_log(y_train, y_CV_pred, g1)
    
    g1.write('\n Performance on held-out dataset:')
    y_ho_pred = best_mdl.fit(X_train,y_train).predict(X_ho)
    ho_log_vec = reg_report_log(y_ho, y_ho_pred, g1)
    
    end_time = time.time()
    time_taken = round((end_time-start_time)/60,3)
    time_taken = str(time_taken) + ' min'        
    
    g1.write('\n Model type = ' + mdl_type)
    g1.write('\n Param grid = ' + str(param_grid))
    g1.write('\n Chosen params = ' + str(gridCV_mdl.best_params_))
    g1.write('\n\nFor logging:')
    g1.write('\n time \t tr_R2_score \t tr_RMSE \t tr_NRMSE \t tr_PCC \t tr_pval \t tr_SCC \t tr_pval ')
    g1.write('\t ho_R2_score \t ho_RMSE \t ho_NRMSE \t ho_PCC \t ho_pval \t ho_SCC \t ho_pval ')
    g1.write('\n ' + time_taken + ' \t ' + ' \t '.join(tr_log_vec) + ' \t ' + ' \t '.join(ho_log_vec))
    
    ## Figure 1. Scatter plot (pred vs true) with main diagonal (ref line)  ##
    fig, ax = plt.subplots()

    ax.scatter(y_train, y_CV_pred, c = 'b', label = 'trainCV')
    ax.scatter(y_ho, y_ho_pred, c = 'r', label = 'held-out')

    # Add 'y=x' reference line
    line = mlines.Line2D([0, 1], [0, 1], color = 'black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    txt_size = 12
    plt.xlabel('Measured half-life (in min)', fontsize = txt_size)
    plt.ylabel('Predicted half-life (in min)', fontsize = txt_size)
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.legend(loc = 'best')
    #plt.show()
    fig.savefig('fig1_scatter_v1.svg', bbox_inches = 'tight')
    
    ## Figure 2. Feature importance plot ##   
    fig = plt.figure()
    feature_importance = best_mdl.feature_importances_
    
    # Normalizing importance_scores relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    print('len(feature_importance) = ',len(feature_importance))
    print('len(flist)              = ',len(flist))
    sorted_idx = np.argsort(feature_importance)

    top_sorted_idx = sorted_idx[-20:]                # Analyze top 'X' features only
    bottom_sorted_idx = sorted_idx[:20]
    pos = np.arange(top_sorted_idx.shape[0]) + .5
    print('20 most  significant feature scores = ',feature_importance[top_sorted_idx])
    print('20 least significant feature scores = ',feature_importance[bottom_sorted_idx])
    
    plt.barh(pos, feature_importance[top_sorted_idx], align='center')

    # Unlike feature_importance, flist is not subscriptable as it is not a 'numpy' array
    flist_subset = []
    for idx in top_sorted_idx:
        flist_subset.append(flist[idx])
    
    plt.yticks(pos, flist_subset)
    plt.xlabel('Relative Importance')
    #plt.show()    
    fig.savefig('fig2_franking_v1.svg', bbox_inches = 'tight')
    
    # Write the entire set of feature importance scores to file
    g2.write(mdl_type + '\t' + str(gridCV_mdl.best_params_))

    '''
    ToDo ???:
    - booleanize the counts during pos_fval and neg_fval computation (for speed) 
    '''

    # keep track of number of times the top feature has a value of 1 among
    # positive and negative examples
    stable_threshold   = round(np.mean(y_train) + (np.std(y_train)),1)
    unstable_threshold = round(np.mean(y_train) - (np.std(y_train)),1)
    g2.write('\n  Stable threshold = ' + str(stable_threshold) + ' min')
    g2.write('\nUnstable threshold = ' + str(unstable_threshold) + ' min')
    
    pos_fval = [0]*len(flist)       # each bit tells the no. of times that feature was present among 'good/stable' designs
    neg_fval = [0]*len(flist)       # each bit tells the no. of times that feature was present among 'bad/unstable' designs
    pos_count = 0                   # number of positive/stable transcripts
    neg_count = 0                   # number of negative/unstable transcripts
    
    for i in range(len(X_train)):
        if y_train[i] > stable_threshold:
            # entry corresponds to positive example
            pos_count += 1
            for j in range(len(pos_fval)):
                if X_train[i][j]:
                    pos_fval[j] += 1            # as X_train consists of counts of sequence motif 
            
        elif y_train[i] < unstable_threshold:
            neg_count += 1
            for j in range(len(neg_fval)):
                if X_train[i][j]:
                    neg_fval[j] += 1

    # Normalizing based on the number of +ve and -ve examples in the true data set
    print('  Stable threshold = ' + str(stable_threshold) + ' min')
    print('Unstable threshold = ' + str(unstable_threshold) + ' min')
    print('pos_count = ',pos_count,' neg_count = ',neg_count)
    
    if pos_count and neg_count:
        pos_fval = np.array(pos_fval)/pos_count
        neg_fval = np.array(neg_fval)/neg_count
    else:
        print('Stable and unstable thresholds are too stringent for the supplied dataset !!')
        g2.write('\nCaution: Stable and unstable thresholds are too stringent for the supplied dataset !!')

    g2.write('\nFeature_name \t Feature Importance Score \t Stable assoc. score \t Unstable assoc. score \t Verdict')    
    for idx in sorted_idx:
        hflag = 'N/A'                       # is it a heuristic for good or bad design?
        
        if pos_fval[idx] - neg_fval[idx] > disc_threshold:
            hflag = 'stable motif'          # feature popularly found in good designs
        elif neg_fval[idx] - pos_fval[idx] > disc_threshold:
            hflag = 'unstable motif'        # feature popularly found in bad designs
        
        s_op = '\n' + flist[idx] + ' \t ' + str(feature_importance[idx])
        s_op += ' \t ' + str(pos_fval[idx]) + ' \t ' + str(neg_fval[idx]) + ' \t ' + hflag
        g2.write(s_op)
            
    e.close()
    f.close()
    g1.close() 
    g2.close()
    
    if g_wflag:
        g.close()

if __name__ == "__main__":
    main()
