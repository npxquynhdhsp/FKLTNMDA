# %%
import pickle
import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_auc_score, accuracy_score, precision_recall_curve, precision_score
from params import args
# from GENERAL_UTILS.Q_eval import eval #QD
from GENERAL_UTILS.Genrl_utils import save_file

#####################################
methods = 'X'  # 'MRX' # M(MLP), X(XGBoost), R(RF)
if args.type_eval == 'kfold':
    loai_tam = '_fold'
    b_ix, e_ix = args.bgf, args.nfold + args.bgf
else:
    if args.type_eval == 'dis_k':
        loai_tam = '_dis'
    else:
        loai_tam = '_gl'
    b_ix, e_ix = 0, len(args.set_dis)


###########################################################################

# %%
def read_data(args, ix, triplet_i, loop_i):
    print("--- READ TRAIN SET ---")
    if args.dis_k_koxoa == '':
        tr_X, tr_y = pickle.load(open(
            args.fi_out + "For combination/L" + str(loop_i) + "_Data_train_from_tripletnet" + str(
                triplet_i) + args.dis_k_koxoa + loai_tam + str(ix) + ".pkl", "rb"))
    else:
        tr_X, tr_y = pickle.load(
            open(args.fi_out + "For combination/L" + str(loop_i) + "_Data_train_from_tripletnet" + str(
                triplet_i) + args.dis_k_koxoa + loai_tam + ".pkl", "rb"))
    tr_y = np.array(tr_y)
    print('tr_X.shape', np.array(tr_X).shape)
    print(tr_y.shape)

    print("--- READ TEST SET ---")
    te_X, te_y = pickle.load(open(args.fi_out + "Data_test/L" + str(loop_i) + "_From_tripletnet" + str(
        triplet_i) + args.dis_k_koxoa + loai_tam + str(ix) + ".pkl", "rb"))
    te_y = np.array(te_y)
    print('te_X.shape', np.array(te_X).shape)
    print(te_y.shape)
    return tr_X, tr_y, te_X, te_y


def get_yprob_ypred(tr_X, tr_y, te_X, te_y, args, ix, method, triplet_i, loop_i):
    if method == 'M':  # MLR
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(20, 20, 20))
        model.fit(tr_X, tr_y)
        prob_y = model.predict(te_X)
        pred_y = np.array(prob_y > 0)
    else:
        if method == 'X':  # xgboost
            from xgboost import XGBClassifier
            # code estimator from Q16_6
            lrr = args.lrr  # Q
            ne = args.xgne  # ori 500
            model = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=lrr, n_estimators=ne, random_state=48)
            model.fit(tr_X, tr_y)
            prob_y = model.predict_proba(te_X)[:, 1]
            pred_y = model.predict(te_X)
        else:  # RF
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=0)
            model.fit(tr_X, tr_y)
            prob_y = model.predict_proba(te_X)[:, 1]
            pred_y = model.predict(te_X)

    np.savetxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + method + '_triplet' + str(
        triplet_i) + args.dis_k_koxoa + \
               loai_tam + str(ix) + '.csv', prob_y)
    # np.savetxt(args.fi_out + 'Results/Combination/Loop' + str(loop_i) + '_pred_' + method + '_triplet' + str(triplet_i) + args.dis_k_koxoa +\
    #            loai_tam + str(ix) + '.csv', pred_y, fmt='%d')
    np.savetxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_true_' + method + '_triplet' + str(
        triplet_i) + args.dis_k_koxoa + \
               loai_tam + str(ix) + '.csv', te_y, fmt="%d")
    return prob_y, pred_y


def get_test_score(yprob, ypred, ytrue):
    acc = accuracy_score(ytrue, ypred)
    pre = precision_score(ytrue, ypred)
    auc_ = roc_auc_score(ytrue, yprob)  # ko đặt trùng tên auc, mà phải auc_
    precision, recall, _ = precision_recall_curve(ytrue, yprob)
    aupr_ = auc(recall, precision)
    return acc, pre, auc_, aupr_


# %%
def main(args, loop_i):  # prob of train
    for itam in range(b_ix, e_ix):
        if args.type_eval == 'kfold':
            ix = itam
            print('########################Fold ', ix, '######################')
        else:
            ix = args.set_dis[itam]
            print('########################Dis: ', ix, '######################')
        print('Triplet 1')
        tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(args, ix, 1, loop_i)
        prob_y_trp1_m, pred_y_trp1_m = get_yprob_ypred(tr_X_1, tr_y_1, te_X_1, te_y_1, \
                                                       args, ix, method, 1, loop_i)  # test

        print('Triplet 2')
        tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(args, ix, 2, loop_i)
        prob_y_trp2_m, pred_y_trp2_m = get_yprob_ypred(tr_X_2, tr_y_2, te_X_2, te_y_2, \
                                                       args, ix, method, 2, loop_i)
    return


# %%
def main2(args, a, opt_para, loop_i, method):
    ACC, PRE, AUC, AUPR = [], [], [], []  # each loop
    name_trb_true = 'L' + str(loop_i) + '_true_trb' + method + args.dis_k_koxoa
    name_trb_prob = 'L' + str(loop_i) + '_prob_trb' + method + args.dis_k_koxoa
    for itam in range(b_ix, e_ix):
        if args.type_eval == 'kfold':
            ix = itam
            print('########################Fold ', ix, '######################')
        else:
            ix = args.set_dis[itam]
            print('########################Dis: ', ix, '######################')
        print('Triplet 1')
        tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(args, ix, 1, loop_i)
        prob_y_trp1_m = np.genfromtxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                      method + '_triplet1' + args.dis_k_koxoa + loai_tam + str(ix) + '.csv')

        print('Triplet 2')
        tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(args, ix, 2, loop_i)
        prob_y_trp2_m = np.genfromtxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                      method + '_triplet2' + args.dis_k_koxoa + loai_tam + str(ix) + '.csv')

        prob_y_m = (prob_y_trp1_m * a + prob_y_trp2_m * (1 - a))

        if method == "M":  # MLR
            pred_y_m = np.array(prob_y_m > 0)
        else:
            pred_y_m = np.array(prob_y_m >= 0.5)

        if opt_para == 1:
            save_file(args.fi_out + 'Results/Combination/', name_trb_true, loai_tam, ix, 'onedim_int', te_y_1)  # q
            save_file(args.fi_out + 'Results/Combination/', name_trb_prob, loai_tam, ix, 'onedim', prob_y_m)  # q

        acc_m, pre_m, auc_m, aupr_m = get_test_score(prob_y_m, pred_y_m, te_y_1)

        ACC.append(acc_m)
        PRE.append(pre_m)
        AUC.append(auc_m)
        AUPR.append(aupr_m)

    if opt_para == 1:
        # print('########################FINAL RESULT OF METHOD ', method, '######################')
        # print("AUC        ", np.mean(np.array(AUC)))
        # print("AUPR       ", np.mean(np.array(AUPR)))

        # -------------FINAL EVALUATION------------------
        path_eval = args.fi_out + 'Results/Combination/'
        path_out_fig = path_eval + 'FIG/'
        path_out_kq = path_eval + 'KQ/'
        name_plot = 'AUC_AUPR-' + args.type_eval + args.dis_k_koxoa
        fig_file = path_out_fig + name_plot
        # eval(args,path_eval, b_ix, e_ix, name_trb_prob, name_trb_true, 0, 0.5, path_out_fig, path_out_kq, fig_file,\
        #            loai_tam = loai_tam)

    # print('Saved figure in', path_out_fig)

    return np.mean(np.array(AUC)), np.mean(np.array(AUPR))


# %%
def join2_prob(_folder, _name1, _name2, b_ix, e_ix):  # @Q join fold
    for itam in range(b_ix, e_ix):
        PROB_all_fold = []
        TRUE_all_fold = []
        for loop_i in range(1, num_loop + 1):
            prob_i = np.genfromtxt(_folder + 'L' + str(loop_i) + _name1 + str(itam) + '.csv')
            true_i = np.genfromtxt(_folder + 'L' + str(loop_i) + _name2 + str(itam) + '.csv').astype(int)
            PROB_all_fold.extend(prob_i)
            TRUE_all_fold.extend(true_i)
        np.savetxt(_folder + 'PROB_all_fold' + str(itam) + '.csv', np.array(PROB_all_fold))
        np.savetxt(_folder + 'TRUE_all_fold' + str(itam) + '.csv', np.array(TRUE_all_fold), fmt='%d')
    return


# %%
def danhgia_rieng():
    path_eval = args.fi_out + 'Results/Combination/'
    method = 'X'
    str_fi1 = 'PROB_all'
    str_fi2 = 'TRUE_all'
    cla = 0  # 0: lop (0,1); -1: lop (-1,1)
    ther = 0.5  # nguong 0 hay 0.5
    path_out_fig = path_eval + 'FIG/'
    path_out_kq = path_eval + 'KQ/'
    name_plot = 'AUC_AUPR-' + args.type_eval + args.dis_k_koxoa
    fig_file = path_out_fig + name_plot

    from GENERAL_UTILS.Q_eval import eval
    eval(args, path_eval, b_ix, e_ix, str_fi1, str_fi2, cla, ther, path_out_fig, path_out_kq, fig_file,
         loai_tam='_fold')
    return


# %%
if __name__ == "__main__":
    bg_time = time.time()
    for i in range(len(methods)):
        print('*' * 100)
        method = methods[i]
        print('method:', method)
        print('-' *50)
        ###################################(1)######################################
        print('Gen prob of each triplet')
        for loop_i in range(1, args.nloop + 1):
            print('Loop ', loop_i)
            main(args, loop_i)

        ###################################(2)######################################
        print('Find opt')
        opt_para = ''
        set_para = np.arange(0.2, 0.8, 0.01)
        AUC_all = []
        AUPR_all = []
        for i in range(set_para.shape[0]):
            auc_all_loop_i, aupr_all_loop_i = [], []
            for loop_i in range(1, args.nloop + 1):
                print('loop:', loop_i)
                auc_loop_i, aupr_loop_i = main2(args, set_para[i], opt_para, \
                                                loop_i, method)  # change alpha, belta
                auc_all_loop_i.append(auc_loop_i)
                aupr_all_loop_i.append(aupr_loop_i)
            AUC_all_para_i_value = np.mean(auc_all_loop_i)
            AUPR_all_para_i_value = np.mean(aupr_all_loop_i)
            AUC_all.append(AUC_all_para_i_value)
            AUPR_all.append(AUPR_all_para_i_value)
            print(i,' ',AUC_all_para_i_value, AUPR_all_para_i_value)
        AUC_all = np.array(AUC_all)
        AUPR_all = np.array(AUPR_all)
        print(np.max(AUC_all), np.argmax(AUC_all))
        print(np.max(AUPR_all), np.argmax(AUPR_all))
        result = np.array([set_para, AUC_all, AUPR_all])
        result= result.T

        if args.type_eval == 'kfold':
            np.savetxt(args.fi_out + 'Results/Combination/AUC_AUPR_all_' + args.db + \
                   '_' + method + '.csv', result, delimiter=',')
        print('NOTE!, save optimal parameters!!!!')

        ###################################(3)######################################
        print('Save final')
        if args.type_eval == 'kfold':
            result = np.genfromtxt(args.fi_out + 'Results/Combination/AUC_AUPR_all_' + \
                               args.db + '_' + method + '.csv', delimiter=',')
        opt_para = 1  # save the last time
        AUC_all_loop_fi = result[:, 1]
        alpha_all_loop_fi = result[:, 0]

        AUC_all, AUPR_all = [], []
        print('METHOD: ', method)
        for loop_i in range(1, args.nloop + 1):
            auc_i, aupr_i = main2(args, alpha_all_loop_fi[np.argmax(AUC_all_loop_fi)], \
                                  opt_para, loop_i, method)
            AUC_all.append(auc_i)
            AUPR_all.append(aupr_i)
        print('AUC final mean:', np.mean(np.array(AUC_all)))
        print('AUPR final mean:', np.mean(np.array(AUPR_all)))

        #QX

