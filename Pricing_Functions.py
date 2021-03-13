############################## INCLUDE RELEVANT PYTHON MODULES ##############################

import numpy as np

GAUSS_CONST = 1.0 / np.sqrt(2.0 * np.pi)

############################## DEFINE PRICING FUNCTIONS FOR CREDIT PRODUCTS ##############################

def CDO_index_price(parameters, modelParameters, freq, mat):
    spreads = []
    payment_delta = 1.0 / freq
    payment_dates = [i * payment_delta for i in range(1, freq * max(mat) + 1)]
    bank_acc = np.exp([p * parameters.r for p in payment_dates])
    LM = LinearModel(parameters, modelParameters)
    LM.simulate()
    L = LM.loss
    Z = [1 - l for l in L]
    alive = Z[:: int(parameters.time_steps ) // (max(mat)* freq)]
    for T in mat:
        temp = alive[: freq * T + 1]
        temp2 = -(np.diff(temp))
        assert temp2.all() >= 0
        btemp = bank_acc[: freq * T]
        fee_leg = payment_delta * np.sum(temp[1:] / btemp)
        prot_leg = np.sum(temp2 / btemp)
        spreads.append((1.0 - parameters.rec) * prot_leg / fee_leg * 10 ** 4)
    return spreads

def CDO_tranches_price(parameters, modelParameters, freq, mat, tranches, samples, NSim):
    spreads = []
    payment_delta = 1.0 / freq
    payment_dates = [i * payment_delta for i in range(1, freq * max(mat) + 1)]
    bank_acc = np.exp([p * parameters.r for p in payment_dates])
    MCLosses = []
    for i in range(NSim):
        LM = LinearModel(parameters, modelParameters, samples[i])
        LM.simulate()
        L = LM.loss
        MCLosses.append(L[:: int((parameters.time_steps / parameters.total_time) // freq)])
    for T in mat:
        spreads_for_mat = []
        temp = [loss[: freq * T+1] for loss in MCLosses]
        btemp = bank_acc[: freq * T]
        for [a, d] in tranches:
            Z = [np.maximum([d - (1. - parameters.rec)*l for l in loss], 0)
                 - np.maximum([a - (1. - parameters.rec)*l for l in loss], 0) for loss in temp]
            averageZ = np.sum(Z, axis=0)/NSim
            fee_leg = payment_delta * np.sum(averageZ[1:] / btemp)
            diff = -(np.diff(averageZ))
            assert diff.all() >= 0
            prot_leg = np.sum(diff / btemp)
            spreads_for_mat.append(prot_leg / fee_leg * 10 ** 4)
        spreads.append(spreads_for_mat)
    return spreads
