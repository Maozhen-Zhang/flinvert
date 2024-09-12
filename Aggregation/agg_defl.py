import copy

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger


class AggDeFL(FedAvg):
    def __init__(self, conf, metric=None):
        super().__init__(conf)
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.MUOD = MUODV()
        self.aggVGN = VGN()

    def aggregateWeightDeFL(self, PreGlobalModel, clients, choice_id):
        CLPC = CPA(Delta=0.05)
        CLP = CLPC.Judge(ClientDatasetSize, TransGNs)

        TotalParam = {i: clients[i].local_model.state_dict() for i in choice_id}
        PreGlobalWeight = PreGlobalModel.state_dict()

        GradVecs = []
        for clientID in choice_id:
            gradvec = TensorGradVec(TotalParam[clientID], 1, PreGlobalWeight)
            GradVecs.append(gradvec)

        # each epoch choice k client
        # gradvec sample len is n
        # reshape GradVecs from k*n to n*k to use detection
        DMat = []
        for i in range(len(GradVecs[0])):
            Dmat = []
            for j in range(len(GradVecs)):
                Dmat.append(GradVecs[j][i])
            DMat.append(Dmat)
        Bads = self.MUOD.detection(DMat, choice_id)
        GlobalParas, _ = self.aggVGN.AggParas(choice_id, PreGlobalWeight, self.RecvParas, self.RecvLens, Bads, CLP)
        return GlobalParas, _


class VGN:
    def __init__(self):
        self.Alphas = {}
        self.Betas = {}

    def Add(self, Id):
        self.Alphas[Id] = 1
        self.Betas[Id] = 1

    def AggParas(self, IDs, RPara, Paras, Lens, Bads, CLP=1):
        for ky in IDs:
            if ky not in self.Alphas.keys():
                self.Add(ky)

        Ls = {}
        Pks = {}
        for i in range(len(Paras)):
            Ky = IDs[i]
            Ls[Ky] = Lens[i]
            Pks[Ky] = self.Alphas[Ky] / (self.Alphas[Ky] + self.Betas[Ky])

        GDs = []
        GLs = []
        for i in range(len(Paras)):
            ky = IDs[i]
            if CLP == 1:
                if i not in Bads:
                    GDs.append(Paras[i])
                    GLs.append(Ls[ky] * Pks[ky])
            else:
                GDs.append(Paras[i])
                GLs.append(Ls[ky] * Pks[ky])
        Res = wavgParas(GDs, GLs)

        BIDs = []
        for i in range(len(Paras)):
            ky = IDs[i]
            if i in Bads:
                self.Betas[ky] += 1
                BIDs.append(ky)
            else:
                self.Alphas[ky] += 1

        return Res, -1


class MUODV:
    def __init__(self):
        self.AGrades = {}
        self.BGrades = {}
        self.T = 3
        self.Warmup = False
        self.Round = 0
        self.Scores = {}
        self.BNum = []

    def MUOD(self, Mat):
        Is = []
        Ia = []
        Im = []
        Sims = []
        for i in range(len(Mat)):
            PCs = []
            Alpha = []
            Beta = []
            sim = []
            for j in range(len(Mat)):
                X = Mat[i]
                Xi = Mat[j]
                pc = pearsonr(X, Xi)[0]
                if np.isnan(pc):
                    pc = 0.00001
                beta = np.cov(X, Xi)[0][1] / max(np.var(Xi), 0.00000000001)
                alpha = np.mean(X) - beta * np.mean(Xi)
                if pc < 0.98:
                    PCs.append(pc)
                    Alpha.append(alpha)
                    Beta.append(beta)

            Is.append(abs(np.mean(PCs) - 1))
            Im.append(abs(np.mean(Alpha)))
            Ia.append(abs(np.mean(Beta) - 1))

        Loc0 = max(min(LorenzThr(Is), 25), 10)
        Loc1 = max(min(LorenzThr(Im), 25), 10)
        Loc2 = max(min(LorenzThr(Ia), 25), 10)
        Up0 = np.percentile(Is, 100 - Loc0)
        Up1 = np.percentile(Im, 100 - Loc1)
        Up2 = np.percentile(Ia, 100 - Loc2)
        Outlier0 = []
        Outlier1 = []
        Outlier2 = []
        L = len(Mat)
        for i in range(len(Is)):
            val = Is[i]
            if val >= Up0:
                Outlier0.append(i)
            val = Im[i]
            if val >= Up1:
                Outlier1.append(i)
            val = Ia[i]
            if val >= Up2:
                Outlier2.append(i)

        Finds = Outlier2
        return Finds

    def bestLayer(self):
        Grades = {}
        for ky in self.AGrades.keys():
            grade = self.AGrades[ky] / (self.AGrades[ky] + self.BGrades[ky])
            Grades[ky] = grade

        Good = list(Grades.keys())
        if self.Warmup:
            Good = []
            SGds = sorted(Grades.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.T):
                Ky = SGds[i][0]
                Good.append(Ky)
        return Good

    def detection(self, DMat, UIDs):
        self.Round += 1
        # Init every client Agrades and Bgrades as 1
        if len(self.AGrades) == 0:  # init AGrades and BGrades with 1
            for i in range(len(DMat)):  # len DMat is client params length
                self.AGrades[i] = 1
                self.BGrades[i] = 1
            self.T = max(self.T, int(len(DMat) - 3))  # threshold, T = max(3,7) ?

        Good = self.bestLayer()
        Votes = {}
        Records = {}
        ThL = 0
        for i in range(len(DMat)):
            Mat = DMat[i]
            layer = i
            Bad = self.MUOD(Mat)
            Records[i] = Bad
            if i in Good:
                if len(Bad) > 0:
                    ThL += 1
                for b in Bad:
                    if b not in Votes.keys():
                        Votes[b] = 1
                    else:
                        Votes[b] += 1

        ThL = max(1, ThL)
        Bads = []
        Step = 0
        while len(Bads) <= 2:
            Bads = []
            ThL -= Step
            for ky in Votes.keys():
                if Votes[ky] >= ThL:
                    Bads.append(ky)
            Step += 1
            if ThL <= 0:
                break

        L = len(Bads)
        for ky in Records.keys():
            Get = Records[ky]
            VL = len(list(set(Get) & set(Bads)))
            BL = len(Get) - VL
            if VL > L / 2:
                self.AGrades[ky] += 1
            else:
                self.BGrades[ky] += 1

        if self.Round >= 5:
            self.Warmup = True

        return Bads


class CPA:
    def __init__(self, Delta=0.05, Recent=10):
        self.R = Recent
        self.Round = 0
        self.Threshold = Delta
        self.FGNs = []
        for i in range(20):
            self.FGNs.append(0)
        self.MLim = 5

    def Proc(self, Ls, Gs):
        self.Round += 1
        SumL = np.sum(Ls)
        FGN = 0
        for i in range(len(Ls)):
            FGN += Ls[i] / SumL * Gs[i]
        return FGN

    def Judge(self, Ls, Gs):
        FGN = self.Proc(Ls, Gs)

        Old = np.mean(self.FGNs[-self.R:]) + 0.00000001
        self.FGNs.append(FGN)
        New = np.mean(self.FGNs[-self.R:])

        Is = 0
        if (New - Old) / Old >= self.Threshold or self.Round <= self.MLim:
            Is = 1

        return Is


def Lorenz(a, b, X):
    return np.exp(-(a - X) / b)


def tangent(a, b, X):
    A = np.exp(-(a - X) / b) * 1 / b
    X1 = X
    Y1 = Lorenz(a, b, X1)

    B2 = Y1 - A * X1
    X2 = -B2 / A
    return X2


def LorenzThr(Vs):
    Vs = sorted(Vs, reverse=True)
    Fs = []
    Ls = []
    for i in range(len(Vs)):
        if np.isnan(Vs[i]) == False:
            Fs.append(i + 1)
            Ls.append(np.log(Vs[i]))

    Xs = np.array(Ls).reshape((len(Ls), 1))
    Ys = np.array(Fs).reshape((len(Fs), 1))

    LModel = LinearRegression()
    LModel.fit(Xs, Ys)

    b = LModel.coef_[0][0]
    a = LModel.intercept_[0]

    Cut1 = tangent(a, b, 0)
    Cut2 = min(tangent(a, b, Cut1), len(Vs))

    R = (Cut1 + (Cut2 - Cut1) * 0.5) / len(Vs) * 100

    return R


def TensorGradVec(T1, Sym, T2):
    Res = {}
    Kys = list(T1.keys())
    C = 0
    for ky in Kys:
        if "weight" in ky:
            V1 = T1[ky].cpu().detach().numpy()
            V2 = T2[ky].cpu().detach().numpy()
            res = V1 - Sym * V2
            res = res.reshape(-1)
            if len(res) > 500:
                gvec = []
                L = int(len(res) / 500)
                gets = []
                for i in range(len(res)):
                    if i % L == 0:
                        gets.append(res[i])
                Res[ky] = np.abs(gets)
            else:
                Res[ky] = np.abs(res)
            C += 1
    return Res


def wavgParas(Paras, Lens):
    Res = copy.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = Lens[i] / np.sum(Lens)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res
