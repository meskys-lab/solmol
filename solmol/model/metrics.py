from fastai.torch_core import flatten_check
from sklearn.metrics import recall_score, roc_auc_score, matthews_corrcoef, average_precision_score
from torch import ByteTensor, FloatTensor, Tensor



def prep_pred(pred):
    _pred = pred.sigmoid()
    _pred = (_pred > 0.5).type(FloatTensor).detach().cpu()
    return _pred


def accuracy(pred: Tensor, targ: Tensor) -> float:
    _pred = prep_pred(pred)
    _pred, _targ = flatten_check(_pred, targ)
    correct = (_pred == _targ.type(FloatTensor))
    acc = correct.type(FloatTensor).mean() * 100
    return acc.detach().cpu().numpy()


def recall(pred: Tensor, targ: Tensor) -> float:
    _pred = prep_pred(pred)
    _pred, _targ = flatten_check(_pred, targ)
    r = recall_score(targ.detach().cpu(), _pred)
    return r


def auc(pred: Tensor, targ: Tensor) -> float:
    _pred = prep_pred(pred)
    _pred, _targ = flatten_check(_pred, targ)
    r = roc_auc_score(targ.detach().cpu(), _pred)
    return r


def mcc(pred: Tensor, targ: Tensor) -> float:
    _pred = prep_pred(pred)
    _pred, _targ = flatten_check(_pred, targ)
    m = matthews_corrcoef(targ.detach().cpu(), _pred)
    return m


def precision(pred: Tensor, targ: Tensor) -> float:
    _pred = prep_pred(pred)
    _pred, _targ = flatten_check(_pred, targ)
    p = average_precision_score(targ.detach().cpu(), _pred)
    return p
