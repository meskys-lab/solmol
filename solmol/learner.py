from fastai.text.all import *
from torch import nn

from solmol.model.metrics import accuracy, mcc, auc, recall, precision


def get_learner(model: nn.Module, dataloaders: DataLoaders, model_dir: Union[str, Path] = 'models') -> Learner:
    learner = Learner(dls=dataloaders,
                      model=model,
                      loss_func=nn.BCEWithLogitsLoss(),
                      model_dir=model_dir,
                      wd=0.1,
                      metrics=[accuracy, mcc, auc, recall, precision])
    return learner
