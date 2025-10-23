import numpy as np


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict



def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def Kappa(confusion_matrix):
    """计算kappa值系数"""
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    mean_recall = np.nanmean(recall)
    mean_precision = np.nanmean(precision)
    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    kappa_value = Kappa(confusion_matrix)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1, 'mrecall':mean_recall, 'mprecision':mean_precision, 'kappa':kappa_value}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict

# def cm2score(confusion_matrix):
#     hist = confusion_matrix
#     tp = hist[1, 1]
#     fn = hist[1, 0]
#     fp = hist[0, 1]
#     tn = hist[0, 0]
#     # acc
#     oa = (tp + tn) / (tp + fn + fp + tn + np.finfo(np.float32).eps)
#     # recall
#     recall = tp / (tp + fn + np.finfo(np.float32).eps)
#     # precision
#     precision = tp / (tp + fp + np.finfo(np.float32).eps)
#     # F1 score
#     f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
#     # IoU
#     iou = tp / (tp + fp + fn + np.finfo(np.float32).eps)
#     # pre
#     pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
#     # kappa
#     kappa = (oa - pre) / (1 - pre)
#     score_dict = {'Kappa': kappa, 'IoU': iou, 'F1': f1, 'OA': oa, 'recall': recall, 'precision': precision, 'Pre': pre}
#     return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']
