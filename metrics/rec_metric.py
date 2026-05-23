import Levenshtein

class RecMetric(object):
    def __init__(self, main_indicator='acc'):
        self.main_indicator = main_indicator
        self._reset()

    def __call__(self, post_result, batch=None):
        preds, labels = post_result
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        
        for (pred, pred_score), (target, _) in zip(preds, labels):
            # ==========================================
            # 防护层 1：彻底清理可能残留的填充符和不可见空格
            # ==========================================
            pred = pred.replace(" ", "").replace("<BLANK>", "")
            target = target.replace(" ", "").replace("<BLANK>", "")
            
            # ==========================================
            # 防护层 2：统一大小写
            # 解决字典中存在大写 'X'，而预测可能输出小写导致的 0 acc 问题
            # ==========================================
            pred = pred.lower()
            target = target.lower()
            
            # 计算编辑距离（分母最小为 1，防止全空字符串报错）
            norm_edit_dis += Levenshtein.distance(pred, target) / max(
                len(pred), len(target), 1)
            
            # 精确匹配判断
            if pred == target:
                correct_num += 1
            all_num += 1
            
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        
        # ==========================================
        # 防护层 3：防除零保护（Batch 过滤导致的崩溃）
        # ==========================================
        if all_num == 0:
            return {
                'acc': 0.0,
                'norm_edit_dis': 0.0
            }
            
        return {
            'acc': correct_num / all_num,
            'norm_edit_dis': 1 - norm_edit_dis / all_num
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        if self.all_num == 0:
            acc = 0.0
            norm_edit_dis = 0.0
        else:
            acc = 1.0 * self.correct_num / self.all_num
            norm_edit_dis = 1 - self.norm_edit_dis / self.all_num
            
        self._reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    def _reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0.0  # 规范为浮点数