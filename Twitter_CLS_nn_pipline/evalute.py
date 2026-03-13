import torch

"""
模型的效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger, valid_loader):
        self.config = config
        self.model = model
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.valid_data = valid_loader
        self.stats_dict = {"correct":0, "wrong":0} # 存储结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct":0, "wrong":0}
        for idx, batch_data in enumerate(self.valid_data):
            input_ids      = batch_data["input_ids"].to(self.device)
            attention_mask = batch_data["attention_mask"].to(self.device)
            labels         = batch_data["labels"].to(self.device)
            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask)
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc
    
    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label, dim=-1).item()   # 通过.item()取出tensor中的值
            true_label= true_label.squeeze().item()
            if true_label == pred_label:
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return 

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)