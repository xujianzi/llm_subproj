# 简历项目条目 — PRISM Model

---

## 项目名称

**PRISM — COVID-19 序列预测模型**
*Pandemic Response Inference via Sequence Modeling*

---

## 一句话描述

基于 ZIP Code 级别移动出行数据与社会结构特征，构建时序深度学习模型预测 COVID-19 周度感染率曲线。

---

## 核心要点

- 设计**双流神经网络架构**，分别用 LSTM / Transformer 编码动态出行时序 `(B, T, 3)`，用 MLP 编码静态社会经济特征 `(B, 15)`，通过特征融合统一解码为完整预测序列，避免自回归误差累积

- 针对时序 + 跨区域数据提出**差异化归一化策略**：特征采用跨 ZIP z-score 保留区域间差异，目标值采用 ZIP 内 z-score 使模型专注学习曲线形态而非绝对量级，显著提升训练收敛（train R² 0.23 → 0.82）

- 以 ZIP Code 为单位划分训练 / 测试集，保证每个 ZIP 时间序列完整，**避免数据泄漏**

- 融合 ACS 年度普查数据与周度移动出行数据，设计跨时间粒度 join 流程，构建覆盖 **98 个 ZIP Code、38 周**的多模态面板数据集

- Transformer 模型测试集 **R² = 0.68**，LSTM 测试集 **R² = 0.58**

---

## 技术栈

`Python` · `PyTorch` · `LSTM` · `Transformer` · `时间序列预测` · `pandas` · `深度学习` · `社会经济数据分析`

---

## 投递方向说明

| 投递方向 | 重点突出 |
|---------|---------|
| 数据科学 / ML | 双流架构设计动机、差异化归一化策略、R² 指标 |
| 算法工程 | Transformer vs LSTM 对比实验、特征工程流程 |
| 软件工程 | 模块化代码结构（config / loader / model / main 分离） |
