# CATLASS 样例设计文档

本文档汇总当前一些样例的设计思路和代码拆解，读者可按照个人兴趣查阅具体内容。

- [10_grouped_matmul_slice_m_per_token_dequant](./contents/example_design/10_grouped_matmul_slice_m_per_token_dequant.md) - 拆解模板库下的样例10，包含原型设计、样例实现、example组装、kernel实现方案。对希望了解“groupMatmul+后处理”类型的算子实现有指导价值。