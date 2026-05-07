# CATLASS 进阶实践

本文档按照由浅入深的次序，组织CATLASS模板库中的进阶材料，读者可按照个人兴趣查阅具体内容。

- [catlass_optimize_guidance](./contents/advanced/catlass_optimize_guidance.md) - 介绍模板库下的基础调优方式，包括如何通过Tiling调参、应用不同的Dispatch策略的方式，快速获得性能提升。
- [matmul_template_summary](./contents/advanced/matmul_template_summary.md) - 对模板库的`examples`目录内已有的`matmul`模板设计进行介绍，包含样例模板清单、理论模板清单、工程优化清单、模板应用浅述，可用于matmul性能调优时参考。
- [swizzle_explanation](./contents/advanced/swizzle_explanation.md) - 对模板库中`Swizzle`策略的基本介绍，这影响了AI Core上计算基本块间的顺序。
- [dispatch_policies](./contents/advanced/dispatch_policies.md) - 对模板库在`Block`层面上`BlockMmad`中的一个重要模板参数`DispatchPolicy`的介绍。