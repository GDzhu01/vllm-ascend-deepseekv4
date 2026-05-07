# CATLASS性能调测

CATLASS示例工程可适配大多数[CANN](https://www.hiascend.com/cann)提供的调测工具，算子开发阶段，可基于CATLASS示例工程进行初步开发调优，无需关注具体的工具适配操作，待算子基础功能、性能达到预期，再迁移到其他工程中。

下述文档介绍使用[CANN](https://www.hiascend.com/cann)已有的工具进行调测、调优的开发实践。

## 功能调试

- [msDebug](./tools/msdebug.md) - 类gdb/lldb的调试工具[msDebug](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/optool/atlasopdev_16_0062.html)
  - ⚠️ **注意** 此功能依赖社区版`CANN`包版本为[8.2.RC1.alpha003](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha003)。
- [printf](./tools/print.md) - 基于[CCE Intrinsic](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/cceintrinsicguide/cceprogram_0001.html)，在算子device侧进行打印调试
  - ⚠️ **注意** 此功能依赖社区版`CANN`包版本在CANN 8.3及以上版本（如[8.3.RC1.alpha001](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha001)）。
- [ascendc_dump](./tools/ascendc_dump.md) - 基于[原生AscendC API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendcopapi/atlasascendc_api_07_0192.html)，对关键数据打点调测

## 性能调优

- [msProf&Profiling](./tools/performance_tools.md) - 基于性能调优工具[msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/Profiling/atlasprofiling_16_0010.html)和[Profiling](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/graph/graphdevg/atlasag_25_0056.html)进行调优实践
  - [单算子性能分析：msProf](./tools/performance_tools.md#使用msProf进行单算子性能分析)
  - [整网性能分析：Profiling](./tools/performance_tools.md#使用Profiling进行整网性能分析)
- [msTuner_CATLASS](../tools/tuner/README.md) - Tiling自动寻优工具