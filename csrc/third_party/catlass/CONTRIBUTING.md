# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行[CLA协议签署](https://gitcode.com/cann/community#%E7%AD%BE%E7%BD%B2cla)，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过[Issue](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4issue%E5%A4%84%E7%90%86issue%E4%BB%BB%E5%8A%A1)进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。


开发者贡献场景主要包括：

- Bug修复

  如果您在本项目中发现了某些Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。
  
- 贡献新算子样例

  您在使用模板库的过程中，如果对某算子的参考实现有泛化性增强/性能优化思路，或实现了某类新算子，希望将其贡献为参考样例，欢迎您进行样例算子贡献。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对新的样例算子予以说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。
  
  新增算子样例请放置在`examples/`目录下的`${id}_${op_name}`子目录下，其中`${id}`是唯一递增次序，`${op_name}`表示新增算子名称。
  ```bash
  catlass/examples              # 算子样例主目录
  ├── ${id}_${op_name}          # 单算子样例目录
  │   ├── CMakeLists.txt        # CMake编译文件
  │   ├── README.md             # 单算子样例说明文档
  │   └── ${op_name}.cpp        # 主文件
  ...
  ```
  其中`README.md`是对该算子的简要说明，应包含“代码组织”、“使用示例”两方面内容（可参考Matmul的[README文档](https://gitcode.com/cann/catlass/blob/master/examples/00_basic_matmul/README.md)），以便开发者理解。

- 贡献模板库代码
  
  如果您对模板库代码某一部分有新的特性实现，欢迎您在Issue中提出新的想法与设计。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对新的特性设计予以说明，项目成员会与您进行沟通确认，并在`include/catlass/`下提供合适的位置以贡献您的特性代码。

  同时，您需要在提交的Issue中评论“/assign”或“/assign @yourself”，认领该Issue并在后续完成代码上库。

- 文档纠错

  如果您在本项目中发现某些文档描述错误，或内容不够清晰，欢迎您新建Issue进行反馈和修复。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。
  
- 帮助解决他人Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。