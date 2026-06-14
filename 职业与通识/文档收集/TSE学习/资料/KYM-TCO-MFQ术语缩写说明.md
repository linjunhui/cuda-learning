# KYM、TCO、MFQ 术语缩写说明文档

本文档系统整理KYM、TCO、MFQ方法中涉及的所有术语和缩写，便于快速查阅和理解。

---

## 📋 目录

1. [核心方法缩写](#核心方法缩写)
2. [KYM相关术语](#kym相关术语)
3. [TCO相关术语](#tco相关术语)
4. [MFQ相关术语](#mfq相关术语)
5. [PPDCS建模方法术语](#ppdcs建模方法术语)
6. [测试设计相关术语](#测试设计相关术语)
7. [其他相关术语](#其他相关术语)

---

## 核心方法缩写

### KYM
- **全称**：Know Your Mission
- **中文**：了解你的使命/了解测试任务
- **定义**：一种价值剖析（ValueOpsy）的思维框架，用于在任务开始时从宏观上思考任务的上下文和价值
- **核心思想**：价值认知先行，避免细节陷阱，建立全景图

### TCO
- **全称**：Test Coverage Outline
- **中文**：测试覆盖大纲
- **定义**：用于整理测试覆盖范围，是TDR（Test Driven Requirement development）的第二步
- **作用**：细化功能点，确保测试覆盖的完整性，为后续建模和测试设计提供输入

### MFQ
- **全称**：Modeling, Function Interaction, Quality Attribute
- **中文**：单功能建模、功能交互、质量属性
- **定义**：测试分析和测试设计的核心方法，代表三个测试维度
- **组成**：
  - **M**：Modeling（单功能建模）
  - **F**：Function Interaction（功能交互）
  - **Q**：Quality Attribute（质量属性）

---

## KYM相关术语

### CPPM（KYM的四个维度）

- **C** - **Customer**（客户）
  - **中文**：客户
  - **含义**：Who & Why，识别"人"的部分
  - **包含**：Customers（客户）、Users（用户）、Stakeholders（干系人）

- **P** - **Product**（产品）
  - **中文**：产品
  - **含义**：What，识别"产品"的部分
  - **包含**：范围、业务规则、价值创造逻辑

- **P** - **Project**（项目）
  - **中文**：项目
  - **含义**：How，识别"项目"的部分
  - **包含**：团队、依赖、交付件、进度计划

- **M** - **Mission**（使命）
  - **中文**：使命
  - **含义**：Why，任务的目标和价值
  - **作用**：明确要达成什么

> **注意**：KYM的CPPM维度不包括Process。Process是PPDCS建模的维度。

### CIDTESTD（KYM的八个要素）

- **C** - **Customer**（用户）
- **I** - **Information**（信息）
- **D** - **Developer**（开发者）
- **T** - **Test**（测试）
- **E** - **Environment**（环境）
- **S** - **Schedule**（进度）
- **T** - **Test Item**（测试项）
- **D** - **Deliverable**（交付件）

### CUS（Customer的三个组成部分）

- **C** - **Customers**（客户）
  - **中文**：客户
  - **定义**：与我们直接有交易、商务往来的人或企业

- **U** - **Users**（用户）
  - **中文**：用户
  - **定义**：直接使用产品或服务的人或企业

- **S** - **Stakeholders**（干系人）
  - **中文**：干系人
  - **定义**：不直接参与开发，但关心产品或服务的利益相关者

### INSEBDPS（Product的引导词启发式）

- **I** - **Information**（参考信息）
  - **中文**：参考信息
  - **定义**：与OUA相关但不属于OUA的参考信息

- **N** - **Naming Conventions**（术语定义）
  - **中文**：术语定义
  - **定义**：对术语、专有名词进行解释

- **S** - **Scope**（范围）
  - **中文**：范围
  - **定义**：从大的方面描述需求的边界范围

- **E** - **Exclusions**（除外）
  - **中文**：除外
  - **定义**：从大的方面描述不包含在本需求范围的情况

- **B** - **Business Rules**（业务规则）
  - **中文**：业务规则
  - **定义**：描述需求内容的主体部分，包括功能、非功能等方面的业务逻辑

- **D** - **Document**（参考文档）
  - **中文**：参考文档
  - **定义**：与该需求相关的各类参考资料

- **P** - **Problems Unsolved**（未解决的问题）
  - **中文**：未解决的问题
  - **定义**：尚未解决的技术或非技术类问题

- **S** - **Standard**（标准规范）
  - **中文**：标准规范
  - **定义**：该业务需求必须满足的标准规范

### GDZH（KYM的四个价值判断逻辑）

- **G** - **Gang Xu**（刚需）
  - **中文**：刚需
  - **定义**：一定要做的需求，有明确的应用场景和用户，价值足够大
  - **判断方法**：识别刚需的具体场景；如果不做这个需求，会怎样？

- **D** - **Du Li**（独立）
  - **中文**：独立
  - **定义**：该需求的价值是否可以独立体现，所在的团队可以独立交付完成
  - **判断方法**：是否存在与其他系统/特性/需求的依赖？

- **Z** - **Zhi Jie**（直接）
  - **中文**：直接
  - **定义**：相比当前理解的需求，用户有没有更直接的需求？
  - **判断方法**：用户的需求是否可以更直接地满足？

- **H** - **He Li**（合理）
  - **中文**：合理
  - **定义**：价值创造逻辑链条的合理性，价值实现逻辑链条的合理性
  - **判断方法**：价值创造逻辑是否通畅？价值实现逻辑是否合理？

### KYM其他术语

- **OUA** - **Object Under Analysis**
  - **中文**：被分析对象
  - **定义**：KYM分析的目标对象

- **ValueOpsy** - **Value Opsys**
  - **中文**：价值剖析
  - **定义**：KYM是一种价值剖析的思维框架

- **2-3-4原则**
  - **2**：两个思考视角（Now现在、Future未来）
  - **3**：三个思考维度（CPPM）
  - **4**：四个价值判断逻辑（GDZH）

---

## TCO相关术语

### TCO核心术语

- **TCO** - **Test Coverage Outline**
  - **中文**：测试覆盖大纲
  - **定义**：用于整理测试覆盖范围

- **TDR** - **Test Driven Requirement development**
  - **中文**：测试驱动需求开发
  - **定义**：TCO是TDR的第二步

### TCO横向逻辑

- **MFQ**：Modeling（建模）、Function Interaction（功能交互）、Quality Attribute（质量属性）
- **Bugs**：缺陷
- **Questions**：问题
- **Risks**：风险

### TCO纵向逻辑

- **归纳**：将相关信息归类整理
- **总结**：提炼关键信息
- **梳理**：结构化呈现

---

## MFQ相关术语

### MFQ流程术语

- **KYM** → **TCO** → **Modeling** → **Tcon** → **TC**

  - **KYM**：Know Your Mission（了解测试任务）
  - **TCO**：Test Coverage Outline（测试覆盖大纲）
  - **Modeling**：建模
  - **Tcon**：Test Condition（测试条件）
  - **TC**：Test Case（测试用例）

### MFQ三个维度

- **M** - **Modeling**（单功能建模）
  - **中文**：单功能建模
  - **定义**：对单个功能进行建模分析

- **F** - **Function Interaction**（功能交互）
  - **中文**：功能交互
  - **定义**：功能之间的交互关系

- **Q** - **Quality Attribute**（质量属性）
  - **中文**：质量属性
  - **定义**：非功能性需求，如性能、可靠性等

### Given-When-Then（建模结构）

- **Given**（预置条件）
  - **中文**：预置条件
  - **定义**：测试执行前的初始状态

- **When**（执行动作/输入）
  - **中文**：执行某个动作/输入
  - **定义**：触发测试的操作或输入

- **Then**（预期结果）
  - **中文**：发生了什么/关注被测系统
  - **定义**：执行后的预期结果或系统行为

### TEST原则（建模方法选择）

- **T** - **Triggers**（触发关键词）
  - **中文**：触发关键词
  - **定义**：抓住核心功能

- **E** - **Essentials**（关键要素）
  - **中文**：关键要素
  - **定义**：尝试不同特征

- **S** - **Spanning Differences**（跨越差异）
  - **中文**：跨越差异
  - **定义**：围绕既定目标

- **T** - **Targets**（目标）
  - **中文**：目标
  - **定义**：测试的目标

---

## PPDCS建模方法术语

### PPDCS（五种建模方法）

- **P** - **Process**（流程）
  - **中文**：流程
  - **定义**：需求中有明显的"业务流程"的含义
  - **建模技术**：流程图技术
  - **适用工具**：流程图、判定表、判定树、因果图

- **P** - **Parameter**（参数）
  - **中文**：参数
  - **定义**：对象中存在很多参数，并且这些参数互相之间有一定的逻辑联系
  - **建模技术**：决策表、决策树
  - **适用工具**：判定表、判定树、因果图

- **D** - **Data**（数据）
  - **中文**：数据
  - **定义**：每个数据都有它特殊的范围值，不同数据的范围可能存在限制
  - **建模技术**：等价类划分、边界值

- **C** - **Combination**（组合）
  - **中文**：组合
  - **定义**：因子个数多，每个因子存在多个可能的状态（水平）
  - **建模技术**：正交试验、正交分析、正交矩阵、pairwise
  - **适用工具**：因子-状态表

- **S** - **State**（状态）
  - **中文**：状态
  - **定义**：涉及到多种状态，最好是针对同一个对象的多个状态
  - **建模技术**：状态流转图
  - **三要素**：状态、事件、转移迁移

### PPDCS相关术语

- **Pairwise**（成对测试）
  - **中文**：成对测试
  - **定义**：一种组合测试技术，测试所有因子对之间的组合

- **等价类划分**（Equivalence Partitioning）
  - **中文**：等价类划分
  - **定义**：将输入域划分为若干等价类，每个等价类选择代表性测试数据

- **边界值**（Boundary Value）
  - **中文**：边界值
  - **定义**：测试输入域的边界值，包括边界值和边界附近的值

- **正交试验**（Orthogonal Array）
  - **中文**：正交试验
  - **定义**：一种组合测试方法，用最少的测试用例覆盖最多的因子组合

- **状态流转图**（State Transition Diagram）
  - **中文**：状态流转图
  - **定义**：描述系统状态及状态之间转换关系的图形化表示

---

## 测试设计相关术语

### 测试用例要素

- **测试输入**（Test Input）
  - **中文**：测试输入
  - **定义**：测试用例的输入数据

- **执行条件**（Execution Condition）
  - **中文**：执行条件
  - **定义**：测试执行的前置条件

- **预期结果**（Expected Result）
  - **中文**：预期结果
  - **定义**：测试执行后的预期输出

### 测试用例原则

- **正确性**（Correctness）
  - **中文**：正确性
  - **定义**：用例设计正确

- **完备性**（Completeness）
  - **中文**：完备性
  - **定义**：覆盖全面

- **连贯性**（Coherence）
  - **中文**：连贯性
  - **定义**：逻辑连贯

- **可判定性**（Determinability）
  - **中文**：可判定性
  - **定义**：结果可判定

- **可执行性**（Executability）
  - **中文**：可执行性
  - **定义**：可实际执行

### 测试目标

- **证实**（Verification）
  - **中文**：证实
  - **定义**：验证功能正确性

- **证伪**（Falsification）
  - **中文**：证伪
  - **定义**：发现缺陷

- **预防缺陷**（Defect Prevention）
  - **中文**：预防缺陷
  - **定义**：通过测试设计预防问题

---

## 其他相关术语

### 需求相关

- **PR** - **Product Requirement** / **Pull Request**
  - **中文**：产品需求 / 拉取请求
  - **定义**：根据上下文可能是产品需求或代码拉取请求

- **MR** - **Market Requirement** / **Merge Request**
  - **中文**：市场需求 / 合并请求
  - **定义**：根据上下文可能是市场需求或代码合并请求

- **INVEST**
  - **中文**：需求划分原则
  - **定义**：用于判断需求是否满足单功能划分原则
  - **组成**：
    - **I** - Independent（独立的）
    - **N** - Negotiable（可协商的）
    - **V** - Valuable（有价值的）
    - **E** - Estimable（可估算的）
    - **S** - Small（小的）
    - **T** - Testable（可测试的）

### 测试方法相关

- **FMEA** - **Failure Mode and Effects Analysis**
  - **中文**：失效模式与影响分析
  - **定义**：一种系统性的分析方法，用于识别潜在的失效模式及其影响

### 流程相关

- **需求测试流程**：
  ```
  方案评审 → 测试策略、用例编写 → 需求评审 → 输出文本用例 → 测试执行
  ```

- **MFQ流程**：
  ```
  KYM → TCO → Modeling → Tcon → TC
  ```

---

## 快速查找索引

### 按字母顺序

| 缩写 | 全称 | 中文 | 分类 |
|------|------|------|------|
| **C** | Customer | 客户 | KYM |
| **C** | Combination | 组合 | PPDCS |
| **CIDTESTD** | Customer, Information, Developer, Test, Environment, Schedule, Test Item, Deliverable | 用户、信息、开发者、测试、环境、进度、测试项、交付件 | KYM |
| **CPPM** | Customer, Product, Project, Mission | 客户、产品、项目、使命 | KYM |
| **CUS** | Customers, Users, Stakeholders | 客户、用户、干系人 | KYM |
| **D** | Data | 数据 | PPDCS |
| **F** | Function Interaction | 功能交互 | MFQ |
| **GDZH** | Gang Xu, Du Li, Zhi Jie, He Li | 刚需、独立、直接、合理 | KYM |
| **INSEBDPS** | Information, Naming Conventions, Scope, Exclusions, Business Rules, Document, Problems Unsolved, Standard | 参考信息、术语定义、范围、除外、业务规则、参考文档、未解决问题、标准规范 | KYM |
| **KYM** | Know Your Mission | 了解你的使命 | 核心方法 |
| **M** | Modeling | 单功能建模 | MFQ |
| **MFQ** | Modeling, Function Interaction, Quality Attribute | 单功能建模、功能交互、质量属性 | 核心方法 |
| **OUA** | Object Under Analysis | 被分析对象 | KYM |
| **P** | Process | 流程 | PPDCS |
| **P** | Parameter | 参数 | PPDCS |
| **PPDCS** | Process, Parameter, Data, Combination, State | 流程、参数、数据、组合、状态 | 建模方法 |
| **Q** | Quality Attribute | 质量属性 | MFQ |
| **S** | State | 状态 | PPDCS |
| **TC** | Test Case | 测试用例 | 测试设计 |
| **TCO** | Test Coverage Outline | 测试覆盖大纲 | 核心方法 |
| **Tcon** | Test Condition | 测试条件 | 测试设计 |
| **TDR** | Test Driven Requirement development | 测试驱动需求开发 | TCO |
| **TEST** | Triggers, Essentials, Spanning Differences, Targets | 触发关键词、关键要素、跨越差异、目标 | MFQ |

---

## 使用说明

1. **查找术语**：使用"快速查找索引"按字母顺序查找
2. **理解上下文**：每个术语都标注了所属分类（KYM、TCO、MFQ等）
3. **深入学习**：参考相关文档了解术语的详细用法和实践

---

**文档版本**：v1.0  
**最后更新**：2024年  
**维护者**：TSE学习团队

