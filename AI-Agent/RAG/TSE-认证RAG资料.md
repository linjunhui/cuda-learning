# RAG（检索增强生成）知识体系

## 目录

1. [RAG基础概念](#rag基础概念)
2. [RAG知识库框架](#rag知识库框架)
3. [RAG评测体系](#rag评测体系)
4. [RAG最佳实践](#rag最佳实践)

---

## RAG基础概念

### 1. 大模型的本质与局限

#### 大模型的本质

1. **基于统计学的语言模型**：能够理解语言意图并根据意图进行任务分配，具有强大的推理能力
2. **将输入的文本转化为高层次的特征表示**：这种表示能够捕捉文本中的语义和语法信息，表现出了很强的泛化能力
3. **语言知识的概率表达**：通过统计学习对语言各层次规律建模，表征语言生成的先验分布，从而具备语言预测生成能力
4. **大模型是一个隐形知识库**
5. **压缩即学习**：OpenAI首席科学家Ilya Sutskever提出，压缩可能就是学习的本质，大模型本质就是压缩
6. **大模型是"造梦机"**：OpenAI科学家Andrej Karpathy认为，从某种意义上说，大语言模型的全部工作恰恰就是制造"幻觉"，大模型就是"造梦机"

**核心观点：**
> 大模型是一个通过巨量数据而掌握巨量知识，以此可以推理、泛化的概率函数。"幻觉"是其基本特征。

#### 大模型的局限

1. **幻觉问题**：LLM文本生成的底层原理是基于概率的token by token形式，因此会不可避免地产生"一本正经的胡说八道"的情况

2. **无用户建模**：大语言模型没有建模特定用户的能力，对不同用户给出同样的反应和回复，无法进行个性化的对话

3. **知识盲点**：大模型通过预训练获得通用语言能力，但不具备专业领域的知识。对某些专业问题无法做出准确回答。有些知识不停地有更新，大模型需要在训练和微调时才能灌入新知识

4. **记忆力有限**：大语言模型参数量虽然很大，但仍然无法记住大量具体的事实知识。容易在需要记忆的任务上表现不佳

5. **时效性问题**：大语言模型的规模越大，训练的成本越高，周期也就越长。那么具有时效性的数据也就无法参与训练，所以也就无法直接回答时效性相关的问题

6. **数据安全问题**：通用大语言模型没有企业内部数据和用户数据，那么企业想要在保证安全的前提下使用大语言模型，最好的方式就是把数据全部放在本地，企业数据的业务计算全部在本地完成

### 2. RAG的定位与价值

#### 技术路线对比

通过二维坐标系展示不同技术的关系：

**坐标轴：**
- **Y轴**：外部知识需求（Low - High）
- **X轴**：模型适配程度（Low - High）

**技术分类：**

1. **提示词工程**（左下角）：外部知识需求低，模型适配程度低
   - Prompt初步尝试
   - Adding few shot case COT（添加少量示例的思维链）

2. **RAG**（左上角）：外部知识需求高，模型适配程度低
   - Naive RAG：添加相关上下文段落
   - Advanced RAG：索引/预检索/后检索优化
   - Modular RAG：多个模块的有机组合

3. **精调**（右下角）：外部知识需求低，模型适配程度高
   - Generator fine-tuning（生成器精调）
   - Retriever fine-tuning（检索器精调）
   - Collaborative fine-tuning（协同精调）

4. **综合方案**（右上角）：外部知识需求高，模型适配程度高

#### RAG与知识库的核心观点

1. **RAG增强大模型生成能力**：RAG通过检索外部知识库，补充和增强Prompt信息，以支持大型模型更好地生成内容

2. **外部知识库是必要补充**：考虑到大型模型的训练周期、训练成本和数据安全等因素，外部知识库是大型模型不可或缺的核心补充

3. **RAG与模型训练协同**：RAG与将相应知识用于大型模型的预训练和精调并不冲突，两者应相互配合，共同提升大型模型的生成质量

4. **语料/知识质量是关键**：语料和知识的质量是确保大型模型和知识库有效性的关键因素

### 3. RAG定义与工作流程

#### RAG的定义

**RAG的定义：**
> 检索增强生成 (Retrieval Augmented Generation, RAG) 是一种技术，它通过从数据源中检索信息来辅助大语言模型 (Large Language Model, LLM) 生成答案。

**工作机制：**
简而言之，RAG 结合了搜索技术和大语言模型的提示词功能，即向模型提出问题，并以搜索算法找到的信息作为背景上下文，这些查询和检索到的上下文信息都会被整合进发送给大语言模型的提示中。

**公式：**
- `RAG = Retrieval Augmented Generation = 检索 + 增强 + 生成`
- `GPT = Generative Pre-trained Transformer = 生成式 + 预训练 + Transformer框架`

#### RAG核心流程

RAG核心流程分为三个阶段：

1. **数据准备（Data Preparation）**
   - 私域数据（.txt, email, .pdf, JSON, Word, xlsx）
   - Embedding Model（嵌入模型）
   - Semantic Vector（语义向量）
   - Vector Database（向量数据库）

2. **数据检索（Data Retrieval）**
   - Users（用户）
   - Question（问题）
   - Query Vector（查询向量）
   - Knowledge（知识）

3. **LLM生成（LLM Generation）**
   - 大语言模型（ChatGPT、文心一言、通义千问）
   - Prompt（提示词）
   - Answer（答案）

### 4. 与大模型交互的通用框架

#### Prompt模版的构成要素

1. **限制（Constraints）**：管理大模型的"公域知识"
   - 角色定义
   - 背景说明
   - 上下文
   - 输入要求
   - 输出要求
   - 过程要求
   - 任务
   - 示例

2. **补充知识（Supplementary Knowledge）**：提供大模型不具备的"私域知识"
   - **RAG技术**
   - **补充知识的类型**：
     - 实时查询信息
     - 关系数据库信息
     - 知识图谱信息
     - 向量库信息
   - **业务相关知识**：
     - 需求
     - 代码
     - 用例
     - 术语
     - 检查单
     - 历史经验

---

## RAG知识库框架

### 1. RAG核心流程总览

#### 数据摄入流程（Data Ingestion Flow）

- **原始文档** → **分割**（段落、语义、正则、摘要、上下文增强）
- **向量化**（Embedding模型、元数据、分级索引、HyDE、数据库存储）
- **数据库**（向量数据库）

#### 用户查询处理流程（User Query Processing Flow）

- **用户问题** → **检索前处理**（问题转换、问题扩写、子查询、路由选择）
- **检索**（向量检索、关键字检索、混合检索、合并检索）
- **检索后处理**（元数据过滤、相似度得分、结果重排）
- **prompt组装**（prompt模板、内容替换、上下文压缩）
- **LLM**（模型选择、总结输出、结构化输出）
- **答案**

### 2. 数据准备阶段

#### 分割（Chunking）

**分割方法：**
1. **字符串分割**：根据特定的字符串（如句号"。"、换行符等）进行文本分割
2. **NLTK分割**：利用自然语言工具包 (NLTK) 进行语义识别，然后根据识别结果进行文本分割
3. **MarkDown分割**：专门针对Markdown格式的文本进行结构化分割
4. **递归分割**：使用一组分隔符，以分层和迭代的方式将输入文本分割成更小的块

**最佳实践：**
1. **清洗数据**：确保数据质量，删除HTML标记或特定元素
2. **选择一个范围的块大小**：从较小的块开始（128或256个tokens），尝试较大的块（512或1024个tokens）
3. **评估每个块大小的性能**：对不同大小的块进行标记，创建Embedding并保存，运行一系列查询来评估质量

#### 向量化（Embedding）

**定义：**
Embedding是将文本数据转化为向量矩阵的过程。它将高维度的数据（如文字、图片、音频）映射到低维度空间，形成数值向量形式。

**价值：**
- **降维**：将高维数据映射到低维空间
- **捕捉语义信息**：语义上相近的词在向量空间中也会相近
- **适应性**：通过数据驱动的方式学习，自动适应数据的特性
- **泛化能力**：能够捕捉数据的内在规律，对未见过的数据给出合理表示
- **可解释性**：可以通过可视化工具（如t-SNE）观察和理解Embedding的结构

**方法：**
- **Word2Vec**：CBOW（通过上下文预测中心单词）和Skip-gram（通过中心单词预测上下文）
- **GloVe**：基于共现矩阵的模型，使用统计方法计算单词之间的关联性
- **FastText**：在Word2Vec的基础上添加了字符级别的n-gram特征
- **大模型的Embeddings**：如OpenAI的`text-embedding-ada-002`（最长输入8191个tokens，输出维度1536）

### 3. 检索阶段

#### 基本检索方法

**1. 基本索引检索（Basic Index Retrieval）**
- Documents → Single vector store of all chunks vectors → Query → Top k relevant chunks → LLM → Answer

**2. 层次检索（Hierarchical Index Retrieval）**
- Query → Index of summary vectors → Vector store of all chunks vectors → Top k relevant chunks → LLM → Answer

**3. 假设性问题和HyDE（Hypothetical Questions and HyDE）**
- 让LLM为每个文档块生成一个假设性问题，并将这些问题以向量形式嵌入
- 在运行时，针对这些假设性问题向量的索引进行查询搜索
- 检索完成后，将原始文本块作为上下文发送给LLM以获取答案
- 通过提高查询和假设性问题之间的语义相似度，显著提高搜索质量

#### 混合检索（Hybrid Search）

**定义：**
结合了两种不同的搜索方法：
1. 基于关键词的传统搜索方法，使用稀疏检索算法（如tf-idf或行业标准BM25）
2. 现代的语义或向量搜索

**流程：**
1. Query和Documents → 并行检索
   - **向量检索**：Vector index → Top k results
   - **稀疏检索**：sparse n-grams index (BM25) → Top k results
2. **Reciprocal Rank Fusion**（倒数排名融合）：将两种方法的结果融合，重新排名
3. Top n → LLM → 混合检索输出

**关键：**
正确结合具有不同相似度评分的检索结果。通过Reciprocal Rank Fusion算法解决，该算法会对检索结果进行重新排名，以产生最终输出。

### 4. 检索前处理（Pre-Retrieval）

#### 查询转换（Query Transformation）

**流程：**
1. Query → LLM
2. LLM生成Subquery 1和Subquery 2（或更通用的查询）
3. 两个子查询 → Vector index → Top k results for each subquery
4. 所有结果 → LLM → Answer

**示例：**
"在Github上，Langchain和LlamaIndex哪个框架的star更多？"不太可能直接在语料库中找到这样的比较，因此将这个问题分解为两个假设更简单和具体信息检索的子查询：
- "Langchain在Github上有多少star？"
- "LlamaIndex在Github上有多少star？"

#### 查询路由（Query Routing）

**定义：**
查询路由是一种基于大型语言模型（LLM）的决策步骤，用于确定针对用户查询应采取的后续行动。

**常见选项：**
- 概括回答：直接提供一个总结性的答案
- 数据索引搜索：针对特定的数据索引执行搜索
- 多途径尝试与综合：尝试多种不同的途径来获取信息，然后将它们的结果综合成一个最终答案

**应用：**
- 选择合适的索引
- 选择数据存储位置（向量存储、图数据库、关系型数据库、索引层级结构）

**设置：**
在多文档存储场景下，一个常见的设置是结合使用一个概要索引（用于整体概述）和另一个文档块向量的索引（用于详细内容检索）。

### 5. 检索后处理（Post-Retrieval）

#### 上下文扩充（Context Expansion）

**Sentence Window Retrieval**

**流程：**
1. 检索到相关句子
2. 自动扩展上下文（增加前后文）
3. 将更完整的上下文提交给LLM

#### 自动合并检索（Parent-child chunks retrieval）

**流程：**
1. Query → Vector store of all child or leaf chunk vectors
2. Top k relevant child chunks
3. Parent chunks, linked to the retrieved child chunks
4. Documents
5. LLM → Answer

**说明：**
将几条结果自动补充中间内容，以更完整的上下文提交给大模型。

#### 响应合成器（Response Synthesizer）

**响应合成主要方法：**
1. 总结检索到的上下文，使其适应输入提示
2. 将检索到的上下文分块后逐次发送给大语言模型（LLM），以此迭代地精炼答案
3. 基于不同上下文块生成多个答案，然后将这些答案连接或总结起来

### 6. RAG架构演进

#### 三种RAG架构

**1. Naive RAG（朴素RAG）**
- User Query + Documents → Indexing → Retrieval → Prompt → Frozen LLM → Output

**2. Advanced RAG（高级RAG）**
- User Query → Pre-Retrieval（Query Routing, Query Rewriting, Query Expansion）
- Documents → Indexing
- Retrieval → Post-Retrieval（Rerank, Summary, Fusion）
- Prompt → Frozen LLM → Output

**3. Modular RAG（模块化RAG）**

**Modules（模块）：**
- Core RAG Loop：Rewrite, RAG, Rerank, Read, Retrieve
- External Modules：Routing, Search, Predict, Fusion, Memory, Demonstrate

**Patterns（模式）：**
- Naive RAG Pattern：Retrieve → Read
- Advanced RAG Pattern：Rewrite → Retrieve → Rerank → Read
- DSP Pattern：Demonstrate → Search → Predict
- ITER-RETGEN Pattern：循环的 Retrieve → Read

### 7. 特殊数据源RAG

#### 基于SQL知识库的RAG

**Python代码示例：**
```python
llm = CustomLLM()
host = 'localhost'
port = '3306'
username = 'root'
password = 'renhongliang'
database_schema = 'mydb'
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"

db = SQLDatabase.from_uri(mysql_uri,
                          include_tables=['job'],
                          sample_rows_in_table_info=1)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

**Prompt模板：**
- 你是一个MySQL专家
- 给定输入问题，首先创建语法正确的MySQL查询
- 查看查询结果并返回答案
- 最多查询5个结果（使用LIMIT子句）

**示例：**
- Question: Which company has the most number of jobs for an English Teacher in Canada
- SQLQuery: SELECT `company`, COUNT(*) AS `num_jobs` FROM `job` WHERE `title`='English Teacher' AND `location`='Canada' GROUP BY `company` ORDER BY `num_jobs` DESC LIMIT 5
- SQLResult: ZTE, 1
- Answer: The company with the most number of jobs for an English Teacher in Canada is ZTE with 1 job.

#### 基于知识图谱的RAG

**Neo4j图数据库示例：**

**Python代码：**
```python
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

llm = CustomLLM()
graph = Neo4jGraph()
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
response = chain.invoke({"query": "What was the cast of the Casino?"})
```

**Schema：**
- Node properties: Movie {imdbRating, id, released, title}, Person {name}, Genre {name}
- Relationships: (:Movie)-[:IN_GENRE]->(:Genre), (:Person)-[:DIRECTED]->(:Movie), (:Person)-[:ACTED_IN]->(:Movie)

**Cypher查询：**
```cypher
MATCH (m:Movie {title: "Casino"})<-[:ACTED_IN]-(p:Person)
RETURN p.name as Cast
```

**结果：** James Woods, Robert De Niro, Sharon Stone, Joe Pesci

---

## RAG评测体系

### 1. RAG评估框架

#### RAG评估四要素

1. **用户问题**（User Question）：用户提交的原始问题
2. **知识库应答**（Knowledge Base Response）：知识库处理以后返回的应答
3. **大模型应答**（Large Model Response）：大模型根据请求返回的应答
4. **真实答案**（True Answer）：问题的理想答案

**流程：**
用户问题 → 知识库 → 大模型 → 知识库应答 + 大模型应答 → 与真实答案对比评估

#### RAGAS 开源框架度量框架

**主要组件：**
- question（用户问题）
- contexts（上下文）
- answer（LLM生成的答案）
- ground_truth（标准答案）

### 2. RAG评估指标

#### 基础评估指标概念

**二分类模型基础指标：**
- **真正例**（True Positive, TP）：模型正确预测为正例的样本数
- **真反例**（True Negative, TN）：模型正确预测为反例的样本数
- **假正例**（False Positive, FP）：模型错误地将反例预测为正例的样本数
- **假反例**（False Negative, FN）：模型错误地将正例预测为反例的样本数

**评估指标公式：**
- **准确率**：`Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **精准率**：`Precision = TP / (TP + FP)`
- **召回率**：`Recall = TP / (TP + FN)`
- **F1值**：`F1 = 2 * Precision * Recall / (Precision + Recall)`

#### 忠诚度（Faithfulness）

**定义：**
忠诚度衡量生成的答案与给定上下文的事实一致性。它根据答案和检索到的上下文计算得出，结果缩放到 (0,1) 范围，值越高越好。

**公式：**
```
Faithfulness score = Number of claims in the generated answer that can be inferred from given context / Total number of claims in the generated answer
```

**计算方法：**
1. 从生成的答案中识别一组独立的声明
2. 将这些声明中的每一项与给定的上下文进行交叉检查，以确定是否可以从上下文中推断出它

**示例：**
- 问题：爱因斯坦出生于何时何地？
- 上下文：阿尔伯特·爱因斯坦 (1879年3月14日出生) 是一位出生于德国的理论物理学家
- 高忠实答案：爱因斯坦1879年3月14日出生于德国
- 低忠实度答案：爱因斯坦于1879年3月20日出生于德国（日期错误）

**评估方法：**
通过自然语言推理（Natural Language Inference）评估忠诚度：
- 给定上下文和答案中的陈述，判断每个陈述是否可以从上下文中推断出来
- 提供简短解释，然后给出判断（Yes/No）
- 最终计算忠诚度分数

#### 上下文相关性（Context Relevancy）

**定义：**
衡量问题和检索到的上下文的相关性，根据question和contexts计算。值落在(0, 1) 范围内，值越高表示相关性越好。

**公式：**
```
context relevancy = |S| / |Total number of sentences in retrieved context|
```
其中|S|表示相关句子数量

**示例：**
- 问题：法国的首都是哪里？
- 上下文1（高相关性）：法国位于西欧，其首都巴黎以其时装屋、卢浮宫等古典艺术博物馆和埃菲尔铁塔等古迹而闻名
  - Context Relevancy: 0.5000（1个相关句子/2个总句子）
- 上下文2（低相关性）：包含更多无关信息（葡萄酒、美食、历史等）
  - Context Relevancy: 0.2500（1个相关句子/4个总句子）

#### 上下文精度（Context Precision）

**定义：**
评估所有存在于事实中的相关项目是否在上下文中排名较高。理想情况下，所有相关的块（或上下文）都必须出现在顶层（即排名靠前）。

**公式：**
```
Context Precision@K = Σ(Precision@k × v_k) / Total number of relevant items in the top K results
```

其中：
```
Precision@k = true positives@k / (true positives@k + false positives@k)
```

**示例：**
- 问题：法国在哪里，首都是哪里？
- 事实真相：法国位于西欧，其首都是巴黎
- 上下文1：法国位于西欧，其首都巴黎以其时装屋、卢浮宫等古典艺术博物馆和埃菲尔铁塔等古迹而闻名
  - LLM判断：是（上下文精确）
- 上下文2：该国还以其葡萄酒和精致的美食而闻名。拉斯科的古代洞穴壁画、里昂的罗马剧院和巨大的凡尔赛宫都证明了其丰富的历史
  - LLM判断：否（上下文不精确，未提及首都）

#### 上下文召回率（Context Recall）

**定义：**
衡量检索到的上下文与知识库中正确信息（基本事实）的一致程度。根据 `ground truth`（真实答案）和 `retrieved context`（检索到的上下文）进行计算，值范围在0到1之间。

**公式：**
```
context recall = |GT sentences that can be attributed to context| / |Number of sentences in GT|
```

**计算方法：**
1. 将真实答案分解为单独的陈述
2. 对于每个基本事实陈述，验证它是否可以归因于检索到的上下文
3. 计算可归因的句子数量占总句子数量的比例

**示例：**
- 问题：法国在哪里，首都是哪里？
- 事实真相：法国位于西欧，其首都是巴黎
- 检索到的上下文：法国位于西欧，拥有中世纪城市、高山村庄和地中海海滩。该国还以其葡萄酒和精致的美食而闻名...
- 计算：
  - 陈述1："法国位于西欧" → 可归因（上下文提到了"法国位于西欧"）
  - 陈述2："其首都是巴黎" → 不可归因（上下文未提及首都）
  - Context Recall = 1 / 2 = 0.5000

#### 答案相关性（Answer Relevance）

**定义：**
评估生成的答案与给定提示的相关程度。
- 较低的分数分配给不完整或包含冗余信息的答案
- 较高的分数表示更好的相关性

**计算公式：**
$$
\text{answer relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{gi}, E_o)
$$

其中：
- $E_{gi}$ 表示问题中第i个句子的嵌入向量
- $E_o$ 表示答案的嵌入向量
- N表示问题中的句子数量

#### 答案语义相似性（Answer Semantic Similarity）

**定义：**
评估生成答案与基本事实之间语义相似性的方法。

**评估机制：**
- **基础**：评估基于 `ground truth`（基本事实）和 `answer`（生成的答案）
- **评分范围**：评估结果的取值范围在0到1之间
- **分数含义**：分数越高，表示生成的答案与实际答案之间的一致性越好

**评估步骤：**
1. **步骤1**：使用指定的嵌入模型对 `ground truth` 进行向量化
2. **步骤2**：使用相同的嵌入模型对生成的答案 `answer` 进行向量化
3. **步骤3**：计算这两个向量之间的余弦相似度

#### 答案正确率（Answer Correctness）

**定义：**
衡量生成答案与真实答案的准确性。该评估依赖于answer和ground truth，分数范围为0到1。分数越高，表示生成的答案与真实情况越接近，意味着正确性越好。

**关键方面：**
答案正确性包含两个关键方面：
1. **生成的答案与基本事实之间的语义相似性**
2. **事实相似性**

使用加权方案将这些方面结合起来以制定答案正确性分数。如果需要，用户还可以选择使用"阈值"将结果分数四舍五入为二进制。

#### 上下文实体召回率（Context Entities Recall）

**定义：**
召回率基于"ground truths"和"contexts"中共存实体的数量相对于仅在"ground truths"中存在的实体数量，给出检索上下文的召回率。

**简化定义：**
衡量从ground_truths中召回的实体比例。

**公式：**
```
context entity recall = |CE ∩ GE| / |GE|
```

其中：
- `CE`：Context Entities（上下文中对象的个数）
- `GE`：Ground Truth Entities（ground truths中对象的个数）
- `∩`：交集

**应用场景：**
在基于事实的用例中非常有用，例如旅行服务台和历史问答，帮助评估实体检索机制。

**示例计算（泰姬陵）：**
- **真相（Ground Truth）**：The Taj Mahal is an ivory-white marble mausoleum located on the right bank of the Yamuna River in Agra, India. In 1631, the Mughal emperor Shah Jahan commissioned the construction of this mausoleum to house the tomb of his favorite wife, Mumtaz Mahal.
- **步骤1**：查找基本事实中存在的实体
  - GE中的实体：['泰姬陵', '穆纳', '阿格拉', '1631', '沙贾汗', '蒙塔兹·玛哈']
- **步骤2**：查找上下文中存在的实体
  - CE1中的实体：['泰姬陵', '阿格拉', '沙贾汗', '蒙塔兹·玛哈', '印度']
  - CE2中的实体：["泰姬陵", "联合国教科文组织", "印度"]
- **步骤3**：计算实体召回率
  - CE1 = 4/6 = 0.667
  - CE2 = 1/6 = 0.167

### 3. RAG评估指标总结

**RAGAS评估指标：**
1. **答案相关性**（Answer Relevancy）：答案与问题的相关程度
2. **上下文精度**（Context Precision）：检索上下文的精确程度
3. **上下文相关性**（Context Relevancy）：上下文与问题的相关性
4. **忠实度**（Faithfulness）：答案是否基于提供的上下文
5. **答案相似性**（Answer Similarity）：答案与标准答案的语义相似度
6. **答案正确性**（Answer Correctness）：答案与标准答案的事实一致性
7. **上下文召回率**（Context Recall）：检索上下文是否覆盖标准答案
8. **上下文实体召回率**（Context Entities Recall）：检索到的实体是否覆盖标准答案中的实体

---

## RAG最佳实践

### 1. RAG的核心价值

**核心观点：**
1. **通过知识库对大模型进行知识补充是大模型的本质要求**
2. **RAG是一个多步骤的复杂工程**
3. **通过评估方式不断尝试是当前最好的方式**
4. **建立以知识为核心的快速迭代反馈的工程体系，重构软件工程**

### 2. RAG的核心流程

**研发过程**（创建知识）→ **大模型**（利用知识）→ **提示词工程** → **知识库**（沉淀知识）

**知识工程循环：**
- 研发过程：创建知识
- 知识库：沉淀知识
- 大模型：利用知识

**目标与愿景：**
- 成为领先的以大模型为中心的研发组织
- 建立以知识为核心的软件工程体系

### 3. 应用场景示例

#### 测试设计助手：通过需求生成文本用例

**角色定义：**
你是一个测试设计专家，熟悉各种测试设计的方法

**任务要求：**
- 根据需求分组输出详细的测试用例名称
- 采用流程图、状态机、边界值、等价类等软件测试设计方法
- 采用分类分组的方式输出测试用例
- 测试用例条目采用"用例名称--用例目的"的格式
- 给出的测试用例需要覆盖完整、全面
- 不输出无关的内容

**补充知识来源：**
- 背景知识
- 波及需求
- 相关用例
- 方案设计
- 具体代码
- 相关故障
- 历史经验
- checklist
- 场景因子
- 测试方法

**输出示例：**
```
登录功能
1.1 用例名称：用户正常登录
用例目的：验证系统在正常情况下用户登录功能的可用性和稳定性
```

### 4. Prompt模板示例

**Prompt模板：**
```
【任务描述】
假如你是一个专业的智能客服，请参考【背景知识】，回答问题：

【背景知识】
{content} //数据检索得到的相关文本

【问题】
{question} //用户问题
```

---

## 总结

### RAG技术路线选择

根据外部知识需求和模型适配程度：

1. **提示词工程**：外部知识需求低，模型适配程度低
2. **RAG**：外部知识需求高，模型适配程度低
3. **精调**：外部知识需求低，模型适配程度高
4. **综合方案**：外部知识需求高，模型适配程度高

### RAG关键成功因素

1. **数据质量**：语料和知识的质量是确保大型模型和知识库有效性的关键因素
2. **工程化能力**：RAG是一个多步骤的复杂工程，需要系统化的设计和实现
3. **持续评估**：通过评估方式不断尝试和优化，建立快速迭代反馈的工程体系
4. **知识沉淀**：建立以知识为核心的软件工程体系，实现知识的持续积累和复用

---

## 参考资源

- RAGAS: Retrieval-Augmented Generation Assessment Framework
- LangChain: Framework for developing applications powered by language models
- LlamaIndex: Data framework for LLM applications
- Neo4j: Graph Database Platform