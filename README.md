# Intern_experiences_at_Baidu
## 1. read_data.py
PaddlePaddle按数据条数来读取数据的，成batch;
其中data形式为：
山\t东\t省1\n3\t0\t0\nprov\tHED\tHED\n

vocab形式为：
HED\nprov\n

## 2. paddle动转静
出现assign错误，首先就要考虑paddle的版本问题
## 3. Dependency Parsing
>将句子变为Directed graph, node表示word, edge是relation, root指向最关键的那个

>一个clssifier决定左右输入决定有没有关系，binary classifier; 关系是什么为multi-class classification

>单纯classifier会制造出矛盾，制造出不合法的tree, 因此可以用maximum spanning tree, 看score大的。
## 4. Consitituency Parsing
>将句子里面所有可以组成一个单位的词汇找出来，每个有个单位

>给一个句子，给一个span， binary classifier决定span是不是constituenct, 然后用multi-class classifier决定label

解法-
>1. Chart-based methods
>>有可能出现矛盾状况-->穷举所有可能的树状结构(CKY), 合法-->找分数高的

>2. transition-based methods
>>RNN决定采取哪个action

# NER知识积累
## 1. from "Modularized Interaction Network for Named Entity Recognition"
Why-
>sequence labeling-based model难以捕获长距离实体；segment-level models难以捕获segment内word之间的dependency; boundary detection与type prediction是相关的【DDparser的话可以看作是联合的】

What-
>文章主要内容：捕获了segment-level information, word-level dependencies, 结合一种交互机制，支持边界检测和类型预测之间的信息共享（boundary detection and type prediction）

小知识-
>1) NER可以包含两类: Sequence labeling-based methods, Segment-based methods
>2) NER分着看的话，可以将其分成两步: boundary detection, type prediction
>3) pointer network (from "Pointer networks")


## 2. from "Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition"
Why-
>当前span-based methods存在的问题：1）低质量candidate span多->计算量大【感觉biaffine生成的s_arc矩阵进行loss计算的时候也是这样，只用到了几个位置，其余都是non-entity】；2）识别长的实体的能力较差；3）没有完全利用boundary 信息；4）将部分匹配的实体当作是negative example

What-
>本文做的工作：将NER分为boundary+给label(联合的), 具体来说，由表示得到一些span，然后找一些high overlap的span当作proposal span【本文叫这个名字】, low overlap的叫contextual span; 过滤掉contextual span然后还有个机制可以调节 boundary, 最终进一个classifier

## 3. Efficient Second-Order TreeCRF for Neural Dependency Parsing
Eisner algorithm from "Bilexical grammars and their cubic-time parsing algorithms"
>computing the highest- scoring projective dependency tree under an arc-factored model，  using bottom–up dynamic programming, storing solutions to sub-problems in a table

What-
>提出second-order TreeCRF extension to the biaffine parser

Why-
>TreeCRF复杂度高
>biaffine parser是采用的local token-wise cross-entropy training loss(first-order)
>max_margin traning algorithm 会预测出一个最高得分的tree
>biaffine得到的score不如TreeCRF得到的概率(??听起来貌似合理，但是Why?)
>边缘概率支持Mininum Bayes Risk decoding(???)

小知识-
>1. biaffine parser是graph-based dependency parser
>2. biaffine可以看作是local head selection策略

## 4. UniRE: A Unified Label Space for Entity Relation Extraction
Entity relation extraction: 不分成 entity detection and relation classification， 联合进行

## 5. GEMNET: Effective Gated Gazetteer Representations for Recognizing Complex Entities in Low-context Input
一. What-
>本文提出GEMNET 模型，包含一个encoder for Contextual Gazetteer Representations (CGRs) + 一个gated Mixture-of-Experts (MoE) method to fuse CGRs with Contextual Word Representations (CWRs) from any word-level model (like Bert).

二. Why-

Gazetteers的引入与related works-
>标注的 NER 数据只能覆盖到有限的实体集合, 但现实可能存在无限的实体空间. 于是引入gazetteers. 
>>1. 有些人将其用作one-hot然后与Bert产生的表示结合，这会导致 feature “under-training"; 
>>2. 用 gazetteers 来训练一个 subtagger model 来 识别span，缺点在于needs retraining and evaluation
on gazetteer updates

Mixture-of-Experts (MoE) Models 与 related works-
>A gating network is trained to dynamically weight experts perinstance, according to the input
>>有些人 proposed a Mixture of Entity Experts (MoEE) approach where they train an expert layer for each entity type, and then combine them using an MoE approach--缺点在于没有用到gazetteer,并且没有得到的representation与word representation是independent


Gazetteers 的limitations-

>1. gazetteer feature representation（One hot embedding of gazetter feature cannot capture contextual info and span boundary. 单独训练的gazetteer feature 难以训练并且feature 效果并不好);
>2. their integration with contextual models（often add extra features to a word-level model’s Contextual Word Representations (CWRs), 会导致sub-optimal); 
>3. and a lack of data.

三. Model-
看那个图，比较清晰

四. 小知识-

>1. Mention Detection (MD) is a simpler task of identifying entity spans, without the types

## 6. Better Feature Integration for Named Entity Recognition
Why
>1. 当前方法在捕获contextual info(captured bu linear sequences)与structured info(captured by dependency tree)时, 聚焦于stack LSTM与GNN， 真正的两者之间的关系没有捕获到
>2. 难以捕获长距离dependency


What
>提出Synergrid-LSTM：在LSTM基础上加了一个graph-encoded representation,看原文献图就好，比较清晰.
