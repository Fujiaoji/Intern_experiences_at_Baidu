# Intern_experiences_at_Baidu
## 1. read_data.py
PaddlePaddle按数据条数来读取数据的，成batch;
其中data形式为：
山\t东\t省1\n3\t0\t0\nprov\tHED\tHED\n

vocab形式为：
HED\nprov\n

## 2. paddle动转静
出现assign错误，首先就要考虑paddle的版本问题

# NER知识积累
## from "Modularized Interaction Network for Named Entity Recognition"
1) NER可以包含两类: Sequence labeling-based methods, Segment-based methods
2) NER分着看的话，可以将其分成两步: boundary detection, type prediction
3) pointer network (from "Pointer networks")
