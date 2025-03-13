# Transformer

由 Coursera 课程作业 [Transformers Architecture with TensorFlow
](https://www.coursera.org/learn/nlp-sequence-models/programming/roP5y/transformers-architecture-with-tensorflow) 改编而来。

## 环境

- python=3.7
- tensorflow=2.4
- numpy=1.19
- pytest=7.4

```sh
conda env create -f environment.yml
```

## 测试

```sh
pytest
```

## 注意事项

- 由于 GPU 精度不同，计算结果可能有误差，可能会导致测试不通过。
- 使用的环境不同也会导致计算结果不同，从而导致测试不通过。因此请使用 `environment.yml` 文件安装环境。
