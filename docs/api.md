# 前后端接口约定

本文档约定后端本地服务与前端网页之间的通信格式。开发阶段可依据本协议并行开发，前端可使用下方示例数据进行调试。

- **默认服务地址**：`http://127.0.0.1:8000`，可在 `config.json` 中修改 `host` 与 `port`
- **编码格式**：UTF-8，`application/json`
- **同源与相对路径**：后端同时托管 `frontend/index.html`，前端调用接口须使用相对路径 `/api/...`，禁止写死主机和端口。

---

## 1. 获取预设测试句

用于在页面加载时渲染一键填入的对比测试句。

- **请求**：`GET /api/examples`
- **响应 200**：
```json
{
  "examples": [
    {
      "tag": "正常好评对照",
      "text": "今天快递上午下单下午就到了，太快了"
    },
    {
      "tag": "Level 1 显性反讽",
      "text": "这游戏真好玩，玩了五分钟就退款了"
    },
    {
      "tag": "Level 2 常识反讽",
      "text": "快递速度真快，整整走了一个星期"
    },
    {
      "tag": "Level 2 高难度翻车案例",
      "text": "手机续航真强，一天只充三次电"
    }
  ]
}
```

---

## 2. 单句情感对比分析

同时运行 `SnowNLP` 与 `Erlangshen-RoBERTa` INT8 模型，返回各自结果及分歧标记。

- **请求**：`POST /api/analyze`
- **请求体**：
```json
{
  "text": "快递速度真快，整整走了一个星期"
}
```

- **正常响应 200**：
```json
{
  "text": "快递速度真快，整整走了一个星期",
  "snownlp": {
    "label": "中性",
    "positive_prob": 0.486
  },
  "roberta": {
    "label": "负面",
    "confidence": 0.745,
    "positive_prob": 0.255,
    "negative_prob": 0.745
  },
  "disagree": true,
  "elapsed_ms": 18.4
}
```

### 字段说明
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `text` | `string` | 去除首尾空白后的输入文本 |
| `snownlp.label` | `string` | `"正面"`、`"负面"` 或 `"中性"`，阈值由 `config.json` 控制 |
| `snownlp.positive_prob` | `float` | `SnowNLP` 正面概率，范围 0 ~ 1，保留 3 位小数 |
| `roberta.label` | `string` | `"正面"` 或 `"负面"` |
| `roberta.confidence` | `float` | 最终判定类别置信度，范围 0.5 ~ 1，保留 3 位小数 |
| `roberta.positive_prob` | `float` | 预测为正面的概率 |
| `roberta.negative_prob` | `float` | 预测为负面的概率 |
| `disagree` | `boolean` | 两个模型判定结论是否不一致，可用于前端高亮提示 |
| `elapsed_ms` | `float` | 本次推理耗时，单位毫秒 |

- **错误响应 400**（输入为空或非法 JSON）：
```json
{
  "error": "请输入待检测的中文文本"
}
```
