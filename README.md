# 布料分拣复核 API

按归一化光谱向量建图、随机游走求首达化验点的材质概率，并评估“暂撤哪条化验结论最影响分箱”的 FastAPI 服务。

## 规则实现

- **建批**：一次提交 1–100 份带整数编号的布样、欧氏距离阈值、已有化验结论。同一批次内布样编号唯一，化验结论只能指向批次内布样，向量维度一致。服务端不校验向量是否已 L2 归一化（由上游归一化），但拒绝零向量和非有限值。
- **连边**：两份布样欧氏距离 **≤ 阈值** 即相连（边界含等号，带 1e-12 相对容差）。
- **随机游走**：未化验布样每步在其邻居中等概率选一个，化验布样为吸收态。每个连通分量内解调和方程，得到“首次到达某材质化验点”的概率。
- **入箱建议**：某材质概率 **≥ 0.8** 时建议入箱（含 0.8），否则待化验。概率为空（连通分量内没有化验点）时也待化验，原因 `no_reachable_lab_point`。
- **复核**：对每条已化验结论，只移除该标签、保留布样与连边，重算概率；只统计**原先可入箱且未化验**的布样中“改箱或转待化验”的数量。返回影响最多的已化验布样编号（并列取最小编号）及受影响布样的前后去向。全部为 0 时返回 `needs_priority_review=false` 和“无需优先复核”提示。
- **主动送检队列**：`POST /batches/{batch_id}/lab-queue` 提交当前版号和 1–10 个名额（字段 `quota`，兼容 `count`）。候选为全部未化验布样，按材质概率不确定度 `1 − 最高概率` 排序；连通分量内没有可达化验点时不确定度记为 **1.0**。
  - **先覆盖分量**：每个有候选的连通分量先取得一件（分量按其最佳候选的不确定度降序、布样编号升序排列；名额少于分量数时优先不确定度高的分量）。
  - **再全局补齐**：剩余名额按不确定度降序、布样编号升序（不确定度按 12 位小数比较，数学上并列时取小编号）。
  - 队列绑定 `(版号, 名额)` 写入批次；同版同名额重试**逐字节相同**，不同名额各自独立计算。携带旧版号返回 `409 version_conflict`。候选不足时返回实际可选的件数，不会凑数或重复。
- **版号**：建批版号为 1，每次成功改填/撤销化验结论后 +1。`GET` 查询随响应附带当前版号；复核结果也带 `version`。改填/撤销必须携带提交时看到的版号，旧版号返回 `409 version_conflict`。
- **原子更新**：化验结论与入箱建议在同一把锁内更新。计算失败返回 `500 computation_failed`，版号、标签与建议全部回滚到操作前。成功改填/撤销时，同一锁内同时清空**送检队列缓存**和**复核缓存**，再递增版号——因此任何基于旧标签的队列或复核结果都不可能被当作当前结果返回；写入失败回滚时缓存保持原样，旧版查询仍然有效。
- **并发**：版号校验、队列缓存命中/计算、缓存写入都在同一把存储锁内完成。同一版同名额的并发请求要么命中同一缓存，要么由第一个请求算好后其余请求读取同一结果，不会出现两份不同或超过名额的队列。该保证以单进程为边界（见“已知限制”）。

## 运行

```bash
docker compose up --build -d
curl -sS http://127.0.0.1:8000/health
```

交互式文档：<http://127.0.0.1:8000/docs>

本地（不用 Docker）：

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/batches` | 建批，返回版号 1、标签与建议 |
| `GET` | `/batches/{batch_id}` | 查询标签、建议和当前版号 |
| `GET` | `/batches/{batch_id}/review` | 复核影响分析（含版号） |
| `POST` | `/batches/{batch_id}/lab-queue` | 主动送检队列，body 带 `version` 与 `quota`(1–10) |
| `PUT` | `/batches/{batch_id}/samples/{sample_id}/label` | 改填/新增化验结论，body 带 `version` |
| `DELETE` | `/batches/{batch_id}/samples/{sample_id}/label?version=N` | 撤销化验结论 |
| `GET` | `/health` | 健康检查 |

### 可调用请求

完整脚本见 [`examples/demo.sh`](examples/demo.sh)，示例数据见 [`examples/batch.json`](examples/batch.json)。

```bash
# 建批
curl -sS -X POST http://127.0.0.1:8000/batches \
  -H 'Content-Type: application/json' \
  --data-binary @examples/batch.json

# 复核：哪条化验结论最影响分箱（返回附 version）
curl -sS http://127.0.0.1:8000/batches/DEMO-001/review

# 主动送检队列：当前版号 1，2 个名额
curl -sS -X POST http://127.0.0.1:8000/batches/DEMO-001/lab-queue \
  -H 'Content-Type: application/json' \
  --data '{"version": 1, "quota": 2}'

# 改填结论（必须带当前版号）
curl -sS -X PUT http://127.0.0.1:8000/batches/DEMO-001/samples/3/label \
  -H 'Content-Type: application/json' \
  --data '{"material": "棉", "version": 1}'

# 撤销结论（必须带当前版号）
curl -sS -X DELETE 'http://127.0.0.1:8000/batches/DEMO-001/samples/1/label?version=2'
```

`examples/demo.sh` 会依次执行健康检查、建批、复核、送检队列、同版同名额幂等重试、改填、旧版号冲突（含队列接口）、撤销及再次复核与送检。

### 响应示例（送检队列）

```json
{
  "batch_id": "DEMO-001",
  "version": 1,
  "quota": 2,
  "selected_count": 2,
  "unlabelled_count": 3,
  "component_count": 1,
  "covered_component_count": 1,
  "sample_ids": [2, 4],
  "samples": [
    {
      "position": 1,
      "sample_id": 2,
      "component_index": 0,
      "uncertainty": 0.384615384615,
      "probabilities": {"棉": 0.384615384615, "羊毛": 0.615384615385}
    },
    {
      "position": 2,
      "sample_id": 4,
      "component_index": 0,
      "uncertainty": 0.153846153846,
      "probabilities": {"棉": 0.153846153846, "羊毛": 0.846153846154}
    }
  ],
  "message": "按材质概率不确定度选出 2 份未化验布样，覆盖 1 个连通分量。"
}
```

`uncertainty` = `1 − 最高概率`，即“第二可能材质追上来的余地”；无化验点可达时为 `1.0`。`position` 从 1 开始，表示送检顺序。

### 响应示例（复核）

```json
{
  "batch_id": "DEMO-001",
  "version": 1,
  "priority_review_sample_id": 3,
  "priority_review_material": "羊毛",
  "max_affected_count": 2,
  "needs_priority_review": true,
  "affected_samples": [
    {
      "sample_id": 4,
      "before": {"status": "recommend_bin", "material": "羊毛", "probabilities": {"棉": 0.153846153846, "羊毛": 0.846153846154}, "reason": null},
      "after":  {"status": "recommend_bin", "material": "棉",   "probabilities": {"棉": 1.0}, "reason": null}
    }
  ],
  "candidates": [
    {"sample_id": 1, "material": "棉", "affected_count": 0, "affected_sample_ids": []},
    {"sample_id": 3, "material": "羊毛", "affected_count": 2, "affected_sample_ids": [4, 5]}
  ],
  "baseline_binnable_count": 2,
  "message": "建议优先复核已化验布样 3（材质 羊毛）：暂撤该结论会使 2 个原先可入箱的未化验布样改箱或转为待化验。"
}
```

### 错误格式

```json
{"error": {"code": "version_conflict", "message": "...", "details": {"submitted_version": 1, "current_version": 2}}}
```

错误码：`validation_error`(422)、`batch_not_found`/`sample_not_found`(404)、`batch_exists`/`version_conflict`/`label_absent`(409)、`computation_failed`(500)。

## 测试

```bash
python3 -m pytest -q
```

覆盖：首达概率解析值（2/3、1/3、0.8、0.846…）、阈值含等号、连通分量拆分、无化验点分量、0.8 边界、并列取最小编号、只统计原先可入箱样品、改箱与转待化验两种去向、旧版号 409、改填/撤销后的建议联动、计算失败回滚、100 份上限、缓存随版本失效等。

送检队列另有专项覆盖：不确定度定义（含无化验点 = 1.0）、先按分量覆盖再全局补齐、不确定度并列取小编号、名额大于候选数时截断且不重复、同版同名额幂等、不同名额独立、旧版号 409、改填/撤销后队列缓存失效、计算失败不写缓存且不递增版号、失败回滚后旧队列仍有效、8 线程并发重试得到完全一致的队列且不超名额、名额 1–10 的边界校验。

## 已知限制

- 批次存储在进程内存中，服务重启后丢失；单进程内以锁保证版号、队列与更新原子性，多副本部署需要改用共享存储（届时“同版同名额重试一致”和“并发不超额”需要数据库唯一约束或分布式锁来维持）。
- 送检队列在每个 `(版号, 名额)` 上缓存完整结果；标签变更会整体清空。最多 100 份布样、10 个名额时缓存条目极少，暂不做过期淘汰。
- 概率计算为 O(未化验点数³) 的稠密线性求解；单批 ≤100 份，实测毫秒级（100 份、25 条结论的复核约 0.35 秒）。若未来放大批次规模，应改用稀疏求解。
- 没有鉴权与限流，需在网关层补齐。
- 服务端按提交值计算距离，不校验向量是否真的已归一化；上游必须保证归一化，否则距离含义会偏差。
