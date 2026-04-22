# Eval 已知问题与改进备忘

> 持续更新。记录 eval 过程中发现的问题、已修复项和待观察项。

## 来源

2026-04-22，`eval --level 4 --runs 1`，模型 deepseek-v3.2

---

## 已修复

| 问题 | 根因 | 修复 |
|------|------|------|
| validate.py 输出协议 | L4 的 validate.py 输出纯文本"验证通过"，runner 期望最后一行是 `{"passed": true}` | 改为符合协议 |
| graph_planner 手动解析 JSON | 容易被 LLM 不规范输出搞挂 | 改用 Pydantic schema + `response_format` 约束 |
| verification 字段中文 | LLM 生成中文自然语言描述被当 shell 命令执行 | `run_verification` 加 `_is_shell_command` 检测 + prompt 强化 |
| eval CLI `--level` 缺少 4 | `choices=[1,2,3]` 没加 4 | 加上 |
| 80% token 警告不可见 | `LoopGuard` 的 80% 警告只写 logger，LLM 看不到 | 注入 conversation |

## 待观察（不急改，换强模型可能自动解决）

### 1. npm install 30s 超时（L4-express-auth）

BashTool 默认 30s timeout，npm install 冷缓存要更久。Agent 被迫去掉 bcryptjs/jsonwebtoken 用纯 JS 替代。

**可能的方案：**
- A：workspace 预装 node_modules（eval 场景专用）
- B：BashTool timeout 可配置（eval task.yaml 里指定）

### 2. Agent 修 bug 后不重跑测试（L4-script-to-project）

step 22 修好 bug 后花 11 步追 ghost bug，始终没重跑 pytest。强模型大概率不会犯这个错。

**可能的方案：** prompt 加"每次 edit 后立即重跑失败的测试"

### 3. EditFile no-op 静默成功

`old_content == new_content` 时 EditFile 不报错，Agent 以为改了但实际没变，导致循环 edit 同一段代码。

**可能的方案：** EditFile 检测 no-op 返回警告

### 4. 收尾纪律 / 软警告对弱模型效果差

80% token 警告注入 conversation 了，DeepSeek 口头回应但不改行为。收尾纪律 prompt 也没遵守（测试通过后继续做手动验证）。

**结论：** 硬上限兜底是必要的；prompt 层优化对弱模型收益有限。

---

## 设计原则

编程 Agent 的核心体验取决于模型能力。**机制层做到位（硬上限兜底、Pydantic schema 约束输出），不为弱模型过度定制。** 随模型迭代，上述待观察问题大概率自动消失。
