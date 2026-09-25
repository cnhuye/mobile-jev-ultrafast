# mobile-jev-ultrafast

[English](README.md) · **简体中文**

**给一句话，它自己把手机操作完。**

一个手机 agent，沿用 [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast)
的架构，只把设备层从 Chrome DevTools Protocol 换成了
[AutoX.js](https://github.com/cnhuye/AutoX) 的 MCP 服务。

[TypeSafe 的 Jev](https://docs.typesafe.ai/introduction) 从一份「当前屏幕上有什么」的
**编号元素表**里选**一个操作 + 一个目标**。执行由代码负责。模型不会输出坐标、选择器，
也不会输出任何可执行脚本。

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![手机端](https://img.shields.io/badge/%E6%89%8B%E6%9C%BA%E7%AB%AF-cnhuye%2FAutoX-orange)](https://github.com/cnhuye/AutoX)

---

## 这是什么

大多数「大模型操控手机」的方案，都是让模型直接吐一个点击坐标或者一个类 XPath 选择器。
这两种东西都是**猜测**，而且从截图到真正点击之间的几百毫秒里就可能已经失效。

这个项目换了个问法：

> 当前屏幕上有这些元素，**该操作哪一个**、**怎么操作**？

手机的无障碍树被压平成一份带编号的元素表，Jev 在一次请求里回答 `operation + target`，
执行器再把这个 target 解析回**当下**的屏幕坐标，然后才动手。如果这中间界面变了，
动作会被拒绝，循环重新观察一遍。

```
  手机屏幕                        编号元素表                       一次 Jev 请求
  ────────                        ──────────                       ─────────────
  get_ui_tree  ──────────►  [1] button   设置
                            [2] button   声音和振动
                            [3] textbox  搜索设置项
                            [4] radio    响铃      checked
                            [5] radio    振动
                            [6] radio    静音                     ┌──────────────────┐
                            [scroll_down] [scroll_up] [wait]  ─────►│ operation        │
                                                                    │ click_target     │
                                                                    │ type_text_target │
                                                                    └────────┬─────────┘
                                                                             │
                                                        CLICK [5] ───────────┴───► tap([5] 的中心)
                                                        TYPE_TEXT [3] "wifi" ───► tap + setText
```

这个设计带来两个性质，也正是它存在的意义：

- **每个动作都锚定在一次观察上。** 模型只能从「它做决策时确实在屏幕上」的目标里挑。
- **每次输入前执行器都会重新校验。** 已经和当前屏幕对不上的决策会抛 `StalePage`，
  而不是点错地方。

## 快速开始

```bash
git clone git@github.com:cnhuye/mobile-jev-ultrafast.git
cd mobile-jev-ultrafast
uv sync
cp .env.example .env      # 填 AUTOX_MCP_URL 和 TYPESAFE_API_KEY
```

然后给它一句话：

```bash
uv run autox-run "打开设置，进入「声音和振动」，把当前铃声模式切换成「响铃」。"
```

```
env:    /path/to/mobile-jev-ultrafast/.env
device: http://192.168.2.7:27190/mcp
goal:   打开设置，进入「声音和振动」，把当前铃声模式切换成「响铃」。

  1     2526 ms     ready      CLICK -> 设置
  2     3791 ms     ready      CLICK -> 声音和振动
  3     5955 ms     ready      CLICK -> 静音
  3 ✓   6622 ms      done      CLICK -> 静音

status: done
steps:  3
ops:    CLICK -> CLICK -> CLICK -> DONE
```

`autox-run` 每一步打一行，跑完退出码 0，卡住或出错退出码非 0，可以直接写进 shell 脚本或 CI。

## 命令行用法

```bash
uv run autox-run "打开设置，把铃声模式切换成振动"        # 单条指令
uv run autox-run -g "Open Settings" -g "Tap About"      # 多步 plan
echo "打开设置，把铃声模式切换成振动" | uv run autox-run  # 从 stdin 读
```

| 参数 | 作用 |
|------|------|
| `--show-elements` | 打印 Jev 实际看到的编号元素表 |
| `-v` / `-vv` / `-vvv` | 详细日志：每步观察/决策/动作 / 再加 Jev+辅助 LLM 完整请求与返回 / 再加元素表与死锁窗口（输出到 stderr） |
| `--ocr` | 打开 OCR fallback，用于微信这类屏蔽无障碍的 App |
| `--max-steps N` | 调整行动预算（默认 60） |
| `--settle S` | 点击后停顿，避免下一步观察到过渡中的旧界面（默认 0.35s） |
| `--record DIR` | 每步截图存盘 |
| `--json` | 最终状态输出 JSON，可以直接接 `jq` |
| `--quiet` | 只打摘要 |

`.env` 会自动从当前目录或任意上层目录加载，`--no-env` 可以关掉。

详细日志走 **stderr**，所以 `--json` 的 stdout 依然是一个干净的 JSON：

```bash
uv run autox-run -v  "手机屏幕上滑1下"           # 每步：观察 → 决策 → 动作 → 是否换屏
uv run autox-run -vv "手机屏幕上滑1下" 2>trace.log  # 再加发给 Jev / 辅助 LLM 的完整请求与返回
```

### 看一次运行到底发生了什么

```bash
uv run autox        # 浏览器 UI，http://127.0.0.1:8767
```

可以一步一步走决策，看元素表怎么变，打开 overlay 还能看到模型当时被提供了哪些矩形。

## 手机端

执行器对接的是 **[cnhuye/AutoX](https://github.com/cnhuye/AutoX)** —— 一个在
AutoX.js v7 上加了 MCP 服务的 fork。它就是个普通 Android App，装到你要操控的手机上。

真正在设备上干活的是它：

- 在 `27190` 端口上通过 JSON-RPC 暴露 `get_ui_tree`、`tap`、`swipe`、`screenshot`、
  `ocr`、`run_script` 等工具
- 通过 Android 无障碍服务派发点击
- 通过 MediaProjection 截屏
- 用 Google ML Kit 在设备本地跑 OCR

### 手机端配置

1. 在手机上装好 `cnhuye/AutoX`，打开**设置 → MCP 服务**启用它。监听地址设成
   `0.0.0.0`（否则电脑连不上），记下端口（默认 `27190`）。
2. 打开**无障碍服务**。不开的话 MCP 的 `tap` 会返回 `ok`，但屏幕上什么都不会动。
3. 授予**截图权限**。第一次调 `screenshot` 会弹 MediaProjection 对话框，点「立即开始」。
   这是一次性授权，**App 重装后需要重新授权**。
4. 把 `.env` 指过去：

   ```bash
   AUTOX_MCP_URL=http://192.168.2.7:27190/mcp
   AUTOX_MCP_TOKEN=          # 只有你在 App 里设了 token 才要填
   ```

在怀疑模型之前，先确认链路是通的：

```bash
uv run python examples/diagnostics.py
```

它会报告工具列表、设备信息、前台 App，并探测截图权限，缺什么会直接告诉你怎么补。

## 环境要求

- Python ≥ 3.12 和 [`uv`](https://github.com/astral-sh/uv)
- 一台装了 [cnhuye/AutoX](https://github.com/cnhuye/AutoX) 并启用了 MCP 服务的 Android 手机
- 一个 [TypeSafe](https://docs.typesafe.ai/introduction) API key（决策模型用）
- 可选：一个 OpenAI 兼容端点给小的文本模型用，只有选中 `TYPE_TEXT` 时才会调用

## 致谢

- [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast) —— 控制循环、
  动态 operation/target 策略、提示词。`agent.py`、`model.py`、`questions.py` 都是直接移植，
  只有设备层不同。
- [`cnhuye/AutoX`](https://github.com/cnhuye/AutoX) —— Android 侧。
- [TypeSafe](https://docs.typesafe.ai/introduction) —— Jev 模型。

## License

MIT
