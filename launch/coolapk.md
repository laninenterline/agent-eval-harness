# 酷安文案（安卓 / Termux 角度）

> 定位：不是"AI 评测工具"，而是"用安卓手机完整发布一个开源项目"。
> 酷安受众对"手机能干什么"的兴趣远大于 AI 工程细节。

## 标题（选一个）

1. 只用一台安卓手机，我把开源项目发到了 PyPI、npm、Homebrew 和 Docker
2. 在 Termux 里写代码，然后发布到 5 个包管理器
3. 安卓上折腾了三个月，最后把项目发上了 PyPI

## 正文

手机上写代码这事，很多人试过，但大多停在"能跑个 hello world"。

我这次试的是完整走一遍：**从写代码、跑测试、打 tag，到发布到各大包管理器，全程不碰电脑。**

项目是个 AI Agent 的评测工具（简单说：给 AI 助手的每次运行做"体检"，跑挂了能自动报警）。这部分不重要，重要的是发布链路。

**最终发出去的渠道：**

- PyPI（Python 官方包源）—— `pip install aehcli`
- npm（Node 包源）—— `npx aehcli`
- Homebrew（Mac 包管理器）—— 自己建了个 tap
- Docker（GitHub 容器仓库）—— GitHub Actions 自动构建
- conda-forge —— PR 还在审核

**踩的坑，都是真实教训：**

1. **手机上的 /tmp 是只读的**，脚本写临时文件直接失败，得换到 home 目录
2. **安卓没有 brew**，Homebrew 的 formula 只能盲写，没法本地验证
3. **npm 开了 2FA**，前两个 token 都被 403 拒绝，最后查 token 元数据才发现要 `bypass_2fa: true`
4. **PyPI 有包名相似性保护**，`agent-eval-kit` 和 `traceeval` 都被判定"与现有包太像"，最后改叫 `aehcli` 才过
5. **Termux 和 proot 的 /tmp 不是同一个**，跨环境传文件踩了半天

**几个我觉得挺酷的点：**

- 所有 CI（测试 + 评测 + 回归门禁）都跑在 GitHub Actions 上，手机只负责写和推
- Docker 镜像也是 Actions 自动构建的，本地一行 docker 命令都没跑过
- 项目现在 29 个单元测试，全绿

**链接**

- 项目：https://github.com/laninenterline/agent-eval-harness
- 写的技术文章：https://laninenterline.github.io/agent-eval-harness/
- 安装：`pip install aehcli && aeh demo`

（酷安对引流比较敏感，链接建议放评论区第一条，正文里只留一句"感兴趣的评论区"）

## 配图建议（3-4 张，按这个顺序）

1. **`aeh demo` 的终端输出** —— 有 PASS/FAIL 表格，视觉上最直观
2. **PyPI 页面截图** —— `aehcli 0.1.1`，证明真的发布了
3. **GitHub Actions 全绿的截图** —— 证明 CI 是真的
4. **npm 包页面** —— 增加"多平台"的说服力

## 标签

`#Termux` `#安卓开发` `#开源` `#Python` `#折腾`
