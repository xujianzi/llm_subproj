3 h 学习完harness

3 h 做完项目
 - 重构 viz 
 - 使用agent 交互


archtecture
```bash
                    THE AGENT PATTERN
                    =================

    User --> messages[] --> LLM --> response
                                      |
                            stop_reason == "tool_use"?
                           /                          \
                         yes                           no
                          |                             |
                    execute tools                    return text
                    append results
                    loop back -----------------> messages[]


    这是最小循环。每个 AI Agent 都需要这个循环。
    模型决定何时调用工具、何时停止。
    代码只是执行模型的要求。
    本仓库教你构建围绕这个循环的一切 --
    让 agent 在特定领域高效工作的 harness。
```



s01   "One loop & Bash is all you need" — 一个工具 + 一个循环 = 一个智能体

s02   "加一个工具, 只加一个 handler" — 循环不用动, 新工具注册进 dispatch map 就行

s03   "没有计划的 agent 走哪算哪" — 先列步骤再动手, 完成率翻倍

s04   "大任务拆小, 每个小任务干净的上下文" — 子智能体用独立 messages[], 不污染主对话

s05   "用到什么知识, 临时加载什么知识" — 通过 tool_result 注入, 不塞 system prompt

s06   "上下文总会满, 要有办法腾地方" — 三层压缩策略, 换来无限会话

s07   "大目标要拆成小任务, 排好序, 记在磁盘上" — 文件持久化的任务图, 为多 agent 协作打基础

s08   "慢操作丢后台, agent 继续想下一步" — 后台线程跑命令, 完成后注入通知

s09   "任务太大一个人干不完, 要能分给队友" — 持久化队友 + 异步邮箱

s10   "队友之间要有统一的沟通规矩" — 一个 request-response 模式驱动所有协商

s11   "队友自己看看板, 有活就认领" — 不需要领导逐个分配, 自组织

s12   "各干各的目录, 互不干扰" — 任务管目标, worktree 管目录, 按 ID 绑定

