---
来源: https://linux.do/t/topic/1962519
标题: "GenericAgent——复旦团队研发 | 仅仅~3K 行代码 Self-Evolving Agent"
作者: ozer_23
分类: 开发调优
tags:
  - discourse
  - 人工智能
  - 开源推广
  - 抽奖
  - 公告
  - 集中帖
  - 精华神帖
  - 快问快答
  - 原创
保存时间: 2026-04-14 20:35:53
评论数: 0
---

# GenericAgent——复旦团队研发 | 仅仅~3K 行代码 Self-Evolving Agent

#### 本帖使用社区开源推广，符合推广要求。我申明并遵循社区要求的以下内容：

*   **我的帖子已经打上 [开源推广](/tag/2234-tag/2234) 标签：** 是
*   **我的开源项目完整开源，无未开源部分：** 是
*   **我的开源项目已链接认可 LINUX DO 社区：** 是
*   **我帖子内的项目介绍，AI生成、润色内容部分已截图发出：** 是
*   **以上选择我承诺是永久有效的，接受社区和佬友监督：** 是

*以下为项目介绍正文内容，AI生成、润色内容已使用截图方式发出*

* * *

回应一下预告: [成功开通 4 个 Claude Code Max 5x](https://linux.do/t/topic/1942024)

项目地址：

![image](https://cdn3.linux.do/original/4X/b/a/d/bad3e5f9ad67c1ddf145107ce7032ac1d7b22563.svg)

 [github.com](https://github.com/lsdefine/GenericAgent/tree/main)

### [GitHub - lsdefine/GenericAgent: Self-evolving agent: grows skill tree from...](https://github.com/lsdefine/GenericAgent/tree/main)

[main](https://github.com/lsdefine/GenericAgent/tree/main)

Self-evolving agent: grows skill tree from 3.3K-line seed, achieving full system control with 6x less token consumption - lsdefine/GenericAgent

# 长程任务拳打OpenClaw，Token消耗脚踢Claude Code

hahahaha，目前我们实验组日常就是使用GenericAgent的。贴个最显眼的结果放在开头！！！！  

![8YdadadPj33xwKfkwL2ZGTAePSn](https://cdn3.linux.do/original/4X/3/e/d/3edfd56f193eb898200f727a27f7981d3439cbe7.png)

GA VS Openclaw & Claude Code区别:

1.  仅仅 9 个原子工具，对比 CC 53个内置工具，Openclaw 22个内置工具。GA有更准确的工具调用，高效完成原子工具的MVP  
    
    ![dUseyIDtLLRzw0h2gZ8B1Rb02kE](https://cdn3.linux.do/original/4X/6/1/7/617dfbfdc0e8a54bd66234149582a8a97909e4d8.png)
    
      
    
    ![dcKaoCnMp2L891nHfy970YJpYK2](https://cdn3.linux.do/original/4X/5/c/8/5c8d11d719f81a3e77e9a203bd389fa48be1031e.png)
    
2.  四层分级记忆 | 真的是token开销的杀手锏hhh，具体实现可以代码中看。
    

> 举个例子: “Hello”，GA 的 prompt 长度是 2298 tokens，Claude Code 是 22821，OpenClaw 是 43321。

3.  反思驱动自进化：任务完成后会自动蒸馏成一个SOP | 这个实现起来不难
    
4.  结构化浏览器提取 | GA最大的杀招！！！！！  
    **拳打Playwright, 目前市面上最好的Agent Browser工具。Token开销极大程度降低的同时，精度极高!!! 后续会将该功能上线到skills.sh中**
    

> WebCanvas 比 OpenClaw 高 11.2 分，token 用量是它的 1/4。

**PS: 其实可以偷偷把Web关键代码蒸馏给Claude Code用的，也很丝滑hahaha。**

*   Chrome插件：GenericAgent/tree/main/assets/tmwd_cdp_bridge
*   SOP指南：GenericAgent/blob/main/memory/tmwebdriver_sop.md
*   SOP指南：GenericAgent/blob/main/memory/ljqCtrl_sop.md

建议赶紧删了所有skills.sh上的Agent Browser操作，快快安装这个，非常有用！！！

再贴一点打榜记录吧，其实打榜不重要，更重要的是实际体验：

![o437LO1hdU6iWbI965KdSRr3L0T](https://cdn3.linux.do/original/4X/a/8/a/a8a9631c8c7d9c86bdcfdeb3f4cfb215a84b1ceb.png)

## Tips | 这里才是给L站看的内容:

> 目前GenericAgent可以过**CRS的客户端检测**，小批量测试过Claude Code的CRS反代  
> \[OK​但还是不建议大家用，Claude Code检测一直在变，我们虽然模拟了客户端，对System Prompt做了处理，解析了Claude Code源码，但不能保证100%  
> 关于测试：我们是测试了某闲鱼上的crs Claude Code Max中转， 50刀成功用完，没有出现报错\]。  
> 踩坑：GA不会像openclaw那样傻傻的秒封(虽然前期测试过CRS的时候，报废了好几个Claude Code Max 20x账号，真的就是输入一个hello就被秒封)

> 今天已经完成Anyrouter大善人适配，今天很给力啊！！

> 支持最新的ampere.sh渠道，目前v1/openrouter会有OAI缓存不足的问题，GA中也进行了适配hhh，努力适配L站看到了所有渠道。哭泣，还被Ampere.sh坑了200+，不过fast真的好快。

> **缓存优化**: GA适配了OAI和Claude Code两种接口的缓存优化策略，目前OAI调用claude模型会出现缓存命中=0的问题，GA也进行了实现。如果之后大家需要从5m缓存到1d缓存，我也push老师去弄去哈哈哈哈

> 内置JS逆向skills，Codex破限skills，打野skills，Warp rotate Skills，无限Freemail Skills\[绕过api_key\]，无限socks skills，过CF盾skills。

但是上述skills正式仓库中并不会包含…后续可能L站会发焚决.zip。\[其实还有注册机和2api skills hhh\]

哎，写到凌晨4.17，希望各位大佬明早看到能支持一下吧…

我要摘取你们的小星星和大拇指！！！

对啦，马上会有一个新版本，加了一个小功能，预告一下哈哈哈，可以猜猜是什么：

![AgGdkBl7VEg8fksuaFTcSaCzk9z](https://cdn3.linux.do/original/4X/f/e/3/fe30faafa1dbfe46d2b18316ec46384b993b9429.png)

答案是: Destop Pet hhh，作为作者的小爱好。

后续的预告：MultiAgent Manager Web Service

> 之后也会上线OpenClaw，Claude Code，GA等多任务多窗口守护进程监控，帮你们实时知道你的GA什么时候完成了某个任务,进行桌面或者桌宠的通知！！！！

> 你无需额外安装Manager的各种插件，我们的GA原生支持！