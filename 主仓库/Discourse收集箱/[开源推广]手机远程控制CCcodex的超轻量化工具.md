---
来源: https://linux.do/t/topic/1810124
标题: "[开源推广]手机远程控制CC/codex的超轻量化工具"
作者: ZgDaniel
分类: 开发调优
tags:
  - discourse
  - 人工智能
  - 软件开发
  - 开源推广
  - 抽奖
  - 公告
  - 集中帖
  - 精华神帖
  - 快问快答
  - 原创
保存时间: 2026-04-14 20:35:13
评论数: 0
---

# [开源推广]手机远程控制CC/codex的超轻量化工具

推广格式

#### （整改后重发）本帖使用社区开源推广，符合推广要求。我申明并遵循社区要求的以下内容：

*   **我的帖子已经打上 [开源推广](/tag/2234-tag/2234) 标签：** 是
    
*   **我的开源项目完整开源，无未开源部分：** 是
    
*   **我的开源项目已链接认可 LINUX DO 社区：** 是  
    叠甲如图  
    
    ![dKNwGafKTzEsteieXGTgE7J8HGX](https://cdn3.linux.do/original/4X/6/0/6/60668a1ecc640ac7f3b11c2e6bc8a2c636c04ec3.png)
    
*   **我帖子内的项目介绍，AI生成、润色内容部分已截图发出：** 是  
    叠甲：全部手打
    
*   **以上选择我承诺是永久有效的，接受社区和佬友监督：** 是
    

* * *

04-13-重磅更新：VPS管理和github推送功能

**可以添加SSH配置，通过新建远程会话、/ssh斜杠命令来管理远程主机**

**也可以通过/github命令来提交、修改仓库**

![bBMPod0uZ4Hdv6kA7fzDdYzMB7n](https://cdn3.linux.do/original/4X/5/1/5/515d6bccdd27d1950e160373b9973acbdd0fdf4d.png)

  

![kKpEjl2uT5MJT0ZDIsEkfMz2RwX](https://cdn3.linux.do/original/4X/9/1/6/916a9ce3b94fecc276cf79f05acbc0a2730cefaf.png)

*以下为项目介绍正文内容，AI生成、润色内容已使用截图方式发出*

* * *

cc-web：超轻量化的远程拷打cc/codex工具，手机vibe友好，服务器部署极佳

目前主流解决方案（happy/Termux)总是有些地方不太满意，自己也用了一段时间，特别是是happy，对我捉襟见肘的服务器空间压力太大，加上春节期间happy的几次故障，和claude一拍即合，怒而Vibe了这么个小玩意

![2jZlm7FZFDZdxn3V9Uk5TCsshpA](https://cdn3.linux.do/original/4X/1/0/4/1046cec9b822d8809ac83e09a90be7833a4fe8d6.jpeg)

![d67DOOGiPEjYzT3dhN1clZyQ1lN](https://cdn3.linux.do/original/4X/5/b/c/5bcd755519281bec4fe722e1258f65557664f76b.jpeg)

[github.com](https://github.com/ZgDaniel/cc-web)

![image](https://cdn3.linux.do/optimized/4X/4/8/8/48801251ef9c3baba2bb63d284f612bfa54ab8b9_2_690x344.png)

### [GitHub - ZgDaniel/cc-web: cc-web是通过浏览器使用cc的远程工具，主要是超轻量的设计、后台进程保活、交互界面优化，对于捉襟...](https://github.com/ZgDaniel/cc-web)

cc-web是通过浏览器使用cc的远程工具，主要是超轻量的设计、后台进程保活、交互界面优化，对于捉襟见肘的服务器空间有很好的帮助。支持linux和windows。

1.  核心对比  
    磁盘占用：happy: 5gb+  cc-web: 2mb  
    稳定性：Termux：熄屏必断连  cc-web: 浏览器访问、无中断
    
2.  功能特点  
    不断连：基于–resume的对话续接，后台进程不断
    

API切换：类似cc switch的模板切换功能，cc/codex均可配置并切换

图片上传：字面意思

会话管理：完全与CLI对应，在项目下删除的会话，CLI历史也会同步删除；也支持会话导入，在终端另外展开的对话，可以导入进行继续对话

后台通知：任务在后台完成后，可以通过通知机器人告知，目前支持飞书/telegram/Qmsg等（部分是小克给我找的，总之能用）；增加了AI总结，可以将长任务整理成简短的报告来通知。

主题切换：为了满足初代用户的执着要求，设置三类主题，本着护眼舒适的原则，没有加入过于丰富的元素。有美化需求的佬友可以自行拷打。

更多惊喜欢迎探（kao）索（da）

![apbH3iFdlbfnU4yCuxVO3FBRFm4](https://cdn3.linux.do/original/4X/4/8/e/48eea7bf1237b0ead1e5221eda40c2edff338a44.jpeg)

![rHNuK6wSzull9JEbO5zrGZrYnOJ](https://cdn3.linux.do/original/4X/c/2/2/c22e254948159e3fa7e6513c41307906320853d5.jpeg)

3.  安装说明

推荐使用一键方式：丢链接——给我装

注意事项：  
目前仅支持非root用户，root账户部署可能无法使用  
支持linux/windows，Mac由于本人没有对应环境，需要自行拷打

真的挺好使  
初代用户最后都真香了（纷纷改成了自己的形状）  

![fwgKDTgzxcRVxfWeY4vm5zPlGRW](https://cdn3.linux.do/original/4X/6/c/c/6cc6743252e92f84f89cd9a4a380c27435004220.png)