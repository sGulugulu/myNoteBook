---
来源: https://linux.do/t/topic/1973236
标题: "【教程】免费自建节点搭建！无需服务器，从此告别机场，支持访问openai/gemini/claude"
作者: Victor ferline
分类: 开发调优
tags:
  - discourse
  - 人工智能
  - 订阅节点
  - 机场
  - 抽奖
  - 公告
  - 集中帖
  - 精华神帖
  - 快问快答
  - 原创
保存时间: 2026-04-24 03:28:27
评论数: 0
---

# 【教程】免费自建节点搭建！无需服务器，从此告别机场，支持访问openai/gemini/claude

先上效果图：

![tAjfin65Ada74D7ZzUlLlqTbOad](https://cdn3.ldstatic.com/original/4X/c/f/5/cf59ce8f0c0c3a97666be352cb11cd8bb4eda8d9.jpeg)

只需两样：

1.  一个 cf 账号；
2.  去DNSHE 申请个免费域名；

tip:

1.  整个过程无需开启代理.
2.  建议cf 小号注册，谨防封号风险

步骤：

1.  登录[DNSHE](https://www.dnshe.com/),注册登录后左边菜单有个免费域名点进去。新注册的有 5 个域名额度，每邀请一人双方各增加一个额度，上限 5 个（可以互邀，我的邀请码：YYB3061B0A）
    
    现在 **cc.cd** 和 **ccwu.cc**是直接可以创建的，可以加到 cf 里，剩下两个需要邀请码（可以自己注册两个号，互相邀请），但是现在还不能加到 cf。
    
2.  创建好一个域名以后去 cf 里加入域，没 cf 的先去[cloudflare](https://www.cloudflare.com/zh-cn/)注册一个，托管你的域名。
    
3.  在 cf 里创建 kv 空间，存储和数据库 – Worker KV，名称随意。
    
4.  创建 pages，在计算 - Workers 和 Pages，创建应用程序，进去后点创建 pages，选拖放文件，项目名称随意，上传pages 演示安装包 [direct-upload-demo.zip](/uploads/short-url/k8RXmE6P6obhhK9JVWfC6pN1fxy.zip) (25.4 KB)，部署站点。
    
    完成后，点继续处理站点，上面 设置 - 变量和机密，添加 类型 文本， 名称填 ADMIN ,值设置一个你自己的登录密码，确定。
    
    下面绑定里， 添加， 选择 kv命名空间， 变量名称填 KV，空间选刚才你建的那个命名空间名称，保存。
    
    上面点击自定义域菜单，设置自定义域，把刚才的域名填进去，点继续。看到一个新的 DNS 记录，新开一个页面把新 DNS 记录里把内容添加到你那个域名的 DNS 记录中，完事以后点激活域。在这个页面等几分钟，等从初始化变成活动，再点右上方的创建部署。
    
    进入上传页面，环境选生产，上传项目安装包  
    [edgetunnel-main.zip](/uploads/short-url/3XrGU1DT1QHBmqwXfTlOIuJM8OQ.zip) (289.8 KB)，点保存并部署。
    
    点继续处理项目，看到生产的域名，点进去，看到  
    
    ![d4D8jh5rJ7g3uwd797VgaHselUw](https://cdn3.ldstatic.com/original/4X/5/b/a/5ba247cad0542d4a3f55f78e3d9020cec1246c74.png)
    
      
    说明部署成功。
    
5.  在网站后缀加/admin，弹出让输入密码，就是刚才设置的密码，进去
    

![azMmHg4nUl4BQElqbca8NdBOwbD](https://cdn3.ldstatic.com/original/4X/4/a/2/4a21279014c2cdd4332318c65304a610290f8ab1.jpeg)

到这节点就设置完成了，直接复制拿去用。

如果想要更多节点，像我那样，可以下面选自定义订阅，点在线优选，把里面的每个列表选项都试一下，第一次优选完选覆盖保存，后面的优选完选追加保存。最后选完再点外面的保存，去你的 clash 里更新一下订阅就行了，v2rayn 的重新复制下节点。

ok，完美收工！

最后测试了下速度还都可以，节点 ip 纯净度大多三四十，少数也有优质的，都是正常使用的范围，感觉比什么一毛机场的强多了。优选出来的节点几乎没有 timeout 的，可作为日常使用。

懂一些的还可以再点击上面的我是高手，下面会出来一系列菜单自定义设置，可以打开ECH,保证稳一点，调下cdn 访问设置，设置其他反代，订阅模板多种选择，大家看情况设置即可。