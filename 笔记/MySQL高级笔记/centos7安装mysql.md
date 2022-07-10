- Linux的一些命令

mysql --help | grep my.cnf可以找出mysql的所有配置文件

其中

/etc/my.cnf 全局选项
/etc/mysql/my.cnf 全局选项
SYSCONFDIR/my.cnf 全局选项
$MYSQL_HOME/my.cnf 服务器特定选项（仅限服务器）
defaults-extra-file 指定的文件 --defaults-extra-file，如果有的话
~/.my.cnf 用户特定选项
~/.mylogin.cnf 用户特定的登录路径选项（仅限客户端）



Rpm -qa | grep xxx 找出rpm中已经安装了的xxx软件包

Rpm -e -nodeps xxx或者 yum -y remove xxx卸载已经安装的安装包

Rpm -ivh xxx --force --nodeps忽略依赖关系安装下载好的rpm包

 

- 安装步骤（安装好后自动初始化；在mysql中set修改全局变量，重启服务后会重新加载默认的值，除非在配置文件中修改）

[(70条消息) MySQL 8.0安装以及初始化错误解决方法_Pioneer4的博客-CSDN博客](https://blog.csdn.net/weixin_40780777/article/details/100553505)



- 一些配置：

系统表user表中的Host列指定了允许用户登录所使用的IP，比如user=root Host=192.168.1.1。这里的意思就是说root用户只能通过192.168.1.1的客户端去访问。而%是个通配符，如果Host=192.168.1.%，那么就表示只要是IP地址前缀为“192.168.1.”的客户端都可以连接。如果Host=%，表示所有IP都有连接权限。需要设置这一项才能用IP进行远程登录。



- 查看配置文件的加载顺序以及添加配置文件到其中

```
mysql --verbose --help|grep -A 1 'Default options'
```

```
服务器首先读取的是/etc/my.cnf文件，如果前一个文件不存在则继续读/etc/mysql/my.cnf文件，如果还不存在依次向后查找。
```

[Linux修改mysql配置文件 - 走看看 (zoukankan.com)](http://t.zoukankan.com/mr-wuxiansheng-p-12091037.html)
