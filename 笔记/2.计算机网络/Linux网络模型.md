### 最基本的socket模型：

服务端⾸先调⽤ socket() 函数，创建⽹络协议为 IPv4，以及传输协议为 TCP 的 Socket ，接着调⽤
bind() 函数，给这个 Socket 绑定⼀个 IP 地址和端⼝，绑定这两个的⽬的是什么？
		绑定端⼝的⽬的：当内核收到 TCP 报⽂，通过 TCP 头⾥⾯的端⼝号，来找到我们的应⽤程序，然后
		把数据传递给我们。
		绑定 IP 地址的⽬的：⼀台机器是可以有多个⽹卡的，每个⽹卡都有对应的 IP 地址，当绑定⼀个⽹卡
		时，内核在收到该⽹卡上的包，才会发给我们；

基于 Linux ⼀切皆⽂件的理念，在内核中 Socket 也是以「⽂件」的形式存在的，也是有对应的⽂件

描述符。

绑定完以后调用listen()进行监听，然后通过accept()函数从内核获取已连接Socket，获取不到则阻塞。

连接过程中为每个Socket维护两个队列，一个TCP半连接队列，这个队列都是没有完成三次握⼿的连接，

此时服务端处于 syn_rcvd 的状态，一个TCP全连接队列，这个队列都是完成了三次握⼿的连接，此时服

务端处于 established 状态。

这种模型是一个服务端对应一个客户端的，使用的是同步阻塞的方式。

### 多进程模型：

为每个客户端分配⼀个进程来处理请求；

服务器的主进程负责监听客户的连接，⼀旦与客户端连接完成，accept() 函数就会返回⼀个已连接

Socket，这时就通过 fork() 函数创建⼀个⼦进程，实际上就把⽗进程所有相关的东⻄都复制⼀份，包

括⽂件描述符、内存地址空间、程序计数器、执⾏的代码等，因此可以直接使用已连接Socket与客户端通信。

### 多线程模型

当服务器与客户端 TCP 完成连接后，创建线程，然后将已连接 Socket的⽂件描述符传递给线程函数，接着在线程⾥和客户端进⾏通信，从⽽达到并发处理的⽬的。

由于频繁地创建和销毁线程的开销大，因此可以使用线程池的方法复用创建好的线程。

### I/O 多路复⽤

一个进程维护多个Socket，多个请求复用一个进程。利用系统调用获取连接事件。

1、select:

select 实现多路复⽤的⽅式是，将已连接的 Socket 都放到⼀个⽂件描述符集合，然后调⽤ select 函数将

⽂件描述符集合拷⻉到内核⾥，让内核来检查是否有⽹络事件产⽣，检查的⽅式很粗暴，就是通过遍历⽂

件描述符集合的⽅式，当检查到有事件产⽣后，将此 Socket 标记为可读或可写， 接着再把整个⽂件描述

符集合拷⻉回⽤户态⾥，然后⽤户态还需要再通过遍历的⽅法找到可读或可写的 Socket，然后再对其处

理。

由于使用固定的BitsMap表示文件描述符集合，因此所监听的Socket是有限的。

2、poll

使用链表形式组织，突破了select 的⽂件描述符个数限制，当然还会受到系统⽂件描述符限制。

3、epoll

第⼀点，epoll 在内核⾥使⽤红⿊树来跟踪进程所有待检测的⽂件描述字，把需要监控的 socket 通过

epoll_ctl() 函数加⼊内核中的红⿊树⾥，红⿊树是个⾼效的数据结构，增删查⼀般时间复杂度是

O(logn) ，通过对这棵⿊红树进⾏操作，这样就不需要像 select/poll 每次操作时都传⼊整个 socket 集

合，只需要传⼊⼀个待检测的 socket。

第⼆点， epoll 使⽤事件驱动的机制，内核⾥维护了⼀个链表来记录就绪事件，当某个 socket 有事件发⽣

时，通过回调函数内核会将其加⼊到这个就绪事件列表中，当⽤户调⽤ epoll_wait() 函数时，会通过传出型参数返回给用户发生的事件，而且不需要像 select/poll 那样轮询扫描整个 socket 集合。

​		其解决了C10K问题

[Linux I/O复用中select poll epoll模型的介绍及其优缺点的比较 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/141447239)

epoll ⽀持两种事件触发模式，分别是边缘触发和⽔平触发。

使⽤边缘触发模式时，当被监控的 Socket 描述符上有可读事件发⽣时，服务器端只会从 epoll_wait
中苏醒⼀次，即使进程没有调⽤ read 函数从内核读取数据，也依然只苏醒⼀次，因此我们程序要保
证⼀次性将内核缓冲区的数据读取完；
使⽤⽔平触发模式时，当被监控的 Socket 上有可读事件发⽣时，服务器端不断地从 epoll_wait 中苏
醒，直到内核缓冲区数据被 read 函数读完才结束，⽬的是告诉我们有数据需要读取；

如果使⽤边缘触发模式，I/O 事件发⽣时只会通知⼀次，⽽且我们不知道到底能读写多少数据，所以在收到
通知后应尽可能地读写数据，以免错失读写的机会。因此，我们会循环从⽂件描述符读写数据，那么如果
⽂件描述符是阻塞的，没有数据可读写时，进程会阻塞在读写函数那⾥，程序就没办法继续往下执⾏。所
以，边缘触发模式⼀般和⾮阻塞 I/O 搭配使⽤，程序会⼀直执⾏ I/O 操作，直到系统调⽤（如 read 和
write ）返回错误，错误类型为 EAGAIN 或 EWOULDBLOCK 。