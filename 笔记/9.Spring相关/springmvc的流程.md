DisPatcherServlet是一个Servlet，他拦截请求，最终会调用它的dodiapatch方法();

在里边，首先会从handlermapping中拿到请求处理链，里边包含又handler和拦截器；

然后回调用拦截器的preHandler方法，一个返回false，就会调用afterCompletion()；

然后拿到handler对应的适配器，传入handler，并真正执行方法；

执行前，首先会遍历所有的参数，拿到support这个参数的HandlerMethodArgumentResolver实现类，并解析包装（比如ServletModelAttributeMethodProcessor是用来处理带有@ModelAttribute的，一以及封装非基本类型且不带@RequestParam的javaBean），这个过程也可能会用到类型转换器；

然后传入参数，通过反射执行方法；

获取返回值后，找到合适的HandlerMethodReturnValueHandler处理返回值（比如返回值是String，它会讲name设置到mavContainer里边）；

然后将Model和View包装到ModelAndView，handler执行完返回的是modelAndView；

然后调用拦截器的postHandle()方法；

在catch块后调用processDispatchResult，

在render前，会判断是否产生异常并调用异常处理器进行处理，调用实现了HandlerExceptionResolver接口的实现类进行处理，将返回值作为mv；

在render中，它会调用视图解析器（InternalResourceViewResolver对应的是JSP），根据viewname是否以redirect开头，是否以forward开头，解析到不同的view对象，然后调用视图对象的渲染，渲染中它会讲model里边的数据逐个调用request,setAttribute设置，然后根据view获取到转发路径(或者重定向路径)进行转发或重定向；

最后执行拦截器的afterCompletion方法



在上面方法执行过程中，如果有抛错误并被doDispater方法catch到，它也会执行拦截器的afterCompletion方法；



##### 一些组件：

（1）前端控制器DispatcherServlet（不需要程序员开发）：接收请求、响应结果，相当于转发器，有

了DispatcherServlet就减少了其它组件之间的耦合度。

（2）处理器映射器HandlerMapping（不需要程序员开发）：根据请求的url来查找Handler

（3）处理器适配器HandlerAdapter：在编写Handler的时候要按照HandlerAdapter要求的规则去编

写，这样适配器HandlerAdapter才可以正确的去执行Handler。

（4）处理器Handler（需要程序员开发）

（5）视图解析器ViewResolver（不需要程序员开发）：进行视图的解析，根据视图逻辑名解析成真正

的视图。

（6）视图View（不需要程序员开发jsp）：View是一个接口，它的实现类支持不同的视图类型





**Servlet** **的生命周期** 

1、执行 Servlet 构造器方法 

2、执行 init 初始化方法 

第一、二步，是在第一次访问，的时候创建 Servlet 程序会调用。 

3、执行 service 方法 

第三步，每次访问都会调用。 

4、执行 destroy 销毁方法 

第四步，在 web 工程停止的时候调用。



##### Filter 的生命周期包含几个方法 

1、构造器方法 

2、init 初始化方法 

第 1，2 步，在 web 工程启动的时候执行（Filter 已经创建） 

3、doFilter 过滤方法 

第 3 步，每次拦截到请求，就会执行 

4、destroy 销毁 

第 4 步，停止 web 工程的时候，就会执行（停止 web 工程，也会销毁 Filter 过滤器）



##### Listener也是在web工程启动时创建，停止时销毁

