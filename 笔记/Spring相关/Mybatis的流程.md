（创建SQLSessionFactory时，同时解析配置文件以及mapper.xml到Configuration里边，在其中包含多个mappedStatement，每一个代表一个增删改查标签。

调用openSession后，会返回一个DefaultSqlSession对象，里边包含了用Configuration和用拦截器链包装的Executor

调用DefaultSqlSession的getMapper后，会返回mapper的代理对象。）

（X）



##### 执行流程

调用mapper代理对象的查询，会调用mapperProxy（一个invokationHandler）的invoke，它主要是调用了MapperMethod的excute方法，首先包装目标方法的参数到一个map中，然后根据增删改查，调用sqlSession的方法，

然后根据key获取到ms，传入excutor中执行，

在开启二级缓存的时候，cacheExcutor首先会查二级缓存，没有就查本地缓存，再没有，就doQuery，

首先创建prepareStatementHandler，用拦截器链包装，同时创建parameterHandler，用拦截器链包装，以及创建ResultSetHandler，用拦截器包装，

然后用prepareStatementHandler进行sql预编译，ParameterHandler设置参数（期间会调用TypeHandler处理javaBean类型与数据库类型的映射），执行后，由ResultSetHandler处理结果（也是调用TypeHandler进行类型转换）。

最后将结果放进一级缓存，然后返回。



##### 插件原理

[(96条消息) 超详细的 Mybatis 插件开发指南！_程序员小乐-CSDN博客](https://blog.csdn.net/xiaoxiaole0313/article/details/106184455)

遍历每个拦截器调用plugin的时候，实际上是调用了Plugin.wrap把拦截器自己和目标对象传了进去，而Plugin继承了invokationHandler，它的wrap实际上是通过JDK的动态代理，以Plugin对象为invokationHandler传入了拦截器自己和目标对象和需要拦截的方法Map，最后wrap返回了代理对象。每个拦截器把目标对象一层层地包了起来。

调用目标方法的时候，会调用到Plugin的invoke，首先通过需要拦截的方法Map判断当前方法是不是要拦截的，然后调用拦截器的intercept方法传入一个invocation（传入了目标对象），执行完拦截器逻辑后，会执行invocation的procceed。



##### 缓存

一级缓存(local cache), 即本地缓存, 作用域默认为sqlSession，当执行更新操作或手动清除缓存时，一级缓存会被清空，但是一级缓存还能用。

二级缓存需要缓存对象实现Serializable接口，同时在配置文件中开启，开启后使用CachingExecutor，里边持有一个Executor，查前首先会查二级缓存，没查到再调用Executor去查，最后结果放入二级缓存。二级缓存是NameSpace级别的，相当于一个全局变量；当在namespace下进行更新操作，对应的二级缓存会失效



##### 当实体类中的属性名和表中的字段名不一样 ，怎么办 ？

1.sql语句中起别名

2.使用resultMap



##### 分页

Mybatis使用RowBounds对象进行分页，它是针对ResultSet结果集执行的内存分页，而非物理分页。

可以使用PageHelper分页插件，它利用了Mybatis提供的插件接口，拦截了待执行的sql，并重写sql，添加了分页语句以及分页参数，是物理分页；



##### 延迟加载的原理

Mybatis支持association关联对象和collection关联集合对象的延迟加载，association指的就是一对一，collection指的就是一对多查询。

它的原理是，使用CGLIB创建目标对象的代理对象，当调用目标方法时，进入拦截器方法，比如调用a.getB().getName()，拦截器invoke()方法发现a.getB()是null值，那么就会单独发送事先保存好的查询关联B对象的sql，把B查询上来，然后调用a.setB(b)，于是a的对象b属性就有值了，接着完成a.getB().getName()方法的调用。这就是延迟加载的基本原理。



##### #{}和${}的区别是什么？

#{}是预编译处理,${}是字符串替换

Mybatis在预编译#{}时会将sql中的#{}替换为？号，调用PreparedStatement的set方法来赋值。

Mybatis在动态解析sql，遇到${}时，就是把${}替换成变量的值，编译前这个变量已经被替换为常量。

使用#{}可以有效的防止SQL注入，提高系统安全性







##### 什么时候不适用#{}

由于#{}会给参数内容自动加上引号，会在有些需要表示字段名、表名的场景下，SQL将无法正常执行。



##### Mybatis的XML映射文件中，不同的XML映射文件，id是否可以重复？

不同的XML映射文件，如果配置了namespace，那么id可以重复，如果没有配置namespace，那么id

不能重复。

原因是namespace+id是作为map的key使用的，如果没有namespace，就剩下id，那么，id重复会导

致数据互相覆盖。有了namespace，自然id就可以重复，namespace不同，namespace+id自然也就

不同。
