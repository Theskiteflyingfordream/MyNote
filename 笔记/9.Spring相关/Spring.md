# 无参构造方法

继承了GenricApplicationContext，在这个父类的构造方法里，实例化了一个DefaultListableBeanFactory

然后会注册一些必要的后置处理器比如ConfigurationClassPostProcessor和EventListenerMethodProcessor和AutowiredAnnotationBeanPostProcessor，CommonAnnotationBeanPostProcessor



# Register方法

注册配置类



## Spring refresh()的整个流程

### prepareRefresh

清理缓存；initPropertySources()为protected，给子类提供个性化的属性设置；获取其环境变量，然后校验属性的合法性；初始化一个LinkedHashSet（允许收集早期的容器事件，等待事件派发器可用之后，即可进行发布）

### obtainFreshBeanFactory()

创建并返回BeanFactory对象，并为其设置一个序列化id（实际上由于容器为GenericApplicationContext，调用无参构造器就创建了一个DefaultListableBeanFactory）

## prepareBeanFactory(beanFactory)：

BeanFactory的预准备工作，即对BeanFactory进性一些预处理（比如设置beanfactroy的类加载器等）

## postProcessBeanFactory(beanFactory)：

BeanFactory准备工作完成后进行的后置处理工作（这是一个protected方法，留给子类扩展）

### invokeBeanFactoryPostProcessors

执行了BeanDefinitionRegistryPostProcessor的postProcessBeanDefinitionRegistry和postProcessBeanFactory这俩方法，以及BeanFactoryPostProcessors的postProcessBeanFactory方法。

比如（执行ConfigurationClassPostProcessor，解析配置类，并添加配置类指定的类的beanDefination；同时使用CGLIB增强了配置类（注入了ImportAwareBeanPostProcessor处理器））

### registerBeanPostProcessors

拿出所有BeanPostProcessor实现类的name，遍历将他们按优先级分成四类，最高优先级的一类实现了Priorderd接口，然后是实现了orderd接口的一类，上面两者都会进行排序，然后是两者都没有实现的，最后是MergedBeanDefinitionPostProcessor这种类型的。（然后按优先级注册，最后还会注册一个ApplicationListenerDetector类型的BeanPostProcessor，在bean创建后，探测是否是ApplicationListener，是的话就放到一个地方保存（上面说的注册也就是创建对应的bean，并放到容器的一个地方保存起来））

### initMessageSource

初始化MessageSource组件，判断ioc是否有id为messageSource，有则赋值给this.messageSource，否则自己创建一个

### initApplicationEventMulticaster

初始化事件派发器，id为applicationEventMulticaster

### onRefresh()

protected方法，子类可以重写，给容器多注册一些组件

### registerListeners()

从容器中拿到所有的监听器，然后把这些监听器的id添加到事件派发器中。

### ！！！Bean的生命周期：

### finishBeanFactoryInitialization（初始化剩下的单实例bean）

preInstantiateSingletons()中

会获取到每一个遍历出来的bean的定义注册信息

根据bean的定义注册信息判断bean是否是抽象的、单实例的、懒加载的；

如果都不是，然后判断是不是FactoryBean，是的话会调用它的getObject()方法获取bean

否则进入getBean()的doGetBean()中

​	首先调用getSingleton()从concurrentMap缓存中获取中bean，依次从一级-二级-三级中拿，三级拿到就往二级放

​	获取不到则首先获取父容器，能获取到并且当前容器中没有对应的bean定义，就从父容器中拿

​	（否则，先标记bean已经被创建，防止多个线程同时来创建同一个bean，保证bean的单实例特性）

​	否则，如果是单实例，则调用getSingleton()，传入beanName和一个匿名内部类beanFactory(其重写了getObject方法,主体为createBean())

​		在createBean()中

​		会先调用resolveBeforeInstantiation尝试返回bean（给InstantiationAwareBeanPostProcessor一个机会返回bean）

​		返回为空则调用doCreateBean()，在这个方法中：

​			1.在其中，首先创建出bean实例，然后将创建的实例放入三级缓存中。

​			2.然后遍历获取到的所有后置处理器，若是MergedBeanDefinitionPostProcessor这种类型，则调用其

​			postProcessMergedBeanDefinition方法，

​			3.然后调用populate()方法为bean实例的属性赋值，在其中会再来遍历获取到的所有后置处理器，若是

​			InstantiationAwareBeanPostProcessor这种类型，则调用其postProcessAfterInstantiation方法以及

​			postProcessPropertyValues方法，它的后面一个方法会从容器中拿这个bean，然后在makeAccesible之后，用field的set反射设			置进去；

​			4.populate()方法完了以后，会调用初始化bean的方法，在其中会首先调用invokeAwareMethods方法，			依次判断是否是实现了xxxAware接口，并执行。然后调用后置处理器的BeforeInitialization方法。然后调			用bean的初始化方法（过程中会判断是否是实现了initializationBean，是否有自己的初始化方法）。然后调用后置处理器的AfterInitialization方法

​			5.注册bean的销毁方法

执行完后返回到getSingleton中，调用addSingleton将创建好的bean放入一级缓存中，并从二级和三级中删除。

最终一个bean成功创建了



创建完所有剩余bean后回到preInstantiateSingletons()会调用SmartInitializingSingleton实现类的方法，@EventListener注解的原理就是基于此



（

懒加载或者是多实例的bean的创建也是调用了getBean的doGetBean里头去，

doGetBean除了判断是否单实例外，还会判断是否多实例，是的话，就直接用createBean创建，而不调用getSingleton方法，并返回创建的实例。

如果是其它的Scope类型，它会拿到Scope之后，调用get方法，没有细看

）



### finishRefresh()

首先初始化lifecycleProcessor组件，其定义了两个方法onRefresh和onClose会在生命周期中回调

然后发布容器刷新事件



至此spring启动完成



##### (主bean调用构造器函数注入子bean，会无法解决循环依赖)

循环依赖[(95条消息) Spring的构造函数注入的循环依赖问题_源码之下，了无秘密-CSDN博客_spring构造函数循环依赖不能解决的原因](https://blog.csdn.net/u010013573/article/details/90573901)



##### 三级缓存解决了动态代理时的循环依赖的问题

一级存放的是已经实例化而且依赖注入和初始化完成的bean；二级存的是已经实例化且已经被代理，但是没有依赖注入和初始化完成的bean；

第三级缓存缓存的是一个beanFactory（里边有实例化的bean），从三级缓存中拿是调用了beanFactory的getObject方法，在里边主要是遍历了SmartInstantiationAwareBeanPostProcessor实现类，调用他的方法，AutoCreateProxy就实现了这个接口，在实现方法中会去包装形成代理对象，同时在AutoCreateProxy中记录为已经代理，因此从三级中拿出来的是一个代理对象（如果需要代理），放入二级缓存中；

在实例化后遍历BeanPostProcessor调用AutoCreateProxy的方法的时候，会判断这个bean是不是已经被包装成代理对象（类里边有成员变量记录），是的话就不再继续包装。

[spring为什么要使用三级缓存解决循环依赖 - 简书 (jianshu.com)](https://www.jianshu.com/p/84fc65f2764b)



##### 不用三级缓存也可以？

第三级缓存主要用于处理有aop的循环依赖，而在bean实例化之后，初始化之前就将bean进行代理，可以去除三级缓存；

第二级缓存主要用于区分完整的bean以及半成品的bean，可以用一个set去保存半成品的bean的beanName，这样就能够去除二级缓存；

从而只用一级缓存就能实现三级缓存的功能；

[(107条消息) 关于spring 三级缓存的想法_我的编号9527的博客-CSDN博客](https://blog.csdn.net/weixin_45062785/article/details/119894023)



##### 为什么二三级缓存用HashMap，而不像一级缓存使用ConcurrentHashMap

二级缓存put的同时要保证三级缓存remove；三级缓存put时要保证二级缓存remove，也就是说二三级缓存操作要保证原子性
因为要保证同一个bean是单例的，不然都会lambda回调创建bean，就不是单例的了
如果使用ConcurrentHashMap并不能保证二三级缓存操作的原子性，所以要用synchronized
这三级缓存都是在synchronized内操作的，至于一级缓存为什么用ConcurrentHashMap，可能其他场景的原因吧？

[(107条消息) Spring循环依赖 | Spring三级缓存 | 看完必有收获_做猪呢，最重要的是开森啦的博客-CSDN博客](https://blog.csdn.net/weixin_43901882/article/details/120069307)



# @Autowire工作原理

修饰在构造方法上：createBean()时调用autowireConstructer方法（没有细看）

修饰在属性或者其它方法上：通过AutowiredAnnotationBeanPostProcessor，它实现了MergedBeanDefinitionPostProcessor以及InstantiationAwareBeanPostProcessor，前者的方法会将bean中被@Autowried（当然还包括@Value、@Inject）修饰的field、method找出来，将信息封装成对象并缓存起来，后者的方法会从中拿出来，然后从容器中拿，再通过反射设置进去。

（@Resource是由CommonAnnotationBeanPostProcessor支持，同样也是实现了xxxx）


（X）
## IOC

### 导入Bean

@Configuration告诉Spring这是一个注解类

@Bean添加到方法上，返回值注册到容器中，注解中没有指明名称则使用方法名作为bean的名称

@ComponentScan指定扫描的包，Spring会将添加了@Component的类扫描进去，可以使用指定排除，或者指定只包含（需要禁用默认规则）；它是一个重复注解，可以在类上重复使用，或者使用@ComponentScans。

@Scope的作用域（默认单实例）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201128175053847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3llcmVueXVhbl9wa3U=,size_16,color_FFFFFF,t_70#pic_center)

@Lazy实现懒加载，只对单实例bean有用

@Conditional（标注在方法或类上）和Condition实现按照条件注册bean

@Import（标注在类上）注册bean（导入第三方组件），三种方法：

​	直接填写class数组的方式

​	ImportSelector接口的方式，即批量导入，（在class数组上加入实现类），selectImports方法返回值是要导入	的bean的全类名

​	ImportBeanDefinitionRegistrar接口方式，即手工注册bean到容器中，（在class数组上加入实现类）实现接口	的类会被ConfigurationClassPostProcessor（一个BeanFactoryPostProcessor）处理，因此会优先于其它	   	bean初始化

实现FactoryBean<T>，加入到容器中，容器调用getBean()时实际是调用他的getObject()，控制其isSingleton()返回值可以实现单/多实例；由于BeanFactory类中指明了FACTORY_BEAN_PREFIX="&"，因此在getBean()中加入前缀&即可获得FactoryBean本身。



### bean属性的自动注入（！）

@Autowired默认是按照类型去容器中找对应的组件，如果找到多个相同类型的组件，那么是将属性名称作为组件的id，到IOC容器中进行查找。（不存在则报错，需要指定required=false）（加载方法或字段或类上）

@Qualifier与@Autowired搭配使用作按照名称去找

@Primary，我们可以利用这个注解让Spring进行自动装配的时候，默认使用首选的bean。（可加在@Bean上）（没有指定时才使用最高优先）

@Inject与@Autowired功能一样(无required)，但是通过@Named指定id

**@Resourse与@Autowired的区别**

共同点：两者都可以写在字段和setter方法上。两者如果都写在字段上，那么就不需要再写setter方法。

区别：

@Autowired是spring提供的注解，而@Resource由J2EE提供，需要导入包javax.annotation.Resource；

@Resource有两个中重要的属性：name和type。name属性指定byName，如果没有指定name属性，当注解标注在字段上，即默认取字段的名称作为bean名称寻找依赖对象，当注解标注在属性的setter方法上，即默认取属性名作为bean名称寻找依赖对象。需要注意的是，@Resource如果没有指定name和type，并且按照默认的名称仍然找不到依赖对象时， @Resource注解会回退到按类型装配。但一旦指定了name属性或type，就只能按名称或类型装配了。

@Resource没有requird属性以及不支持@Primary

**@Autowired标注位置**

字段上、实例方法上、构造方法（默认加在IOC容器中的组件，容器启动会调用无参构造器创建对象，然后再进行初始化、赋值等操作）（不需要再用无参构造方法）（只有一个有参构造可省略）、参数上（@Bean的方法参数默认从IOC容器中拿，只有一个时@Autowired可以省略）

上述都是从IOC中拿



### bean的组件的自动注入，实现@xxxAware（！）

原理使用XxxAwareProcessor实现，其实现了BeanPostProcessor，在postProcessBeforeInitialization()中调用invokeAwareInterfaces()会用多个if来判断是否实现了某个Aware接口，是则传入参数并调用bean的所实现的Aware接口的方法。



### bean的初始化方法与销毁方法的使用场景

一个典型的使用场景就是对于数据源的管理。例如，在配置数据源时，在初始化的时候，会对很多的数据源的属性进行赋值操作；在销毁的时候，我们需要对数据源的连接等信息进行关闭和清理。



### bean的初始化与销毁指定：

单实例的销毁随着容器的关闭而执行，多实例的销毁不归Spring管。

@Bean指定initMethod和destroyMethod；

bean实现InitializingBean接口，实现afterPropertiesSet方法作为初始化方法；bean实现DisposableBean接口，实现destroy方法作为销毁方法；（与上一种可同时存在，先执行这种）

@PostConstruct（服务器加载Servlet之后）（init之前）与@PreDestroy（服务器销毁Servlet之前），二者都是Java自己的注解，不是Spring提供的



### 后置处理器BeanPostProcessor的两个方法

postProcessBeforeInitialization方法会在bean实例化和属性设置之后，自定义初始化方法之前被调用，而postProcessAfterInitialization方法会在自定义初始化方法之后被调用。当容器中存在多个BeanPostProcessor的实现类时，会按照它们在容器中注册的顺序执行。对于自定义的BeanPostProcessor实现类，还可以让其实现Ordered接口自定义排序。



### BeanPostProcessor后置处理器作用

后置处理器可用于bean对象初始化前后进行逻辑增强。Spring提供了BeanPostProcessor接口的很多实现类，例如AutowiredAnnotationBeanPostProcessor用于@Autowired注解的实现，AnnotationAwareAspectJAutoProxyCreator用于Spring AOP的动态代理（postProcessAfterInitialization中会判断是否注册了切面，是则将其代理对象放入容器而不是对象本身）。





## AOP

### 开启AOP

配置类加@EnableAspectJAutoProxy注解，切面类加@Aspect，在切面类的方法上加@Before以及切入点表达式等（同时可以使用@Pointcut抽取切入点表达式），再将目标类与切面类加入IOC中。（切面类的方法中JoinPoint参数一定要放在参数列表的第一位，可以获取当前目标方法的参数信息等）

### @EnableAspectJAutoProxy干了什么

向Spring的配置类上添加@EnableAspectJAutoProxy注解之后，会通过ImportBeanDefinitionRegistrar的实现类，向IOC容器中导入AnnotationAwareAspectJAutoProxyCreator（注解装配模式的AspectJ切面自动代理创建器）

### AnnotationAwareAspectJAutoProxyCreator如何工作的

（不重要：

它是一个InstantiationAwareBeanPostProcessor实现类；实现其中的applyBeanPostProcessorsBeforeInstantiation()方法

在finishBeanInitialization()中，每一个bean进行createBean()时，在真正创建方法doCreateBean()之前，会调用resolveBeforeInstantiation()方法，尝试返回bean的实例，如果返回不为null，则调用applyBeanPostProcessorsAfterInitialization()。，返回不了再doCreateBean()。

resolveBeforeInstantiation()方法中，会依次遍历每一个后置处理器，判断类型是否是InstantiationAwareBeanPostProcessor，是则调用applyBeanPostProcessorsBeforeInstantiation()方法，此方法返回值不为空则跳出遍历。

）

1、是正常时的代理

AnnotationAwareAspectJAutoProxyCreator的对应方法中在一般返回空，基本不会用到，真正创建代理对象是在doCreateBean()之后的初始化后的，applyBeanPostProcessorsAfterInitialization()中，依次调用了所有beanPostProcessor的after...方法，AnnotationAwareAspectJAutoProxyCreator的对应方法过程中：

1. 找到候选的所有增强器（一个通知方法对应一个增强器）
2. 通过AopUtils根据切入点表达式获取到能在当前bean中使用的增强器
3. 给增强器排序
4. 如果存在对应的增强器
5. 把当前bean放入advisedBeans中，表示它已经被处理了；
6. 把增强方法等参数代理工厂中，通过代理工厂创建出一个代理对象（代理工厂会根据bean是否实现接口来判断使用jdk动态代理或者cglib动态代理）

2、是循环依赖时的代理

。。。。。

### 获取目标方法的拦截器链

（逻辑在CglibAopProxy的intercept中）

首先通过缓存获取，如果不为空就返回这个拦截器链，

否则，创建一个List，然后遍历每一个增强器Advisor，如果是PointcutAdvisor，则包装成MethodInterceptor放入list，如果是IntroductionAdvisor则包装成Interceptor放入list

PointcutAdvisor的包装逻辑为：先拿到增强器，然后判断这个增强器是不是MethodInterceptor这种类型的，是则直接添加进集合，若不是则使用AdvisorAdapter（增强器的适配器）将这个增强器转为MethodInterceptor这种类型。

将其放入缓存中



### 拦截器链的执行

如果拦截器链不为空，则创建CglibMethodInvocation对象并传入参数，调用它的procceed()，获取到返回值，否则调用直接利用反射调用methodProxy.invoke()传入bean与参数执行，获取返回值；

在procceed中，如果没有拦截器链，或者当前拦截器的索引（currentIntercptorIndex）和拦截器总数-1的大小一样，那么便直接执行目标方法。

否则，调用（++当前索引）（初始值为-1）对应拦截器的invoke()，并传入CglibMethodInvocation对象，再在其中执行CglibMethodInvocation的procceed()方法。

当推进到最后一个拦截器，也就是MethodBeforeAdviceInterceptor时，在其invoke()中调用前置通知。

执行CglibMethodInvocation的procceed()，此时当前索引超了，所以执行目标方法。

执行完后，栈回退到AspectJAfterAdvice的finally中，执行后置通知

然后回退到AfterReturningAdviceInterceptor拦截器（返回后通知），如果没抛异常，那么执行对应逻辑，如果目标方法执行过程中抛了异常，由于这个拦截器没有try-catch捕获异常（前面都有），因此异常抛给了上一个拦截器AspectJAfterThrowingAdvice（异常后通知），其try-catch捕获到对应异常，调用异常通知。



### 如何开启声明式事务

@Transactional标在方法上，配置类标注@EnableTransactionManagement

### 声明式事务原理

@EnableTransactionManagement通过ImportSelector的实现类向容器中导入两个组件：

1、AutoProxyRegistrar，它是一个ImportBeanDefinitionRegistrar，向容器中导入了InfrastructureAdvisorAutoProxyCreator组件（后置处理器），作用是负责为我们包装对象。

2、ProxyTransactionManagementConfiguration配置类，会向容器注册一个事务增强器，增强器中又包含一个TransactionInterceptor，它是一个MethodInterceptor，

在其invoke方法的执行过程中，首先会从容器中拿出事务管理器，然后创建事务，在try块中执行目标方法，catch块中捕获异常并并回滚，同时抛出异常，在finally中进行资源清除，try-catch块外，如果没有异常，则执行commit，并返回返回值。



## spring的扩展原理

### BeanFactoryPostProcessor原理

执行时机是在所有的bean定义已经保存加载到BeanFactory中了，但是bean的实例还未创建，此时可以定制和修改BeanFactory里面的一些内容。

在refresh中的invokeBeanFactoryPostProcessors中：

首先拿出容器中所有beanFactoryPostProcessors实现类的id

然后根据其是否实现priorityOrder接口（同时创建bean），order接口，id分成三类，然后创建bean添加到list中，前两类会进行sort，然后对于每一类

遍历此类的list，执行list中postprocessor的postprocessBeanFactroy方法。

### BeanDefinitionRegistryPostProcessor（上一个的子接口）

同样在refresh中的invokeBeanFactoryPostProcessors中：

首先执行的是BeanDefinitionRegistryPostProcessor的实现类的postProcessBeanDefinitionRegistry方法（如上），在执行它的postprocessBeanFactroy方法，最后再执行BeanFactoryPostProcessor的上述逻辑。

（为了防止在此逻辑中重复执行BeanDefinitionRegistryPostProcessor的postprocessBeanFactroy，方法中使用了set，记录已经执行了的BeanDefinitionRegistryPostProcessor）

### 事件监听

事件监听器实现ApplicationListener<E extends ApplicationEvent>

Spring发布事件时会调用一个publishEvent(事件类型)方法，然后获取到事件多播器，事件多播器根据传入的事件类型获取到对应的事件监听器，然后遍历事件监听器，调用器onApplicationEvent()方法。

Spring中有容器刷新完成事件以及容器关闭事件。

**多播器创建**

在refresh()中的initApplicationEventMulticaster()方法中会判断容器中有没有id=applicationEventMulticaster的组件，有就拿它作为多播器，没有就new一个SimpleApplicationEventMulticaster作为。

在refresh()中的registerListeners()方法中会从容器中拿到所有的监听器，然后把这些监听器注册到事件派发器中。

**@EventListener原理**

使用的是EventListenerMethodProccessors（实现了SmartInitializingSingleton）；

在refresh()中的finishBeanFactoryInitialization()中的preInstantiateSingletons()中，创建了所有剩余的单实例bean后，会进入for循环，判断每个bean是否实现了SmartInitializingSingleton，是则调用其afterSingletonsInstantiated方法。

在EventListenerMethodProccessors对应的方法的processBean()中，遍历容器中的bean找到有@EventListener注解的方法拿到Method对象和对应bean的beanName,将两个参数封装到ApplicationListener的实现类ApplicationListenerMethodAdapter里,并注册到容器中,事件发布时通过Method和beanName反射执行@EventListener注解的方法



# Spring的事务传播行为

事务传播行为用来描述由某一个事务传播行为修饰的方法被嵌套进另一个方法的时事务如何传播。

##### PROPAGATION_REQUIRED：

在外围方法未开启事务的情况下`Propagation.REQUIRED`修饰的内部方法会新开启自己的事务，且开启的事务相互独立，互不干扰；在外围方法开启事务的情况下`Propagation.REQUIRED`修饰的内部方法会加入到外围方法的事务中，所有`Propagation.REQUIRED`修饰的内部方法和外围方法均属于同一事务，一个方法异常，整个事务均回滚。（可以try-catch防止回滚，但是此时会尝试提交，如果提交无法提交，再次抛异常，这时又会回滚）

##### PROPAGATION_REQUIRES_NEW：

外围没开启事务时同上；开启时，内部方法依然会单独开启独立事务，内部事务间相互独立，外围事务抛异常不会影响已提交的内部事务，内部事务抛异常会被外围事务感知并回滚，但是可以catch异常，单独对子事务回滚。

##### PROPAGATION_NESTED：

外围方法没有开事务时同上；开了的话内部事务为外围事务的子事务，外围方法回滚，内部方法也要回滚，内部方法抛出异常回滚，且外围方法感知异常致使整体事务回滚，但是可以catch异常，单独对子事务回滚。

##### PROPAGATION_SUPPORTS：

如果存在一个事务，加入当前事务。如果没有事务，则非事务的执行。

##### PROPAGATION_NOT_SUPPORTED：

总是非事务地执行，并挂起外围事务。

##### PROPAGATION_MANDATORY：

如果外围已经存在一个事务，加入当前事务。如果没有则抛出异常。

##### PROPAGATION_NEVER

总是非事务地执行，如果外围存在一个活动事务，则抛出异常。



## 不那么重要

### 向Bean属性动态注入外部值

@value加在属性上

#{}和${}，$和#使用都必须符合SpEL表达式，可以混用，但是前者的执行时机更早

### 加载配置文件（properties）

@PropertySource注解读取外部配置文件中的key/value之后，是将其保存到运行的环境变量中了，我们也可以通过运行环境（applicationContext.getEnvironment()）来获取外部配置文件中的值。

### 根据环境注册bean

@profile加在配置类或者方法上，当指定的环境生效时才会将bean注册到ioc中，通过命令行-Dspring.profiles.active=test指定参数，或者用applicationContext的environment的setactiveprofiles设置



### BeanFactory与ApplicationContext的区别

支持国际化（实现了MessageResource）;

强大的事件机制；

前者延迟加载；

自动注册后置处理器；

[Spring中 BeanFactory和ApplicationContext的区别_慕课手记 (imooc.com)](https://www.imooc.com/article/264722)



### IOC与DI？

IoC 指控制反转。

控制 ：指的是对象创建（实例化、管理）的权力

反转 ：控制权交给外部环境（Spring 框架、IoC 容器）

DI是依赖注入，是IOC的具体实现，传统的程序都是消费者主动创建对象，现在容器帮我们查找及注入依赖对象,而消费者只是被动的接受依赖对象。



### OOP面向对象编程？

例如：现有三个类，`Horse`、`Pig`、`Dog`，这三个类中都有 eat 和 run 两个方法。

通过 OOP 思想中的继承，我们可以提取出一个 Animal 的父类，然后将 eat 和 run 方法放入父类中，`Horse`、`Pig`、`Dog`通过继承`Animal`类即可自动获得 `eat()` 和 `run()` 方法。这样将会少些很多重复的代码。



### AOP？

是OOP的延申，比如在父类 Animal 中的多个方法的相同位置出现了重复的代码，OOP 就解决不了。这种在多个纵向流程中出现的相同子流程代码，称为横切逻辑代码，纵向代码中除了横切逻辑以外的代码称为业务逻辑代码。AOP叫面向切面编程，AOP把横切逻辑代码分离出来，称为切，而受影响的多个方法称为面。



### Spring通知有哪些类型？

在AOP术语中，切面的工作被称为通知。通知实际上是程序运行时要通过Spring AOP框架来触发的代码段。

Spring切面可以应用5种类型的通知：

1. 前置通知（Before）：在目标方法被调用之前调用通知功能； 
2. 后置通知（After）：在目标方法完成之后调用通知，此时不会关心方法的输出是什么； 
3. 返回通知（After-returning ）：在目标方法成功执行之后调用通知； 
4. 异常通知（After-throwing）：在目标方法抛出异常后调用通知； 
5. 环绕通知（Around）：通知包裹了被通知的方法，在被通知的方法调用之前和调用之后执行自定义的逻辑。



### SpringBoot的@Controller能不能和@Service注解互换？

boot启动时默认加载**spring.factories**文件中的配置类，其中存在一个WebMvcAutoConfiguration；

这个配置类会导入一个HandlerMapping，它有初始化方法，在其中会遍历所有的bean，如果bean加了Controller注解或者加了RequestMapping注解，会扫描其中的方法，注册到HandlerMapping中；

[(119条消息) SpringMVC系列:HandlerMapping初始化_fyygree的博客-CSDN博客_handlermapping初始化](https://blog.csdn.net/fengyuyeguirenenen/article/details/123846662)



### SpringBoot的自动装载原理

##### 是什么？

没有SpringBoot的时候，要使用mybatis-plus这样的第三方依赖的时候，引入相关jar包后，还要在配置类里边配置一些SqlSeesionFactory的bean，十分麻烦；SpringBoot只需要引入starter，以及注解或者一些简单的配置，就能够实现某一块功能；

##### 怎么实现？

`@SpringBootApplication`看作是 `@Configuration`、`@EnableAutoConfiguration`、`@ComponentScan` 注解的集合；@EnableAutoConfiguration是实现自动装配的核心注解

它通过@Import导入了一个AutoConfigurationImportSelector，实现了 `ImportSelector`接口；

在对应的方法里边，他首先读取`META-INF/spring.factories`，获取需要自动装配的所有配置类，并通过@EnableAutoConfiguration注解的exclude属性     以及    要装配的配置类上面的@ConditionalOnXX（比如当容器中有指定bean的时候才装配）进行过滤，再进行类的加载以及装配。

##### 装配流程？

run的时候，会创建一个SpringApplication对象，首先会加载整个应用程序的**spring.factories**文件到缓存中；

调用SpringApplication对象的run方法完成整个应用程序的启动；

里边的**prepareContext**()方法将启动类注册到容器里边；**refreshContext**()方法执行容器的刷新，到了Spring容器的refresh，BFPP接口的实现类会对解析@Import等注解；

[(123条消息) springboot自动装配_好好玩_tyrant的博客-CSDN博客_springboot自动装配](https://blog.csdn.net/qq_57434877/article/details/123933529)
  
  
  
https://juejin.cn/post/7004281096074428423

