#!/bin/bash
#新包根目录 ip1 ip1的旧包根目录 ip1的密码 ip2 ip2的旧包根目录 ip2的密码 ...

#获取新包根目录
newClassPath=$1
shift

#如果此时入参不为3的倍数，报错并退出
if [ $[$#%3] -ne 0 ]; then
	echo "输入正确的参数，格式：新包根目录 ip1 ip1的旧包根目录 ip1的密码 ip2 ip2的旧包根目录 ip2的密码 ..."	
	exit 0
fi

#对每一个节点处理
paramArray=("$@")
for((i=0; i<$#; i=$[i+3]))
do
	ip=${paramArray[i]}
	oldClassPath=${paramArray[$[i+1]]}
	password=${paramArray[$[i+2]]}
	echo "ip:${ip}    oldClassPath:${oldClassPath}    password:${password}"
done