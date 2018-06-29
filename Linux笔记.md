### -- : 利用两个连续的连字符 --，显式地告诉前者到此为止。
  
参考https://liam0205.me/2016/11/11/ways-to-parse-arguments-in-shell-script/
  
`ls`命令是用来列出当前目录下的文件和子目录的。它可以接受一些选项（如`-lrt`）。现在的问题是，如果有一个文件，它的名字叫做`-foobar`，要怎样才能列出它的相关信息呢？

实际上`ls`内部使用了`getopts`解析参数。于是我们可以这样

    ls -lrt -- -foobar
    
利用两个连续的连字符`--`，显式地告诉`getopts`：到这为止！然后，`ls`会读入`-foobar`作为文件名，显示它的相关信息。

### export : 有点像拷贝一份变量
参考http://roclinux.cn/?p=1277


### ~
参考 https://blog.csdn.net/chun_1959/article/details/23243935
在Linux（unix）中，以波浪线“~”开始的文件名有特殊含义。

单独使用它或者其后跟一个斜线（~/），代表了当前用户的宿主目录。（在shell下可以通过命令“echo ~(~\)”来查看）。例如“~/bin”代表“/home/username/bin/”（当前用户宿主目录下的bin目录）


波浪线之后跟一个单词（~word），其代表由这个“word”所指定的用户的宿主目录。

例如
“~john/bin”就是代表用户john的宿主目录下的bin目录。


在一些系统中（像MS-DOS和MS-Windows），用户没有各自的宿主目录，此情况下可通过
设置环境变量“HOME”来模拟。
