## -- : 利用两个连续的连字符 --，显式地告诉前者到此为止。
  
参考https://liam0205.me/2016/11/11/ways-to-parse-arguments-in-shell-script/
  
`ls`命令是用来列出当前目录下的文件和子目录的。它可以接受一些选项（如`-lrt`）。现在的问题是，如果有一个文件，它的名字叫做`-foobar`，要怎样才能列出它的相关信息呢？

实际上`ls`内部使用了`getopts`解析参数。于是我们可以这样

    ls -lrt -- -foobar
    
利用两个连续的连字符`--`，显式地告诉`getopts`：到这为止！然后，`ls`会读入`-foobar`作为文件名，显示它的相关信息。

## export : 有点像拷贝一份变量
参考http://roclinux.cn/?p=1277


## ~
参考 https://blog.csdn.net/chun_1959/article/details/23243935
在Linux（unix）中，以波浪线“~”开始的文件名有特殊含义。

单独使用它或者其后跟一个斜线（`~/`），代表了当前用户的宿主目录。（在shell下可以通过命令“echo ~(~\)”来查看）。例如“~/bin”代表“/home/username/bin/”（当前用户宿主目录下的bin目录）


波浪线之后跟一个单词（~word），其代表由这个“word”所指定的用户的宿主目录。

例如
“~john/bin”就是代表用户john的宿主目录下的bin目录。


在一些系统中（像MS-DOS和MS-Windows），用户没有各自的宿主目录，此情况下可通过
设置环境变量“HOME”来模拟。

## -x
"-x"选项可用来跟踪脚本的执行，是调试shell脚本的强有力工具。“-x”选项使shell在执行脚本的过程中把它实际执行的每一个命令行显示出来，并且在行首显示一个"+"号。 "+"号后面显示的是经过了变量替换之后的命令行的内容，有助于分析实际执行的是什么命令。 “-x”选项使用起来简单方便，可以轻松对付大多数的shell调试任务,应把其当作首选的调试手段。


## head 
举例
    $ head -3 data/train/text
输出：
A02_000 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然
A02_001 他 仅 凭 腰部 的 力量 在 泳道 上下 翻腾 蛹 动 蛇行 状 如 海豚 一直 以 一头 的 优势 领先
A02_002 企业 依靠 技术 挖潜 增效 他 负责 全厂 产品质量 与 技术培训 成了 厂里 的 大忙人

head -number 后面接地址 能输出文件的前number行

## which
用于查找并显示给定命令的绝对路径，环境变量PATH中保存了查找命令时需要遍历的目录。which指令会在环境变量$PATH设置的目录里查找符合条件的文件。也就是说，使用which命令，就可以看到某个系统命令是否存在，以及执行的到底是哪一个位置的命令。比如 `$ which pwd`

## 概念解释
打印到标准输出流，就是打印到屏幕上

## -
用 "-" 来表示标准输入输出。 


## ./
./运行shell脚本，不能直接test.sh，直接写 test.sh，linux 系统会去 PATH 里寻找有没有叫 test.sh 的，而只有 /bin, /sbin, /usr/bin，/usr/sbin 等在 PATH 里，你的当前目录通常不在 PATH 里，所以写成 test.sh 是会找不到命令的，要用 ./test.sh 告诉系统说，就在当前目录找。
这是作为可执行文件的运行shell脚本方法，还有作为解释器参数运行shell脚本，此时 `/bin/sh test.sh`  和 `/bin/php test.php`

参考 http://www.runoob.com/linux/linux-shell-basic-operators.html

## ps -ef |grep 
`ps`命令将某个进程显示出来；

`-e`：显示所有进程。

`-f`：全格式。

`ps -e `列出程序时，显示每个程序所使用的环境变量。

`ps -f` 用ASCII字符显示树状结构，表达程序间的相互关系

`ps -ef |grep -c "ssh"` 返回1

`ps -ef |grep java`：检查java进程是否存在；`ps -ef | grep httpd`：检查httpd进程是否存在
字段含义如下：
UID       PID       PPID      C     STIME    TTY       TIME         CMD
501       6096      5976      0     11:50上午 ttys006    0:00.00    grep java

    ID      ：程序被该 UID 所拥有
    PID      ：就是这个程序的 ID 
    PPID    ：则是其上级父程序的ID
    C          ：CPU使用的资源百分比
    STIME ：系统启动时间
    TTY     ：登入者的终端机位置
    TIME   ：使用掉的CPU时间。
    CMD   ：所下达的是什么指令

## -n
  
  1.  `echo -n "test"` / `echo -n 'test'`输出后不换行
  2.  字符串运算符-n。检测字符串长度是否为0，不为0返回 true。[ -n "$a"]返回 true。
      
    if [ -n "$a" ]
    then
       echo "-n $a : 字符串长度不为 0"
    else
       echo "-n $a : 字符串长度为 0"
    fi
