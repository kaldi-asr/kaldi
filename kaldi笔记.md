## 错误01 run.pl: 4 / 4 failed, log is in exp/make_mfcc/train/make_mfcc_train.*.log
打开`make_mfcc_train.1.log`文件，发现：

    bash: line 1: copy-feats: command not found
    bash: line 1: compute-mfcc-feats: command not found
解决：
    只下载了kaldi，没有装很多库，所以要把这些库装上。主要是`wget`，安装方法（先cd到tools里）：
    
    $ brew install wget --with-libressl
    $ sudo chown -R $(whoami) /usr/local
    $ brew link wget
这个只是把代码下载下来了，要安装成可编译文件，需要`make`一下，一般再`make install`，才是安装。（这里先不`make install`）

然后cd到src里，编译配置文件：
        
        $ ./configure --shared  
        $ make depend; make -j4         ##添加依赖，-j4指的是开4个进程 开始安装了.
经过漫长的安装后，

库一般放在`tools`里，常用的库：openfst（解码时需要用到）

说明：`/egs`里一般存放执行的脚本，`.sh`文件，讲了每一步要执行什么操作，而真正的算法程序部分，在`/src`里的`.cc`、`.h`文件里。

（以下是以为的解决方法，其实并不是：）
    `path.sh`导入script中。

    xielongdeMBP:s5 yelong$ . ./run.sh    //注意，. .中间有空格
法二：
    
    vim ~/.bash_profile
    export PATH=${PATH}:/Users/yelong/kaldi/egs/thchs30/s5
    退出后source ~/.bash_profile 
