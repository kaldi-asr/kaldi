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
经过漫长的安装后，装好了。cd到s5里，运行
        
        $ . ./path.sh       ##系统执行时，只会从path路径里找，每次./执行时，会重新开一个shell，所以就不在当前path了，要想让系统能找到当前path，就需要开头的. 这是让系统添加进当前的path，相当于export或者source ./path.sh。要导入kaldi的路径。
        $ copy-             ##然后按tab（不是回车），copy是打印输出流。见https://blog.csdn.net/by21010/article/details/51776447

库一般放在`tools`里，常用的库：openfst（解码时需要用到）

说明：`/egs`里一般存放执行的脚本，`.sh`文件，讲了每一步要执行什么操作，而真正的算法程序部分，在`/src`里的`.cc`、`.h`文件里。

查看环境 `$ env`

然后就可以cd到s5，执行
       
       $ ./run.sh

（以下是以为的解决方法，其实并不是：）
    `path.sh`导入script中。

    xielongdeMBP:s5 yelong$ . ./run.sh    //注意，. .中间有空格
    法二：
    
    vim ~/.bash_profile
    export PATH=${PATH}:/Users/yelong/kaldi/egs/thchs30/s5
    退出后source ~/.bash_profile 

## scp、ark
**archive(.ark)、script(.scp)** ：是表格(table)一个‘表’就是一组有序的事物，前面是识别字符串（如句子的id），一个‘表’不是一个c++的对象，因为对应不同的需求（写入、迭代、随机读入）我们分别有c++对象来读入数据。

.scp格式是text-only的格式，每行是个key（一般是句子的标识符（id））后接空格，接这个句子特征数据的路径 。

.ark格式可以是text或binary格式，（你可以写为text格式，命令行要加‘t’，binary是默认的格式）文件里面数据的格式是：key（如句子的id）空格后接数据。
    xielongdeMacBook-Pro:train yelong$ head feats.scp 
    A02_000 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:8
    A02_001 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:12868
    A02_002 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:26222
    A02_003 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:38029
    A02_004 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:50239
    A02_005 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:62865
    A02_006 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:75322
    A02_007 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:86401
    A02_008 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:96102
    A02_009 /Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:110912
    xielongdeMacBook-Pro:train yelong$ copy-matrix ark:/Users/yelong/kaldi/egs/thchs30/s5/mfcc/train/raw_mfcc_train.1.ark:8 ark,t:-
    
    
 ## run.sh
 把run.sh看懂，然后在把aishell的nnet3搬到thchs30跑一遍。
 
 ## 流程
 首先进行准备prepare，自动生成`/s5/data`里的文件，这里面的文件是匹配信息，语音信息-说话人是哪个的信息-文件位置 动态生成这种文件。然后`run.sh`中的每个算法（从单音子、三音子，到nnet3等算法）都跑一边。每个算法都会用EM算法迭代多次。
 
 翻译模型 lexicon 是匹配字对应的音素，解码时，用 语言模型*lexicon*声学模型，声学模型是训练出来的，语言模型是一开始就建立好的。比如50帧，50个观测状态，最后要输出50个状态（隐状态），解码就是在一个状态网络里走，从起点开始走50步，对应50帧，每一步找概率最大的那个路径，所谓的走一步，其实下一步可以仍然是和上一步一样的状态，就是比如声学模型中走到本身概率0.6，转移0.4，在语言模型中，每一处的下一步有很多分支可以走，不同概率，与声学模型概率相乘，得到最大概率的路径。最后输出一个50帧的状态序列（隐），从这个**状态序列**，匹配对应的文字。（文字和观测状态没有对应关系）
 
 有**wav**语音，还有对应的 **wav.trn 标注**，训练是以标注作为评判标准。

## CTC
**MFCC**是每一帧对应一个状态（最大概率对应的状态），而**CTC**是很多帧输出一个状态，理解为，好几帧输出的概率不确定（可能最大概率的那个概率没有远大于其他概率，所以是不确定的），那么就输出空白blank，直到有确定概率了，才输出那个状态。

## rspecifier、wspecifier
在`/src`的.cc中，像类型一样的存在。
**rspecifier**用于说明如何读表的字符串；
**wspecifier**用于说明如何写入表的字符串。
需要注意的是 “rspecifier”和“wspecifier”并不是C++中的类或者对象，他们只是为了方便使用，对变量的描述名称，
