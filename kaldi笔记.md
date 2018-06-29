## 错误01 run.pl: 4 / 4 failed, log is in exp/make_mfcc/train/make_mfcc_train.*.log
打开`make_mfcc_train.1.log`文件，发现：

    bash: line 1: copy-feats: command not found
    bash: line 1: compute-mfcc-feats: command not found

解决：path.sh导入script中。

    xielongdeMBP:s5 yelong$ . ./run.sh    //注意，. .中间有空格
法二：
    
    vim ~/.bash_profile
    export PATH=${PATH}:/Users/yelong/kaldi/egs/thchs30/s5
    退出后source ~/.bash_profile 
