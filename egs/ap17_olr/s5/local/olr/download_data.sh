#!/bin/bash

## Download the data for AP17-OLR Challenge.
## License agreement is required.
## Contact the organizer:
## Dr. Dong Wang (wangdong99@mails.tsinghua.edu.cn)
## Dr. Zhiyuan Tang (tangzy@cslt.riit.tsinghua.edu.cn)

address1=''
address2=''

wget -c --referer=$address1 -O data.tar.gz "$address2" || exit 1;
tar zxvf data.tar.gz || exit 1;

exit 0;
