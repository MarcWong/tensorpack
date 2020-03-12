## referseg_rrn

##  https://github.com/lichengunc/refer
##  https://github.com/chenxi116/TF-phrasecut-public

## Result

Arch|metadata|result|
|---|----|----|
Arch|metadata|result|
mynetwork.naive.nocap|bs=1,train_imgs=4k,val_imgs=1k|14.5|
mynetwork.naive|bs=1,train_imgs=4k,val_imgs=1k|22.3|
mynetwork.naive.bs4||25.6|
mynetwork.naive.bs4.train6k||28.97|
mynetwork.naive.bs4.train10k||33.27|
mynetwork.naive.bs4.train20k||37.2|
mynetwork.naive.bs4.train40k||42.24|
mynetwork.naive.nocap.alldataset.bs5.scale1||38.7|
mynetwork.naive.nocap.alldataset.bs5.scale2||37.8(epoch8)|
mynetwork.naive.nocap.train6k.bs4.scale2||26.4
mynetwork.naive.nocap.train10k.bs4.scale2||28.49|
mynetwork.naive.nocap.train20k.bs4.scale2||33.2|
mynetwork.naive.nocap.train40k.bs4.scale2||35.56|



Arch|metadata|result|
|---|----|----|
Arch|metadata|result|
mynetwork.naive.nocap.81fix.train6k.bs4.scale2||29.8|
mynetwork.naive.nocap.81fix.train10k.bs4.scale2||31.8|
mynetwork.naive.nocap.81fix.train20k.bs4.scale2||34.56|
mynetwork.naive.nocap.81fix.train40k.bs4.scale2||38.1|
mynetwork.naive.81fix.bs4.train6k||33.64|
mynetwork.naive.81fix.bs4.train10k||37.29|
mynetwork.naive.81fix.bs4.train20k||41.8|
mynetwork.naive.81fix.bs4.train40k||:question:|
mynetwork.naive666.81fix.bs10.train6k||13.29|
mynetwork.naive666.81fix.bs4.train6k||21.2|
mynetwork.naive666.indrnn.81fix.bs4.train6k||16.0|
mynetwork.naive666.indrnn.81fix.bs4.train10k||21.9|
mynetwork.naive666.indrnn.81fix.bs4.train20k||28.46|
mynetwork.naive666.81fix.bs4.train6k.rnnlayer2||:question:|







## story

#### change image proportion(4k,10k,nocap-80k)
#### add more sentence to an image? see effect
#### attention mechanism
#### multi-scale guiding
#### feature map visualization

## contribution

#### our paper first employ the caption information into the segmentation of the image for better segmentation by exploring the relationship of objects.

#### attentation and multi-scale




