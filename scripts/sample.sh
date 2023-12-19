target= echo "Enter target directory: "
read target

#extin= echo "enter extension of input audio file: "
#read extin

#extout= echo "enter extension of output audio file: "
#read extout

#outch= echo "enter output channel: "
#read outch

#dest= echo "Enter Destination file name with extension: " #Also can be used for customizing output file name along with extension
#read dest
mkdir $target/output
mkdir $target/OG

for f in "$target"/*.wav;
do
#    sox --channels 1 "$f".wav -r 16000 "$f"-converted.wav  
     sox "$f" -r 16000 -b 16 -c 1 "$f"-converted.wav  
     cd $target
     base= basename $f
     mv "$f"-converted.wav output/$f
     mv "$f" OG
done

cd $target
mv output/*.wav "$target"
