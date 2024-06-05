# Instructions

### installing tools
```
$ mkdir tools
$ cd tools
```
#### Moses
```
$ git clone https://github.com/moses-smt/mosesdecoder.git
```
#### Fasttext
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
$ wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O language_recognition.bin
```

#### Idiomata Cognitor 
```
git clone https://github.com/transducens/idiomata_cognitor.git
cd idiomata_cognitor
pip install -r requirements.txt
```