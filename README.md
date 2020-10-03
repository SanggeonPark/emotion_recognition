# emotion_recognition
Python project to build a face emotion recognition model

## Usage

Install pip:

```bash
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python3 get-pip.py
```

Install dependencies:

```bash
$ pip3 install -r requirements.txt --use-feature=2020-resolver
```

Build a model with train data:

```bash
$ python3 src/train.py
```

Run test
```bash
$ python3 src/test.py
```

If you face some issues,
1. please make sure the python3 path is in your PATH.
2. if python complains about six package, please do the following:
```bash
$ pip3 install --ignore-installed six
```
