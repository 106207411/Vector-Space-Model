## Vector Space Model

VSM is a classic and powerful model for information retrieval. It allows you to retrieve documents relevant to the given query. In this project, it offers two kind of vector representation (TF and TF-IDF) and two kind of similarity measurement (cosine and L2 norm).

The program ranks 7034 EnglishNews according to the user's given query, and recommends the top-5 News eventually.

### 3rd party libraries

argparse, nltk

### Usage

User's query is followed by `--query` argument.

```shell
python3 main.py --query "Trump Biden Taiwan China"
```

### Directory layout

    .
    ├── main.py                 # Main program to run
    ├── Parser.py               # Package
    ├── PorterStemmer.py        # Package
    ├── util.py                 # Package
    ├── VectorSpace.py          # Package
    ├── english.stop            # StopWords
    ├── EnglishNews             # Folder with 7034 EnglishNews
    └── README.md

### Files introduction

- main.py

  Main program to run.

- Parser.py

  Remove stopwords and tokenize documents.

- PorterStemmer.py

  Word stemming of the Porter stemming algorithm, ported to Python from the version coded up in ANSI C by the author.

- util.py

  Duplicates word removal, cosine similarity and L2-norm calculation.

  - removeDuplicates(list): set((item for item in list))
  - cosine(vector1, vector2): dot(vector1,vector2) / (norm(vector1) * norm(vector2)),
  - euclidean(vector1, vector2): norm(vector1-vector2)
    norm: sqrt(sum(item**2)) for item in vector

- VectorSpace.py

  A class of vector space generated from the given documents.
  Function feedback_search() is used to get a pseudo relevant feedback.

- english.stop

  English stopwords.