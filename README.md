# WaveNet Implementation for Text-To-Speech
My TUM IDP Project to make Angela Merkel sing.

A blog post on this implementation and experiments is here: https://medium.com/@evinpinar/wavenet-implementation-and-experiments-2d2ee57105d5

## folder structure
```
│
├── data                <- Put your data here (on your local machine just a sample probably)
│                         in the .gitignore 
│
├── log                <- Checkpoints of trained models, evaluations and other logs
│                         in the .gitignore 
│
├── notebooks          <- Jupyter notebooks. Naming convention is lab/pretty (see Wiki) 
│                         followed by a short `-` delimited description, e.g.
│                         `lab-initial-exploration`.
│
├── src                <- Source code for use in this project.
│   │
│   ├── core           <- git submodule of the luminovo core repo
│   │                     containing utils, tools and code re-usable between projects
│   │
│   └── run.py         <- put code/experiments you want to run in here
│
└── README.md          <- The top-level README for developers using this project.

