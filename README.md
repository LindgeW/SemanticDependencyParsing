# Graph-based Semantic Dependency Parsing  

This demo extends the syntactic biaffine parser for any graph-structured semantic dependency scheme, including directed cyclic or acyclic graphs.
In semantic dependency graph, each word is allowed for multiple head nodes. As illustrated in the following figure, “国内” is the argument of “专家” and at the same time it is an argument of “学者” .

### Example  
![sdp_demo](imgs/demo.png)  

### Performance
| Corpus | UF | LF |
| ---- |  ---- | ---- |
| NEWS | 82.47 | 66.64 |
| TEXT | 86.99 | 75.86 |


### Dataset  
[SemEval-2016 Task9](https://github.com/HIT-SCIR/SemEval-2016)
