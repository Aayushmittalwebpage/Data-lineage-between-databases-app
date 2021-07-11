# Data Lineage between Relational databases

## Introduction

I've built a system to create an easy-to-understand visualization of the data spreading across multiple databases where the various transformations on it can be tracked. Additionally, this algorithm finds relationships between data and its degree of replication across databases that can help make better informed decisions and can minimize redundant data.


# Install

## Requirements

Developed and tested on [Python 3.5 and 3.6, 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](
https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system.


## Why build lineage using machine learning and statistics?
While many solutions to the data lineage problem have been proposed before, all available attempts at tackling the problem rely either on handcrafted heuristics , code analysis , and/or manual annotation of data. While these are all valid ways of lineage tracing, they all have limitations. Code analysis requires access to the code, which may be inaccessible when dealing with external applications or if the code was lost/deprecated. Algorithms require a strict set of conditions to be able to run, such as knowing ahead of time all the possible transformations, which cannot always be satisfied. Finally, solutions that require annotations are great for future lineage tracing, but they were not designed to answer the question of where the present non-annotated data came from. This approach does not have these same limitations.







