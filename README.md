# Data Lineage between Relational databases

## Introduction
With multiple departments in an organization, there comes multiple databases for different use-cases. Between these databases data flows and transforms. Understanding where data comes & goes across databases and determining its relationship across is essential to support the business and to get complete understanding of data and systems. 

In data science, the inability to trace the lineage of particular data is a common and fundamental problem that undermines both security and reliability. Understanding data lineage enables auditing and accountability, and allows users to better understand the flow of data through a system.

Hence, I've built a system to create an easy-to-understand visualization of the data spreading across multiple databases where the various transformations on it can be tracked. Additionally, relationships between data and its degree of replication across databases can help make better informed decisions and can minimize redundant data.


# Install

## Requirements

**DataTracer** has been developed and tested on [Python 3.5 and 3.6, 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](
https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system.


## Why using machine learning and statistics?
While many solutions to the data lineage problem have been proposed before, all available attempts at tackling the problem rely either on handcrafted heuristics (which provides algorithms for lineage tracing in data warehouses), code analysis ( which provides a system for lineage tracing relying on source code analysis), and/or manual annotation of data (which provides a data annotation system for relational databases). While these are all valid ways of lineage tracing, they all have limitations. Code analysis requires access to the code, which may be inaccessible when dealing with external applications or if the code was lost/deprecated. 
Algorithms require a strict set of conditions to be able to run, such as knowing ahead of time all the possible transformations, which cannot always be satisfied. Finally, solutions that require annotations are great for future lineage tracing, but they were not designed to answer the question of where the present non-annotated data came from. This approach does not have these same limitations.







