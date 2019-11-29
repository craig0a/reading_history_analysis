# reading_history_analysis

This repository represents a person project to explore new methodologies for data analysis and machine learning, especially in areas which I have not had the opportunity to work much in before. 

As a dataset, I am using an export of the books I have read, as logged on my Goodreads account (https://www.goodreads.com/craig0a). I chose this as a suitable data set to explore because I am of course intimately familiar with the underlying drivers of the data set (my own personal preferences) and therefore can easily check the "reasonability" of new analysis results. The data exported includes the books' titles, authors, publisher,  number of pages, year published, original publication year, date added (typically the day I read it, but mass-uploads of books read before I began using Goodreads also included), my rating (scale 1-5), and whether or not I would recommend the book. As of November 03, 2019 I have data on 610 books. 

### Notebook set 0 - scraping Goodreads for more data & initial hand-curated preprocessing
As a first step in my analysis, I wanted to gather additional data from Goodreads: namely the book descriptions, the author biographies, and what shelves other users had most frequently assigned the book to. Once I had collected the shelves for the books, I map those onto more general book categories. I have also begun work on more general web scraping to expand publisher information when the Goodreads data provides only an acronym. 

### Notebook set 1 - initial exploration & naive Bayes analysis
This was a first look at the categorization of the books (e.g. are the different categories roughly independent or is there strong cross-shelving? are the subcategories within each primary category strongly related?) and then a brief look at how these categories are related to my personal book ratings. 

Although the correlation between categories forbids a true Baysian analysis, as a first indication, I performed a naive Bayes analysis of my reading preferences: the probability of a rating given a category and the probabilty of a category given a rating.

### Notebook set 2 - Topic modeling
Exploring NLP techinques, I am working on developing topic modeling of the books based on their descriptions.  I am also doing "topic" modeling on the author bios and, specifically, the author professions. My hypothesis is that I tend to prefer books written by "professors" than "journalists". 

### (TODO) Notebook set 3 - Multi-label classification of books 
Once I have the topic extractions reasonably refined, I am intending to build a multi-label classifier, which will be validated against the manually-curated categorizations based on Goodreads user shelves, but which will hopefully be somewhat more robust and extendable. 

### (TODO) Notebook set 4 - Book preference modeling 
My hypothesis is that I tend to prefer books written by "professors" than "journalists" and books published by a "University Press" to books published by "Basic Books", regardless of topic. My goal is to explore the truth of these and other hypotheses by building prediction models for my rating based on extracted features. 
