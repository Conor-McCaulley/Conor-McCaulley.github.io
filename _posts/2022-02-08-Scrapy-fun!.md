---
layout: post
title: Scrapy Madness!
---
## Intro

In this blogpost we'll be building a webscraper using the python scrapy library. The scrapy library makes it easy to build your own web scrapers with minimal work. In this example we will be building a scraper to scrape an IMDB page for a movie or TV show, the scraper will then go to the pages or each actor in the show and return dictionaries of that actors name and each movie/tv show they have been in. This will allow us to look for other shows which have a lot of the same actors which we might also like.

### Getting Started

To get started we need to initialize a scrapy project. We do this through the terminal using the command 'scrapy startproject IMDB_scraper'. In order for this to work we need to have scrapy installed on the version of python that we are running. This can be done in anaconda navigator using the extentions tab. If you are new to scrapy or the terminal shell I highly recommend this github repository from Phil Chodrow, it will walk you through everything you need to know about creating a scrapy project.
https://github.com/PhilChodrow/PIC16B/blob/master/lectures/scrape/lecture-scraper/tutorialscraper/spiders/tutorialscraper.py

After running 'scrapy startproject IMDB_scraper' we have now created a scrapy project called IMDB_scraper. In the terminal we can navigate to it using the command 'cd IMDB_scraper'. This will eventually allow us to run our project.




### Building our Scraper


Now that we have a scrapy project. Its time to build the actual scraper. We start by creating a new file 'imdb_spider.py' in our spiders folder. Within this file we need to name the spider which we will do by setting the name variable equal to 'imdb_spider' and set the start_urls variable equal to a list containing the url of the imdb page of our chosen show or movie. We now need to define the three parse methods which will allow our scraper to do its job. We will be defining a parse method for each type of page that the scraper will have to visit. These pages are the homepage for the movie/tv show we selected, the full credits page for that show, and the actor pages for each actor in the show.

### Parse


This method will parse the IMDB page for the show we've selected. All that it needs to do is find the url for the full credits page for the show and use it to call the parse_full_credits method. We do this by retreiving the url using 'response.url' and appending 'fullcredits' to the end of it to get the url for the full credit page. Response in this case is a special object that stores the full information of the page being scraped. We can now use this to call:


```python
yield Request(full_cast_url, callback = self.parse_full_credits)
```

This in effect navigates the spider to the full_cast_url page and the callback tells it to use the parse_full_credits method to parse that page.

### Parse_full_credits

Our parse_full_credits function will retrieve the imdb page urls for each actor who worked on the show and then yield reaquests to each of these pages using the parse_actor_page method. It does this by first retreiving all the anchor elements that link to individual actor pages. Each of these has a css hierarchy of 'table.cast_list tr a:first-child' so we can use:


```python
actor_links =response.css("table.cast_list tr a:first-child")
```

This saves every element with this css hierarchy to the actor_links variable. We can then retreive the hrefs (the links) from each of these anchor elements using (element).attrib['href'] and then append "https://www.imdb.com" to the begining of each to turn the relative links into absolute links. Now all we need to do it itterate through each url and use it to yield a new request with the callback set to use parse_actor page.

### Parse_actor_page

Our final step to to define the parse_actor_page method. This is the method which will actually return the data we want to scrape. We start by obtaining the actors name. We can do this by selecting the element with the css hierarchy 'td.name-overview-widget__section span.itemprop::text' and using the .get() method to retrieve the text it contains. Each page only has one element with this unique css hierarchy so we don't need to worry about multiple elements getting selected. Now we need to retrieve the title for each movie/tv show they have been in. To do this we can select elements with the css hierarchy "div.filmo-category-section div.filmo-row b a::text" and then using .get() on each element to obtain the contained text. We now have our actors name and a list with all their movie/tv titles. We can now itterate through the list of titles and for each one yield a dictionary containing the title and the actors name.

## Running our Scraper

Our scraper is now done and we can run it! To do this we can open a terminal and navigate to the scraper using the command 'cd IMDB_scraper'. We can now use the command 'scrapy crawl imdb_spider -o results.csv' to run the scraper and save the data to a new .CSV file. Woohoo! You did it! We can now use that data to build an interesting visualization of other shows that have many of the same actors.

### Ploting

We can use the code below to extract the data from the .csv file and use it to create a plot of shows with the same actors.


```python
#first we have to import Pandas and Matplotlib
import pandas as pd
from matplotlib import pyplot as plt

#next read the data from the results.csv file into a pandas dataframe
movies = pd.read_csv('results.csv')

#group the datapoints by Movie/Show Name and take the size of each to find how many of the same
#actors were in each show. Then convert to a dataframe and reset the index
movie_overlap=movies.groupby("movie_or_TV_name").size().to_frame().reset_index()

#rename the columns to 'Movie/Show Name' and 'Shared Actors'
movie_overlap.columns = ['Movie/Show Name', 'Shared Actors']

#sort by the number of shared actors and take the 10 with the largest overlap to plot
movie_overlap=movie_overlap.sort_values(by=['Shared Actors'],ascending= False).reset_index()
movie_overlap=movie_overlap[1:11]

#plot the results!
ax = movie_overlap.plot(x='Movie/Show Name', y='Shared Actors', kind='bar')
```


    
![scrapy_graph.png]({{ site.baseurl }}/images/scrapy_graph.png)
    


You're done! You've created an interesing visulization of which other shows/movies have the most actor overlap with your favorite show. As you can see above, Criminal Minds and Grey's Anatomy have the most actors in common with my favorite show, Silicon Valley. 

For the full project code, you can access the [github repository](https://github.com/Conor-McCaulley/Blog-post-3).


