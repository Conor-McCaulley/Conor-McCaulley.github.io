---
layout: post
title: Project Reflection!
---

### Overall Achievement
In this project, we created a webapp that helps our users find new recipes. We aimed to add more variety to people’s diets because many people, especially college students with busy schedules, find it difficult to decide on and cook something new to eat. 

Users can either use the recipe finder function, which allows them to find a recipe according to the ingredients they enter; or use the recipe generator function that generates an entirely new recipe based on recipe length.

### What We're Proud of
#### Gracie:
Successfully writing the scraper. It was significantly harder to scrape from a website that has a lot of nested tags compared to the blog post where we wrote an IMDB scraper. 
Used knowledge from the sqlite and flask lectures to create a functional webapp. 

#### Conor:
Even though I did not build the generator to be as robust and grammatically accurate as I originally would have liked, I’m actually really proud of the way it turned out. It was able to learn the basic structure of a recipe and many of the common words (teaspoon, cook, oven, etc.). Building the generator was a challenging process that took 2 iterations to get right and I’m proud of the end result.
I’m also proud of the fact that I was able to overcome a really personally challenging quarter and get the project done in the last few weeks even though I’d been really struggling before that. 

### Future Work
Our web app still has a very basic aesthetic, and it’s only accessible locally when you download the GitHub folder. I tried deploying it to Heroku, but ran into quite a few bugs. Using pre-existing CSS stylesheets and successfully deploying it onto the internet are things that could be done to further improve the project. 
In order to successfully deploy to Heroku we would need to either purchase enough space that we could host the tensorflow library which is required for our project or simply cut out the recipe generator function. 
I would also like to work on further improving the performance of the generator model. The current model does a character by character generation which means that it can sometimes string together words but has a hard time making coherent sentences. I’d like to rebuild the model using word by word generation combined with a pretrained embedding layer to help the model ‘understand’ the meaning of the words it is generating and produce more coherent output text. 

### What we proposed to do VS What we’ve done

We had really high expectations for our group project at first. As written in our project proposal, we initially wanted our end product to comprehensively recommend recipes according to not only specific ingredients, but also nutritional value and dietary needs. We also wanted to build a restaurant recommender which we never had the time to build. Thus, we first narrowed our project down to just a recipe searcher. And we did manage to make something new: the recipe generator! This was a way of still incorporating machine learning when the original idea to incorporate ML (the restaurant recommender) did not seem feasible. 

### 3 Things You Learned

#### Gracie:
1. I learned to work collaboratively on GitHub. At first, I couldn’t understand why a folder in my computer could be “pushed” to a website, but after doing the 2nd blog post, I started getting used to using GitHub and found it to be very helpful. 
2. I’m now relatively confident in web-scraping.
3. I learned to build flask webapps that connect to a database and require SQL commands.

#### Conor:
1. I also learned how to use GitHub! I was pretty confused at first especially when we started collaborating on code and I somehow managed to end up with a merge conflict. But I was able to solve the merge conflict and am now far more comfortable using GitHub both for personal and group projects. 
2. I learned the importance of building a minimum viable product first and then improving it rather than attempting to build the final product all at once. When I built the generator model I spent a whole day (read 14 hours straight) trying to build a model integrating layers that I did not fully understand because the internet said that was the way to achieve optimal performance. However, even once I got it to run it did not work very well.  Once I went back to the drawing board and built a simpler model that I did understand, it worked much better and I was able to improve on it from there. 
3. I learned a lot about machine learning in TensorFlow. I was already comfortable with SciKit Learn before this class but had no TF experience. I am now pretty comfortable in Tensorflow which is nice since it gives a lot more flexibility in terms of building your own models layer by layer and incorporating pre trained models through transfer learning. This can really help with reducing training time and overall accuracy on more complex machine learning problems. 

### Above and Beyond

I think that this project will prove helpful in both my future personal and professional plans. I’m really interested in machine learning and I want to keep expanding my horizons with TensorFlow. The base knowledge I gained through this project has given me enough context and confidence to keep learning and experimenting with TF on my own. I also think that the knowledge of working collaboratively with GitHub will prove to be extremely useful. I don't yet know if I want to pursue a career in computer science but at the very least I know I want to keep doing personal projects that involve python programming. Specifically I’d really like to learn more about building IOT devices because I also enjoy the hardware aspect. Whether I’m just working collaboratively on personal projects with friends or doing computer science professionally, being able to write and edit code as a team using GitHub will be invaluable. Thank you so much for a great quarter professor Chodrow!
