# Product Classifier 

This repo explores simple methods for classifying content (products) using both image and text features.

The dataset used for classification is pulled from the www.shopstyle.com public API.  The relevant portions of the product definition pulled from the shopstyle api are product name, description, image, and correct category.  The goal is to use some or all of the name/description/image data to correctly predict the category.  An example datum is below:

<img src="https://img.shopstyle-cdn.com/sim/4f/41/4f41ca111ba265702f1d416ea79aebd2_medium/kut-from-the-kloth-womens-natalie-stretch-curvy-bootcut-jeans.jpg"/>

**Name**: Women's Kut From The Kloth 'Natalie' Stretch Curvy Bootcut Jeans

**Description**: Fading and whiskering add well-worn appeal to the dark-blue wash of curve-flattering bootcut jeans, while plain back pockets make for a sleek rear view. Color(s): lift/ dark stone. Brand: KUT FROM THE KLOTH. Style Name:Kut From The Kloth 'Natalie' Stretch Curvy Bootcut Jeans (Lift/dark Stone). Style Number: 5209389. Available in stores.

**Category**: bootcut-jeans

Some challenges with classifying this content comes from the fact that the leafs categories are often ambiguous.  Bootcut vs flare jeans is not always clear.  And what about a pair of bootcut jeans that are also distressed (another subcategory under jeans).  Another class of problems comes from the fact that there are subcategories for "athletic", "maternity", "plus size" and "petites", each of which has a "tops" subcategory.  So a polo could be either a "womens-tops/polo-tops", an "athletic/ahtletic-tops", a "maternity/top", a "plus-sizes/plus-size-tops", or a "petites/petite-tops".

The sample dataset I worked with consisted of all leaf categories under the "women" category, which includes clothing, handbags, shoes, jewelry, and beauty products.  Men's, kids & baby, and living categories were ignored.  Only leaf categories that had 1000 or more products were considered.  These restrictions left 156 leaf categories and just over 146k products (yes there should be 156*1000 products, some bug during the crawl presumably).  The categories covered are listed in `results/categories.txt`.

Experiments were run on an 80%/10%/10% train/validation/test split where total accuracy was measured.  If machine predicted categories were used in a real product UI you would want to be especially sensitive to false positives (you really don't want a bra showing up in the jeans section), so in addition to total accuracy, coverage @ 95%/98% accuracy is measured.  Total accuracy of 85% over the total corpus is not as interesting as knowing that 70% of the corpus can be covered with a threshold that has been shown to have a very very low error rate (98% implies 1 in 50 mistakes)

### Image classification

The first approach was to classify entirely based on the image content.  The approach used [retrain.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining) which repurposes a pre-trained Inception CNN to classify a new image corpus.  This is an example of "transfer learning", which is based on the assumption that the generally trained Inception network will recognize useful features of any image that then can be leveraged for domain specific classification.  The second to last layer outputs for each image are taken essentially as an embedding of the image into a 2048 dimensional space (2048 is the size of the second to last layer).  So with each image represented as a 2048 dimension vector, you can perform simple logistic regression to train a classifier.

Initial results looked promising for simple cases.  For example classifying "clutches" (a kind of handbag) vs "bootcut-jeans" was done with 100% accuracy, which is to be expected givne they are so visuall distinct.  A more complex case of "skinny-jeans" vs "bootcut-jeans" was also promising with 89% accuracy and 73% coverage @95% accuracy.  However when covering the entire corpus of 146k products the overall accuracy dropped to 56%, and coverage @95% was only 12% of the corpus.  Examples of the top 500 (meaning the incorrect categories that the network had the highest confidence in) can be seen in `results/image-classification-top500-errors.html`.  Note the first 20 or so look to be legitimate errors on the part of the dataset (meaning the classifier got it right).

### Text classification

The second approach used the product name and description of the product to classify it.  A simple 3 layer network was used.  The input was a 20k sparse vector representing a bag-of-words representation of the name/description for the product for the top 20k terms found in the corpus.  The hidden layer is 100 ReLus, and the final softmax layer performs classification.  The results on the entire 146k corpus was an overall accuracy of 89% and coverage @95% of 82%.  Top errors can be seen in `results/text-classification-top500-errors.html`.

This bag of words classifier is very naive.  TBD try a word2vec style embedding (or perhaps [Swivel](https://github.com/tensorflow/models/tree/master/swivel)) to see if it improves.

### Combined Image+Text classification

I tried combining the two by constructing a network that essentially concatenated the 100 hidden layer text classifier values with the 2048 image embedding from retrain.py and used a softmax layer over all 2148 inputs for classification.  Results were essentially the same as the text classifier.  Essentially the image classifier seems to add no value which is unsurprising given the text classifier is so far superior to the image classifier.

### How-To

To approximate the results discussed above, first you need a dataset.  Go to http://www.shopstylecollective.com and create an account, and in the account profile section, note your API key and make it available in the environment.

    $ export SHOPSTYLE_API_KEY=uid#####-###########-###

Perform the crawl.  This will take up to a day or so and can be restarted.

    $ python crawl.py

The crawl stores product information in a local sqlite database called `crawl.db`.  You can explore the database using the `sqlite3` command line tool executing queries like `.schema category`, `.schema product`, `select * from category;` and `select * from product limit 20;`.

Now you can attempt to classify using images (drop the `--categories` argument if you want to do the full dataset).  Note it will take awhile the first time because it has to compute the embedding for each image (referred to in the output and the retrain.p code as the `bottleneck`).

    $ python categorize_images.py --categories bootcut-jeans,skinny-jeans

To run the text classifier:

    $ python categorize_words.py --categories bootcut-jeans,skinny-jeans

To run the combined classifier:

    $ python categorize_both.py --categories bootcut-jeans,skinny-jeans

Each of the above classification runs will leave results data in the sqlite database.  This can be interrogated directly with `sqlite3` with queries like `select * from experiment` and `select * from predicted_category pc where pc.experiment_id=1`.  More conveniently you can generate a report of all experiments in the db by running:

    $ python analyze.py
	#1: Image classification of: 
	  Accuracy: 56.19% (N=14867)
	  95% Accuracy Coverage: 12%
	  98% Accuracy Coverage: 4%
	  Parent Accuracy: 70.05%
	  Parent 95% Accuracy Coverage: 19%
	  Parent 98% Accuracy Coverage: 5%
	#2: Text classification of: 
	  Accuracy: 89.06% (N=14638)
	  95% Accuracy Coverage: 83%
	  98% Accuracy Coverage: 54%
	  Parent Accuracy: 94.58%
	  Parent 95% Accuracy Coverage: 98%
	  Parent 98% Accuracy Coverage: 78%
	...

You can use `analyze.py` to generate an html report of the top errors (the ones that were wrong that the classifier was most certain of):

    $ python analyze.py --dump-errors-for-experiment 2 >> errors.html

### Resources

[TensorFlow for Poets Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) - Shows how to leverage a pre-trained Inception CNN to classify your own image corpus.  It is a hands on tutorial for [retrain.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining)

[Course Videos for CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG) - An overview of neural networks for image recognition and an excellent discussion of convolutional neural netowkrs in lecture 7.

[Course Videos for CS224D: Deep Learning for Natural Language Processing](https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG) - Richard Socher's lecture videos cover how to use neural networks in NLP tasks.  Word embeddings are covered as well as some nitty gritty backprop derivations.
