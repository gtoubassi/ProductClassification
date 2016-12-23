# ProductCategorization 

This repo explores simple methods for categorizing content (a product) using both image and text features.

### Data

The dataset used for categorization is pulled from the www.shopstyle.com public API.  The relevant portions of the product definition pulled from the shopstyle api are product name, description, image, and correct category.  The goal is to use some or all of the name/description/image data to correctly predict the category.  An example datum is below:

<img src="https://img.shopstyle-cdn.com/sim/4f/41/4f41ca111ba265702f1d416ea79aebd2_medium/kut-from-the-kloth-womens-natalie-stretch-curvy-bootcut-jeans.jpg"/>
Name: Women's Kut From The Kloth 'Natalie' Stretch Curvy Bootcut Jeans","brandedName":"KUT from the Kloth Women's 'Natalie' Stretch Curvy Bootcut Jeans","unbrandedName":"Women's 'Natalie' Stretch Curvy Bootcut Jeans
Description: Fading and whiskering add well-worn appeal to the dark-blue wash of curve-flattering bootcut jeans, while plain back pockets make for a sleek rear view. Color(s): lift/ dark stone. Brand: KUT FROM THE KLOTH. Style Name:Kut From The Kloth 'Natalie' Stretch Curvy Bootcut Jeans (Lift/dark Stone). Style Number: 5209389. Available in stores.
Category: bootcut-jeans

Again the goal is to train a network that when presented with name/description/image of a never before seen product, it will correctly classify it as bootcut jeans.  Note there are a total of about 250 leaf categories.


''crawl.py'' is used to crawl up to 1000 products from each leaf category of the shopstyle category tree, storing product data in a locally created sqlite database, and images on the filesystem.

### Image base categorization

''categorize_images.py'' will categorize images using a pretrained Inception network (see references below).  After running ''categorize_images.py'' you can run ''analyze.py'' to generate a report of all of your runs.

### Resources

[TensorFlow for Poets Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) - Shows how to leverage a pre-trained Inception neural network to classify your own image corpus.  Essentially it uses a (second to last?) late stage layer of the network to create a 2048 dimension "embedding" of each input image.  It then trains a simple 2-layer softmax network on those embeddings to predict the desired categories associated with the images.

[Course Videos for CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG) - An overview of neural networks for image recognition and an excellent discussion of convolutional neural netowkrs in lecture 7.

[Course Videos for CS224D: Deep Learning for Natural Language Processing](https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG) - Richard Socher's lecture videos cover how to use neural networks in NLP tasks.
