#!/usr/bin/env python
#
import argparse
import urllib
import json
import os
import sys
import database
import threading

ApiKey = os.environ['SHOPSTYLE_API_KEY']

def crawlCategories(categoryId):
    categoryQueryParam = ("&cat=" + categoryId) if categoryId is not None else ""
    url = "http://api.shopstyle.com/api/v2/categories?pid=%s&depth=1%s" % (ApiKey, categoryQueryParam)
    print url
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    parentId = data['metadata']['root']['id']
    categories = data['categories']
    leaf = 1 if len(categories) == 0 else 0
    
    db.addCategory(parentId, data['metadata']['root'].get('parentId'), leaf)
    
    for category in categories:        
        crawlCategories(category['id'])

def crawProductsInCategory(categoryId):
    count = 0
    for offset in xrange(0,20):
        url = "http://api.shopstyle.com/api/v2/products?cat=%s&pid=%s&offset=%d&limit=50" % (categoryId, ApiKey, offset * 50)
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        products = data['products']
        count += len(products)
        if len(products) == 0:
            break
        for product in products:
            db.addProduct(product['id'], product['name'], product['description'], product['categories'][0]['id'], product['image']['sizes']['XLarge']['url'])
    return count

def crawlProducts():
    categories = db.getCategoriesToCrawl()
    for cat in categories:
        print "Crawling " + cat
        count = crawProductsInCategory(cat)
        db.setCategoryProductCount(cat, count)

def crawlImagesInList(outputDir, images):
    count = 1
    for productId, image in images:
        filename = image.split('/')[-1]
        dir = outputDir + '/' + filename[0:2]
        path = dir + '/' + filename
        if not os.path.exists(path):
            if not os.path.isdir(dir):
                os.makedirs(dir)

            print 'fetching # %d of %d (%s)' % (count, len(images), filename)
            urllib.urlretrieve(image, path)
            #db.setProductImageCrawled(productId)
        else:
            print 'skipping # %d of %d (%s)' % (count, len(images), filename)
        count += 1

def crawlImages(outputDir):
    images = db.getProductImagesToCrawl()
    sublists = [images[x:x+30000] for x in xrange(0, len(images), 10000)]

    workers = []
    for sublist in sublists:
        worker = threading.Thread(target=crawlImagesInList, args=(outputDir,sublist))
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default='crawl.db', help="Path to sqlite db file")
    parser.add_argument("--images-path", default='images', help="Path to directory in which images should be saved")
    args = parser.parse_args()

    global db
    db = database.Database(args.db_path)
    
    # Crawl categories if necessary
    if len(db.getCategories()) == 0:
        print "Crawling categories"
        crawlCategories('women')
        db.populateCategoryPath()

    # Crawl products if necessary
    #crawlProducts()
    
    #crawlImages(args.images_path)

if __name__ == "__main__":
    main()
