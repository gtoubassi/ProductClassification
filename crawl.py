#!/usr/bin/env python
#
import argparse
import urllib
import json
import os
import sys
import database

ApiKey = os.environ['SHOPSTYLE_API_KEY']
OutputDir = 'output'

def fetchImages(categoryId):
    dir = OutputDir + '/' + categoryId
    if not os.path.isdir(dir):
        os.makedirs(dir)

    count = 1    
    for offset in xrange(0,10):
        url = "http://api.shopstyle.com/api/v2/products?cat=%s&pid=%s&offset=%d&limit=50" % (categoryId, ApiKey, offset * 50)
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        products = data['products']
        for product in products:
            imageUrl = product['image']['sizes']['XLarge']['url']
            filename = imageUrl.split('/')[-1]
            print '%s fetching # %d (%s)' % (categoryId, count, filename)
            urllib.urlretrieve(imageUrl, dir + '/' + filename)
            count += 1

def fetch(categoryId, depth=1):
    url = "http://api.shopstyle.com/api/v2/categories?cat=%s&pid=%s&depth=1" % (categoryId, ApiKey)
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    categories = data['categories']

    for category in categories:
        if depth == 2:
            print 'Fetching for ' + category['id']
            fetchImages(category['id'])
        else:
            fetch(category['id'], depth + 1)

def fetchProducts(numToFetch, startOffset):
    pageSize = 50
    count = 1 
    allProducts = []
    for offset in xrange(0, numToFetch / pageSize):
        url = "http://api.shopstyle.com/api/v2/products?pid=%s&offset=%d&limit=%d" % (ApiKey, startOffset + offset * pageSize, pageSize)
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        products = data['products']
        for product in products:
            imageUrl = product['image']['sizes']['XLarge']['url']
            simpleProduct = {}
            simpleProduct['id'] = product['id']
            simpleProduct['name'] = product['name']
            simpleProduct['description'] = product['description']
            simpleProduct['imageUrl'] = imageUrl
            simpleProduct['category'] = product['categories'][0]['id']
            allProducts.append(simpleProduct)
            filename = imageUrl.split('/')[-1]
            dir = OutputDir + '/images/' + filename[0:2]
            path = dir + '/' + filename
            if not os.path.exists(path):
                if not os.path.isdir(dir):
                    os.makedirs(dir)

                print 'fetching # %d (%s)' % (count, filename)
                urllib.urlretrieve(imageUrl, path)
            else:
                print 'skipping # %d (%s)' % (count, filename)
                
            count += 1

    with open(OutputDir + '/products.json', 'w') as outfile:
        json.dump(allProducts, outfile, sort_keys = True, indent = 4)
    

#fetch('clothes-shoes-and-jewelry')
#fetch('womens-clothes')
#fetchProducts(20000, 10000)

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
    for offset in xrange(0,30):
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
        print "crawling " + cat
        count = crawProductsInCategory(cat)
        db.setCategoryProductCount(cat, count)

def crawlImages(outputDir):
    images = db.getProductImagesToCrawl()
    count = 1
    for image in images:
        filename = image.split('/')[-1]
        dir = outputDir + '/' + filename[0:2]
        path = dir + '/' + filename
        if not os.path.exists(path):
            if not os.path.isdir(dir):
                os.makedirs(dir)

            print 'fetching # %d of %d (%s)' % (count, len(images), filename)
            urllib.urlretrieve(image, path)
        else:
            print 'skipping # %d of %d (%s)' % (count, len(images), filename)
        count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default='crawl.db', help="Path to sqlite db file")
    parser.add_argument("--images-path", default='images', help="Path to directory in which images should be saved")
    args = parser.parse_args()

    global db
    db = database.Database(args.db_path)
    
    #print "Crawling categories"
    #crawlCategories(None)

    #db.populateCategoryPath()
    
    #print "Crawling products"
    #crawlProducts()
    
    crawlImages(args.images_path)

if __name__ == "__main__":
    main()
