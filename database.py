import sqlite3 as sql


class Database:
    
    def __init__(self, dbPath):
        self.con = sql.connect(dbPath, isolation_level=None) #autocommit mode
        self.upgradeSchema()
    
    def upgradeSchema(self):
        cur = self.con.cursor()        
        cur.execute("select count(*) from sqlite_master where type='table' and name='schema_version'")
        count = cur.fetchone()
        
        if count[0] == 0:
            cur.execute("create table schema_version(id integer primary key autoincrement, date text, name text)")
        
        cur = self.con.cursor()        
        if self.needMigration('category table'):
            cur.execute("create table category(id text primary key, parent text, leaf integer default 0, product_count integer default 0)")

        if self.needMigration('product table'):
            cur.execute("create table product(id integer primary key, category_id text, image text, name text, description text)")

        if self.needMigration('add category.path'):
            cur.execute("alter table category add column path text")

        if self.needMigration('add product.image_crawled'):
            cur.execute("alter table product add column image_crawled integer default 0")

        if self.needMigration('add experiment schema'):
            cur.execute("create table experiment(id integer primary key autoincrement, name text, date text)")
            cur.execute("create table predicted_category(experiment_id integer, product_id integer, category_id text, score real)")

    def needMigration(self, name):
        cur = self.con.cursor()        
        cur.execute("select count(*) from schema_version where name = ?", (name,))
        count = cur.fetchone()
        if count[0] == 0:
            cur.execute("insert into schema_version(date, name) values(datetime(), ?)", (name,));            
            return 1
        return 0

    def addCategory(self, categoryId, parentId, leaf):
        cur = self.con.cursor()        
        cur.execute("insert or ignore into category(id, parent, leaf) values (?, ?, ?)", (categoryId, parentId, leaf))

    def getCategoriesToCrawl(self):
        cur = self.con.cursor()        
        cur.execute("select id from category where leaf=1 and product_count=0")
        cats = cur.fetchall()
        return [cat[0] for cat in cats]
        
    def getCategories(self):
        cur = self.con.cursor()        
        cur.execute("select id from category")
        cats = cur.fetchall()
        return [cat[0] for cat in cats]
        
    def addProduct(self, productId, name, description, categoryId, image):
        cur = self.con.cursor()        
        cur.execute("insert or ignore into product(id, name, description, category_id, image) values (?, ?, ?, ?, ?)", (productId, name, description, categoryId, image))
    
    def setCategoryProductCount(self, categoryId, count):
        cur = self.con.cursor()        
        cur.execute("update category set product_count = ? where id = ?", (count, categoryId))
    
    def getProductImagesToCrawl(self):
        cur = self.con.cursor()        
        cur.execute("select p.id, p.image from product p, category c where c.id=p.category_id and c.product_count >=1000 and p.image_crawled=0")
        return cur.fetchall()
    
    def setProductImageCrawled(self, productId):
        cur = self.con.cursor()        
        cur.execute("update product set image_crawled=1 where id=?", (productId,))
        
    
    def populateCategoryPath(self):
        cur = self.con.cursor()        
        cur.execute("select id from category where leaf=1")
        cats = cur.fetchall()
        leafs = [cat[0] for cat in cats]
        
        for leaf in leafs:
            path = []
            while leaf:
                path.insert(0, leaf)
                cur.execute("select p.id from category c, category p where c.id=? and p.id=c.parent", (leaf,))
                leaf = cur.fetchone()
                if leaf:
                    leaf = leaf[0]
            
            path.pop(0)
            catPath = '/'.join(path)
            cur.execute("update category set path=? where id=?", (catPath, path[-1]))
