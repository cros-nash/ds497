import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = ['http://quotes.toscrape.com/page/1/',
                  'http://quotes.toscrape.com/page/2/',
                  'http://quotes.toscrape.com/page/3/',
                  'http://quotes.toscrape.com/page/4/',
                  'http://quotes.toscrape.com/page/5/',
                  'http://quotes.toscrape.com/page/6/',
                  'http://quotes.toscrape.com/page/7/',
                  'http://quotes.toscrape.com/page/8/',
                  'http://quotes.toscrape.com/page/9/',
                  'http://quotes.toscrape.com/page/10/']

    def parse(self, response):

        author = response.css('small.author::text').get() # FILL ME IN!
        print(author)
