# -*- coding: utf-8 -*-
import scrapy
from dangdang.items import DangdangItem
from scrapy.http import Request


class DdSpider(scrapy.Spider):
    name = 'dd'
    allowed_domains = ['dangdang.com']
    start_urls = ['http://category.dangdang.com/pg1-cid4011011.html']

    def parse(self, response):
        items =  DangdangItem()
        items['title'] = response.xpath("//a[@name='itemlist-picture']/@title").extract()
        items['link'] = response.xpath("//a[@name='itemlist-picture']/@href").extract()
        items['comment'] = response.xpath("//a[@name='itemlist-review']/text()").extract()
        #print(items['comment'])
        #scrapy crawl dd --nolog
        yield items
        for i in range(2, 51):
            url = "http://category.dangdang.com/pg" +str(i)+ "-cid4011011.html"
            yield Request(url=url, callback=self.parse)