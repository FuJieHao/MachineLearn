# -*- coding: utf-8 -*-
import scrapy


class LoginSpider(scrapy.Spider):
    name = 'login'
    allowed_domains = ['iqianyue.com']
    start_urls = ['http://iqianyue.com/']
    header = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"}

    #取代start_urls
    def start_requests(self):
        pass
    def parse(self, response):
        pass
