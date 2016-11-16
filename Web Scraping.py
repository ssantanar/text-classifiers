#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from url import urlparse
from urllib.parse import urljoin
from lxml import html
import time
import os
#Mejores
response = requests.get ( "http://www.imdb.com/list/ls073671144/" )
soup = BeautifulSoup ( response.text , "lxml" )
# clase del DIV donde se guarda cada pelicula
sub_soup = soup.select ( ".list_item" )

links = [ ]
for sub in sub_soup:
    links.append ( urljoin ( "http://www.imdb.com/" , sub.select ( "a" )[ 0 ].attrs[ "href" ] ) )

print ( links )
# -------------Coment=pos
i = 57 #Los comentarios de las mejores 10 se guardan en los comen_pos[:57]
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/Naive_Bayes/IMBD_pos')
for link in links[ :len(links) ]:
    print ( "Crawling %s" % link )
    response = requests.get ( urljoin ( link , "reviews?filter=love" ) )
    soup = BeautifulSoup ( response.text , "lxml" )
    movie_coment_text = soup.find ( "hr" ).findNext ( "div" ).findNext ( "p" ).text.strip ( )
    movie_coment_code = movie_coment_text.encode('utf-8')
    movie_coment_str = str(movie_coment_code)
    i = i+1
    with open("coment_pos{0}.txt".format(i), "w") as out_file:
           out_file.write(movie_coment_str)
           time.sleep(1)
          
#-----------Coment=neg
j = 57#Los comentarios de las mejores 10 se guardan en los comen_neg[:57]
os.chdir('/Users/Seba/R-Proyects/Web Intelligence/Tarea 1/Naive_Bayes/IMBD_neg')
for link in links[ :len(links) ]:
    print ( "Crawling %s" % link )
    response = requests.get ( urljoin ( link , "reviews?filter=hate" ) )
    soup = BeautifulSoup ( response.text , "lxml" )
    movie_coment_text = soup.find ( "hr" ).findNext ( "div" ).findNext ( "p" ).text.strip ( )
    movie_coment_code = movie_coment_text.encode('utf-8')
    movie_coment_str = str(movie_coment_code)
    j = j+1
    with open("coment_neg{0}.txt".format(j), "w") as out_file:
           out_file.write(movie_coment_str)
           time.sleep(1)

