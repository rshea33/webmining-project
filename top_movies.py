from selenium import webdriver
from bs4 import BeautifulSoup

import pandas as pd


def main():

    url = "https://www.imdb.com/chart/top/?ref_=nv_mv_250"
    driver = webdriver.Safari()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.close()

    
    # get rankings
    ranks = soup.find_all('td', class_='titleColumn')
    movie_ranks = [int(rank.text.split('.')[0]) for rank in ranks]

    # get titles
    titles = soup.find_all('td', class_='titleColumn')
    movie_titles = [title.text for title in titles]
    movie_titles = [title.replace('\n', '') for title in movie_titles]
    movie_titles = [title.split('.')[1] for title in movie_titles]
    movie_titles = [title.split('(')[0] for title in movie_titles]
    movie_titles = [title.strip() for title in movie_titles]

    # get ratings
    ratings = soup.find_all('td', class_='ratingColumn imdbRating')
    movie_ratings = [float(rating.text) for rating in ratings]

    # get year movie was released
    years = soup.find_all('span', class_='secondaryInfo')
    movie_years = [int(year.text.replace('(', '').replace(')', '')) for year in years]


    # create a dataframe & csv
    movies = pd.DataFrame({
        'movie_title': movie_titles,
        'movie_year': movie_years,
        'movie_rating': movie_ratings
    })
    movies.set_index(pd.Index(movie_ranks), inplace=True)
    movies.to_csv('movies.csv')


if __name__ == '__main__':
    main()