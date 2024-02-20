import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def scrape_company_info():
    url_industry = "https://finance.yahoo.com/quote/GOEV/profile"
    response_industry = requests.get(url_industry)
    soup_industry = BeautifulSoup(response_industry.content, 'html.parser')
    industry = soup_industry.find("span", text="Industry").find_next("span").text.strip()

    url_auto_manufacturers = "https://finance.yahoo.com/sectors/consumer-cyclical/auto-manufacturers"
    response_auto_manufacturers = requests.get(url_auto_manufacturers)
    soup_auto_manufacturers = BeautifulSoup(response_auto_manufacturers.content, 'html.parser')
    market_cap = soup_auto_manufacturers.find("span", text="Market Cap").find_next("span").text.strip()
    industry_weight = soup_auto_manufacturers.find("span", text="Industry Weight").find_next("span").text.strip()
    top_companies = [company.text for company in soup_auto_manufacturers.find_all("a", class_="Fw(600)")]
    top_5_companies = top_companies[:5]

    return industry, market_cap, industry_weight, top_5_companies

def scrape_company_details(top_5_companies):
    all_company_data = []

    for company in top_5_companies:
        company_data = {}
        url_company = f"https://finance.yahoo.com/quote/{company}/profile"
        response_company = requests.get(url_company)
        soup_company = BeautifulSoup(response_company.content, 'html.parser')
        
        table_rows = soup_company.find_all("tr")
        for row in table_rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].text.strip()
                value = cells[1].text.strip()
                company_data[key] = value

        all_company_data.append(company_data)

    return all_company_data

def scrape_key_statistics():
    url_statistics = "https://finance.yahoo.com/quote/GOEV/key-statistics"
    response_statistics = requests.get(url_statistics)
    soup_statistics = BeautifulSoup(response_statistics.content, 'html.parser')

    key_statistics_data = {}
    table_rows = soup_statistics.find_all("tr")
    for row in table_rows:
        cells = row.find_all("td")
        if len(cells) == 2:
            key = cells[0].text.strip()
            value = cells[1].text.strip()
            key_statistics_data[key] = value

    return key_statistics_data

def scrape_news_headlines():
    url_news = "https://www.reuters.com/finance/stocks/sector-snapshot/auto-vehicles"
    response_news = requests.get(url_news)
    soup_news = BeautifulSoup(response_news.content, 'html.parser')
    headlines = [headline.text.strip() for headline in soup_news.find_all("h2", class_="story-title")]
    return headlines

def store_data_to_csv(scraped_data, headlines, key_statistics_data):
    df = pd.DataFrame(scraped_data)
    df['Headlines'] = headlines
    df.to_csv('scraped_data.csv', index=False)

    df_statistics = pd.DataFrame.from_dict(key_statistics_data, orient='index', columns=['Value'])
    df_statistics.to_csv('key_statistics.csv')

def convert_to_vector_database(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data.values())

    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(X)
    
    return tfidf_matrix

def run_queries_in_vector_database(queries, vector_database, data):
    query_vectors = TfidfVectorizer().fit_transform(queries)
    cosine_similarities = cosine_similarity(query_vectors, vector_database)
    
    query_results = []
    for i, query in enumerate(queries):
        relevant_index = cosine_similarities[i].argsort()[-1]
        relevant_data = data.iloc[relevant_index]
        query_results.append(relevant_data)
    
    return query_results

def text_summarization(text):
    sentences = text.split('.')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(X)
    scores = cosine_similarity(tfidf_matrix, tfidf_matrix)
    highest_score_index = np.argmax(np.mean(scores, axis=1))
    summary = sentences[highest_score_index]
    return summary

def generate_report(data, summaries):
    report = ""
    for i, row in data.iterrows():
        report += f"Data {i+1}:\n"
        report += f"{row}\n"
        report += f"Summary:\n{summaries[i]}\n\n"
    
    return report

if __name__ == "__main__":
    industry, market_cap, industry_weight, top_5_companies = scrape_company_info()
    top_5_companies_data = scrape_company_details(top_5_companies)
    key_statistics_data = scrape_key_statistics()
    headlines = scrape_news_headlines()

    scraped_data = {
        "Industry": [industry],
        "Market Cap": [market_cap],
        "Industry Weight": [industry_weight],
        "Top 5 Companies": [", ".join(top_5_companies)],
        "Headlines": ["; ".join(headlines)],
        **key_statistics_data
    }
    store_data_to_csv(scraped_data, headlines, key_statistics_data)

    vector_database = convert_to_vector_database(scraped_data)

    queries = ["query1", "query2", "query3", "query4"]

    query_results = run_queries_in_vector_database(queries, vector_database, pd.DataFrame.from_dict(scraped_data))

    summaries = [text_summarization(text) for text in scraped_data["Headlines"]]

    report = generate_report(pd.DataFrame.from_dict(scraped_data), summaries)

    print(report)
