
import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"

response = requests.get(url, timeout=15)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

books = []

for book in soup.select("article.product_pod"):
    title = book.h3.a["title"]

    price = book.select_one(".price_color").get_text(strip=True)

    availability = book.select_one(
        ".availability"
    ).get_text(strip=True)

    rating_tag = book.select_one(".star-rating")
    rating = rating_tag["class"][1]

    books.append({
        "Title": title,
        "Price": price,
        "Rating": rating,
        "Availability": availability
    })

df = pd.DataFrame(books)

df.to_csv("books_dataset.csv", index=False)

print("Web scraping completed successfully!")
print(df.head())
print("Total books scraped:", len(df))
print("Dataset saved as books_dataset.csv")
