# Google search result

Application that can scan the top 100 google search results and perform sentiment and emotion analysis on the webpages and then displays the results in categories and graphs.

Application uses flask for backend for creating API to interact with. Uses trafilatura and beautifulsoup4 to web scrape. Tensorflow, Keras and nltk to create models to analyse the data.

## to install

### React

cd client

npm i

npm start

### backend

cd backend

docker pull duffcorp/aibackend

docker run -p 5000:5000 duffcorp/aibackend

## How to use

To use the application start by using the above commands to install and start both the front end and backend.

- Once loaded you will be presented with a search bar to type into.

- Once a search has been entered you can select enter or click the search icon to initiate the search.

- Once a search is initiated do not reload the page, It will show its loading status below the search bar.

- Once loaded the application will display the analyzed results.

- The application can switch between emotion and sentiment analyses by clicking the button at the top of the results.

### Title

The title displays:

- The search the user made.
- The amount of pages analyzed.
- The overall sentiment/emotion.
- Radar chart with the most used keywords.

### Bar chart

The bar chart displays the total number of each category of sentiment or emotion. If there is no results for the category it is not displayed.

### Most results

The most results section displays the highest scored page for each category. If the category has no data is is not displayed. The text can be expanded or collapsed using the "read more"/"read less" button at the end.

### All pages

The all pages section contains each page analyzed categorized by sentiment or emotion. If the category has no data is is not displayed. The text for each page can be expanded or collapsed using the "read more"/"read less" button at the end.
