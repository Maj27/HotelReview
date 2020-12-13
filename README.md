# HotelReview

In this small project, I am showing how to build multiclass text classification machine learning (ML) models to classify hotel reviews into a five-star scale. The goal of this project is to build ML classifiers that predict the ratings of the reviews. Different classification models were trained on a big sample of hotel reviews. The reviews were preprocessed by removing punctuations and nonalphabetic characters and stop words. The features used in the final models were mainly n-grams of term frequency-inverse document frequency (tf-idf) of word lemmas. The accuracy obtained with the best model reached 75%.


To run the code you can retrain the models and explore the data using the Hotel Review Sentiment Classification.ipynp. If you want to test the models you can use the Hotel Review Testing.ipynp which loads the trained models (model.zip) and test them on the dev and test sets (offline classification). There is also the Hotel_Review_app.py which uses Flask to deploy the models. A request can be sent using the request.py where the payload can be a single hotel review.
