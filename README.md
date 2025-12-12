<!--
README for the Customer Service Chat Sentiment Analysis project.

This document explains the purpose, dataset, pipeline and usage of
the project. It also provides instructions on how to set up the
environment and reproduce the results on your own machine.  All
code in this repository is self‑contained and does not rely on any
pre‑trained weights.
-->

# Customer Service Chat Sentiment Analysis

This project implements a complete development pipeline for
predicting customer satisfaction from chat transcripts.  It was
designed as part of an AI model development assignment and covers
data loading, preprocessing, model design, training, evaluation
and result visualisation.  The final deliverable is a simple yet
effective sentiment classifier trained from scratch on chat logs.

## Dataset

The code expects a CSV file containing customer service chat
interactions.  Each row should correspond to a single
transaction and include the following columns:

| Column                   | Description                                                                           |
|-------------------------|---------------------------------------------------------------------------------------|
| `Transaction Start Date` | Timestamp when the chat started                                                      |
| `Agent`                 | Identifier of the customer service agent                                             |
| `Chat Duration`         | Duration of the chat in seconds                                                      |
| `Teams`                 | Team or department handling the conversation                                         |
| `Session Name`          | Unique identifier for the session                                                    |
| `Chat Closed By`        | Who ended the chat (agent or customer)                                               |
| `Interactive Chat`      | Indicator whether the conversation was interactive                                   |
| `Browser`               | Browser used by the customer                                                         |
| `Operating System`      | Operating system used by the customer                                               |
| `Geo`                   | Customer location                                                                    |
| `Response Time of Agent`| Average response time of the agent                                                   |
| `Response Time of Visitor`| Average response time of the customer                                             |
| `Transaction End Date`  | Timestamp when the chat ended                                                        |
| `Customer Rating`       | Rating (typically 1–5) provided by the customer after the interaction                |
| `Customer Comment`      | Free‑text comment left by the customer                                              |
| `Transferred Chat`      | Indicator whether the chat was transferred to another agent                          |
| `Customer Wait Time`    | Time the customer waited before receiving an agent response                          |

Only the **Customer Comment** field and the **Customer Rating**
field are used by default for the sentiment classification task.

The dataset provided for this project is an Excel file named
`Chat_Team_CaseStudy FINAL.xlsx`.  Ratings are integers between 0
and 10 or blank strings.  In this implementation ratings of
**8, 9 and 10** are treated as **positive**, ratings of **0, 1 and 2**
are treated as **negative**, and ratings in the range 3–7 or
blank entries are considered neutral and excluded from training.

You can adjust this mapping in `src/preprocess.py` if you wish to
handle more classes or use a different threshold.

Place your downloaded dataset file in the `data` directory.  By
default the code expects a file called `Chat_Team_CaseStudy FINAL.xlsx`
in `data/`.  The loader automatically detects whether the file is
CSV or Excel.

## Project Structure

```
customer_sentiment_project/
├── data/
│   └── customer_chat_data.csv            # your chat dataset (not included)
├── reports/
│   └── (generated evaluation outputs)
├── src/
│   ├── __init__.py
│   ├── config.py                         # configuration values
│   ├── data_loader.py                    # functions to load the dataset
│   ├── preprocess.py                     # text cleaning and label mapping
│   ├── model.py                          # model definition
│   ├── train.py                          # training script
│   ├── evaluate.py                       # evaluation script
│   └── inference.py                      # quick prediction script
├── requirements.txt                      # Python dependencies
└── README.md                             # project documentation
```

## Getting Started

1. **Clone the repository** and install the required packages.  A
   virtual environment is recommended:

   ```bash
   git clone <this repo>
   cd customer_sentiment_project
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Prepare the dataset.**  Download the chat data CSV (e.g. the
   *Customer Service Chat Data 30k Rows* from Kaggle) and copy it
   into the `data` folder.  Make sure the file name matches the
   `DATA_FILE` value in `src/config.py`.

3. **Train the model** using the provided training script:

   ```bash
   python -m src.train
   ```

   This will load the dataset, clean the text, split it into
   training and test sets, vectorise the text using TF‑IDF, train a
   logistic regression classifier, and save the model and TF‑IDF
   vectoriser to disk under `reports/models/`.

4. **Evaluate the model** using the evaluation script:

   ```bash
   python -m src.evaluate
   ```

   The script will load the saved model and vectoriser, run
   predictions on the test set and output a classification report
   along with a confusion matrix image and a metrics summary in
   `reports/`.

5. **Run inference** on new sentences via the interactive
   inference script:

   ```bash
   python -m src.inference "I waited a long time to get help and the agent was rude."
   ```

   This command loads the trained model and prints the predicted
   sentiment for the provided sentence.

## Methodology

The project uses a classic NLP pipeline:

1. **Data Cleaning:** free‑text comments are normalised by
   converting them to lower case, removing punctuation and numbers,
   and filtering out stop words.  The rating is mapped to a
   discrete sentiment label.
2. **Feature Extraction:** the cleaned comments are converted into
   TF‑IDF feature vectors using `scikit‑learn`'s
   `TfidfVectorizer`.
3. **Model Training:** a logistic regression classifier (from
   `scikit‑learn`) is trained on the training set.  No
   pre‑trained embeddings or weights are used.
4. **Evaluation:** we compute accuracy, precision, recall and
   F‑score on the held‑out test set.  A confusion matrix is plotted
   to visualise prediction errors.
5. **Saving/Loading:** the fitted vectoriser and model are
   serialized using `joblib` so they can be reused without
   retraining.

This approach provides a solid baseline for text sentiment
classification.  Despite its simplicity, logistic regression with
TF‑IDF features often performs competitively on short text
classification tasks.

## License

The code in this repository is provided for educational purposes.
You are free to modify and distribute it provided you include
appropriate attribution.
