from flask import Flask, render_template, request
from sentiment import analyze_sentiment

import os
import sys
import time


# =========================================================
# WINDOWS UTF-8 TERMINAL FIX
# =========================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

READ_FILE = os.path.join(
    DATA_DIR,
    "read.txt"
)


# Make sure data directory exists
os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# =========================================================
# DEFAULT VALUES
# =========================================================

def default_data():

    return {

        "sentiment": None,

        "sentiment_score": 0,

        "sentiment_probabilities": {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        },

        "emotions": "No emotions detected",

        "emotion_count": {},

        "keywords": [],

        "statistics": {
            "words": 0,
            "characters": 0,
            "sentences": 0,
            "unique_words": 0
        },

        "sentence_analysis": [],

        "chart": False,

        "user_text": "",

        "chart_version": int(
            time.time() * 1000
        )
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    data = default_data()

    return render_template(
        "index.html",
        **data
    )


# =========================================================
# ANALYZE TEXT
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # -----------------------------------------------------
    # Get text
    # -----------------------------------------------------

    text = request.form.get(
        "text",
        ""
    ).strip()


    # -----------------------------------------------------
    # Empty input
    # -----------------------------------------------------

    if not text:

        data = default_data()

        return render_template(
            "index.html",
            **data
        )


    # -----------------------------------------------------
    # Save text to read.txt
    # -----------------------------------------------------

    try:

        with open(
            READ_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

    except Exception as error:

        print(
            "Error writing read.txt:",
            str(error)
        )

        data = default_data()

        data["user_text"] = text

        return render_template(
            "index.html",
            **data
        )


    # =====================================================
    # RUN SENTIMENT ANALYSIS
    # =====================================================

    try:

        result = analyze_sentiment()

    except Exception as error:

        print("\n================================")
        print("ANALYSIS ERROR")
        print("================================")

        safe_error = (
            str(error)
            .encode(
                "ascii",
                "replace"
            )
            .decode("ascii")
        )

        print(
            safe_error
        )

        print("================================\n")


        data = default_data()

        data["user_text"] = text

        data["sentiment"] = (
            "Analysis Error"
        )

        data["emotions"] = (
            "Unable to detect emotions"
        )

        return render_template(
            "index.html",
            **data
        )


    # =====================================================
    # EXTRACT RESULTS
    # =====================================================

    sentiment = result.get(
        "sentiment",
        "Neutral"
    )


    sentiment_score = result.get(
        "sentiment_score",
        0
    )


    sentiment_probabilities = result.get(

        "sentiment_probabilities",

        {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
    )


    emotions = result.get(
        "emotions",
        "No emotions detected"
    )


    emotion_count = result.get(
        "emotion_count",
        {}
    )


    keywords = result.get(
        "keywords",
        []
    )


    statistics = result.get(
        "statistics",

        {
            "words": 0,
            "characters": 0,
            "sentences": 0,
            "unique_words": 0
        }
    )


    sentence_analysis = result.get(
        "sentence_analysis",
        []
    )


    # =====================================================
    # SAFETY CHECK
    # =====================================================

    if not isinstance(
        emotion_count,
        dict
    ):

        emotion_count = {}


    # =====================================================
    # DETERMINE WHETHER CHART EXISTS
    # =====================================================

    chart = bool(
        emotion_count
    )


    # =====================================================
    # UNIQUE CHART VERSION
    # =====================================================

    chart_version = int(
        time.time() * 1000
    )


    # =====================================================
    # SAFE TERMINAL OUTPUT
    # =====================================================

    safe_sentiment = (
        str(sentiment)
        .encode(
            "ascii",
            "replace"
        )
        .decode("ascii")
    )


    safe_emotions = (
        str(emotions)
        .encode(
            "ascii",
            "replace"
        )
        .decode("ascii")
    )


    print("\n========================================")
    print("SENTIMENT ANALYSIS RESULT")
    print("========================================")

    print(
        "Sentiment:",
        safe_sentiment
    )

    print(
        "Score:",
        sentiment_score
    )

    print(
        "Detected Emotions:",
        safe_emotions
    )

    print(
        "Emotion Count:",
        emotion_count
    )

    print(
        "Statistics:",
        statistics
    )

    print(
        "Keywords:",
        keywords
    )

    print("========================================\n")


    # =====================================================
    # SEND DATA TO HTML
    # =====================================================

    return render_template(

        "index.html",

        user_text=text,

        sentiment=sentiment,

        sentiment_score=sentiment_score,

        sentiment_probabilities=(
            sentiment_probabilities
        ),

        emotions=emotions,

        emotion_count=emotion_count,

        keywords=keywords,

        statistics=statistics,

        sentence_analysis=(
            sentence_analysis
        ),

        chart=chart,

        chart_version=chart_version
    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True
    )