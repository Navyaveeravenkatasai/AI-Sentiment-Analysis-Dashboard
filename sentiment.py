import os
import re
import string
import matplotlib

# =========================================================
# MATPLOTLIB BACKEND
# =========================================================

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from collections import Counter

import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


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

EMOTIONS_FILE = os.path.join(
    DATA_DIR,
    "emotions.txt"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

CHART_FILE = os.path.join(
    CHART_DIR,
    "emotions_chart.png"
)


# =========================================================
# NLTK DATA
# =========================================================

def download_nltk_data():
    """
    Download required NLTK resources if they
    are not already available.
    """

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon", "vader_lexicon")
    ]

    for path, package in resources:

        try:
            nltk.data.find(path)

        except LookupError:

            try:
                nltk.download(
                    package,
                    quiet=True
                )

            except Exception:
                pass


download_nltk_data()


# =========================================================
# SAFE TOKENIZER
# =========================================================

def safe_word_tokenize(text):

    try:

        return word_tokenize(text)

    except LookupError:

        try:
            nltk.download(
                "punkt",
                quiet=True
            )

            nltk.download(
                "punkt_tab",
                quiet=True
            )

            return word_tokenize(text)

        except Exception:

            return text.split()


# =========================================================
# SAFE SENTENCE TOKENIZER
# =========================================================

def safe_sent_tokenize(text):

    try:

        return sent_tokenize(text)

    except LookupError:

        try:
            nltk.download(
                "punkt",
                quiet=True
            )

            nltk.download(
                "punkt_tab",
                quiet=True
            )

            return sent_tokenize(text)

        except Exception:

            return re.split(
                r'(?<=[.!?])\s+',
                text
            )


# =========================================================
# LOAD STOPWORDS
# =========================================================

def get_stopwords():

    try:

        return set(
            stopwords.words("english")
        )

    except LookupError:

        try:

            nltk.download(
                "stopwords",
                quiet=True
            )

            return set(
                stopwords.words("english")
            )

        except Exception:

            return set()


# =========================================================
# LOAD EMOTION DICTIONARY
# =========================================================

def load_emotions():

    emotions = {}

    if not os.path.exists(
        EMOTIONS_FILE
    ):

        return emotions

    try:

        with open(
            EMOTIONS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                # -------------------------------------------------
                # Supported formats:
                #
                # happy joy
                # happy:joy
                # happy,joy
                # -------------------------------------------------

                if ":" in line:

                    word, emotion = (
                        line.split(
                            ":",
                            1
                        )
                    )

                elif "," in line:

                    word, emotion = (
                        line.split(
                            ",",
                            1
                        )
                    )

                else:

                    parts = line.split()

                    if len(parts) < 2:
                        continue

                    word = parts[0]
                    emotion = parts[1]

                word = word.strip().lower()
                emotion = emotion.strip().lower()

                if word and emotion:

                    emotions[word] = emotion

    except Exception as error:

        print(
            "Error reading emotions.txt:",
            str(error)
        )

    return emotions


# =========================================================
# LOAD VADER
# =========================================================

def get_analyzer():

    try:

        return SentimentIntensityAnalyzer()

    except LookupError:

        try:

            nltk.download(
                "vader_lexicon",
                quiet=True
            )

            return SentimentIntensityAnalyzer()

        except Exception:

            return None


# =========================================================
# SENTIMENT LABEL
# =========================================================

def get_sentiment_label(
    compound
):

    if compound >= 0.05:

        return "Positive"

    elif compound <= -0.05:

        return "Negative"

    return "Neutral"


# =========================================================
# SENTIMENT PROBABILITIES
# =========================================================

def calculate_sentiment_probabilities(
    scores
):

    positive = max(
        0,
        scores.get(
            "pos",
            0
        )
    )

    negative = max(
        0,
        scores.get(
            "neg",
            0
        )
    )

    neutral = max(
        0,
        scores.get(
            "neu",
            0
        )
    )

    total = (
        positive
        + negative
        + neutral
    )

    if total == 0:

        return {
            "positive": 0,
            "neutral": 100,
            "negative": 0
        }

    return {

        "positive": round(
            (positive / total) * 100,
            2
        ),

        "neutral": round(
            (neutral / total) * 100,
            2
        ),

        "negative": round(
            (negative / total) * 100,
            2
        )
    }


# =========================================================
# CREATE EMOTION CHART
# =========================================================

def create_emotion_chart(
    emotion_count
):

    if not emotion_count:

        return False

    try:

        os.makedirs(
            CHART_DIR,
            exist_ok=True
        )

        emotions = list(
            emotion_count.keys()
        )

        counts = list(
            emotion_count.values()
        )

        plt.figure(
            figsize=(10, 5)
        )

        bars = plt.bar(
            emotions,
            counts
        )

        plt.title(
            "Emotion Distribution",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel(
            "Emotions"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.grid(
            axis="y",
            alpha=0.2
        )

        # Add values above bars

        for bar, count in zip(
            bars,
            counts
        ):

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.tight_layout()

        plt.savefig(
            CHART_FILE,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        return True

    except Exception as error:

        print(
            "Chart creation error:",
            str(error)
        )

        try:
            plt.close()

        except Exception:
            pass

        return False


# =========================================================
# MAIN SENTIMENT ANALYSIS
# =========================================================

def analyze_sentiment():

    # =====================================================
    # READ TEXT
    # =====================================================

    if not os.path.exists(
        READ_FILE
    ):

        return {

            "sentiment": "Neutral",

            "sentiment_score": 0,

            "sentiment_probabilities": {
                "positive": 0,
                "neutral": 100,
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

            "sentence_analysis": []
        }

    try:

        with open(
            READ_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read().strip()

    except Exception as error:

        print(
            "Error reading text:",
            str(error)
        )

        text = ""


    # =====================================================
    # EMPTY TEXT
    # =====================================================

    if not text:

        return {

            "sentiment": "Neutral",

            "sentiment_score": 0,

            "sentiment_probabilities": {
                "positive": 0,
                "neutral": 100,
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

            "sentence_analysis": []
        }


    # =====================================================
    # INITIALIZE NLTK
    # =====================================================

    analyzer = get_analyzer()

    stop_words = get_stopwords()


    # =====================================================
    # VADER ANALYSIS
    # =====================================================

    if analyzer:

        scores = analyzer.polarity_scores(
            text
        )

    else:

        scores = {
            "compound": 0,
            "pos": 0,
            "neu": 1,
            "neg": 0
        }


    compound = round(
        scores.get(
            "compound",
            0
        ),
        4
    )


    sentiment = get_sentiment_label(
        compound
    )


    # =====================================================
    # SENTIMENT PROBABILITIES
    # =====================================================

    sentiment_probabilities = (
        calculate_sentiment_probabilities(
            scores
        )
    )


    # =====================================================
    # WORD TOKENIZATION
    # =====================================================

    tokens = safe_word_tokenize(
        text.lower()
    )


    # =====================================================
    # CLEAN WORDS
    # =====================================================

    clean_words = []

    for word in tokens:

        word = word.strip(
            string.punctuation
        )

        if not word:
            continue

        if not re.match(
            r"^[a-zA-Z]+$",
            word
        ):
            continue

        clean_words.append(
            word
        )


    # =====================================================
    # STATISTICS
    # =====================================================

    word_count = len(
        clean_words
    )

    character_count = len(
        text
    )

    sentences = safe_sent_tokenize(
        text
    )

    sentence_count = len(
        [
            sentence
            for sentence in sentences
            if sentence.strip()
        ]
    )

    unique_word_count = len(
        set(clean_words)
    )


    statistics = {

        "words": word_count,

        "characters": character_count,

        "sentences": sentence_count,

        "unique_words": unique_word_count
    }


    # =====================================================
    # EMOTION DETECTION
    # =====================================================

    emotion_dictionary = (
        load_emotions()
    )

    emotion_count = Counter()

    detected_emotions = []


    for word in clean_words:

        if word in emotion_dictionary:

            emotion = (
                emotion_dictionary[word]
            )

            emotion_count[
                emotion
            ] += 1

            if emotion not in detected_emotions:

                detected_emotions.append(
                    emotion
                )


    # =====================================================
    # EMOTIONS TEXT
    # =====================================================

    if detected_emotions:

        emotions = ", ".join(
            emotion.title()
            for emotion in detected_emotions
        )

    else:

        emotions = (
            "No emotions detected"
        )


    # =====================================================
    # KEYWORDS
    # =====================================================

    meaningful_words = [

        word

        for word in clean_words

        if word not in stop_words

        and len(word) > 2
    ]


    word_frequency = Counter(
        meaningful_words
    )


    keywords = [

        word

        for word, count
        in word_frequency.most_common(10)
    ]


    # =====================================================
    # SENTENCE ANALYSIS
    # =====================================================

    sentence_analysis = []


    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue


        if analyzer:

            sentence_scores = (
                analyzer.polarity_scores(
                    sentence
                )
            )

        else:

            sentence_scores = {
                "compound": 0
            }


        sentence_score = round(
            sentence_scores.get(
                "compound",
                0
            ),
            4
        )


        sentence_sentiment = (
            get_sentiment_label(
                sentence_score
            )
        )


        sentence_analysis.append({

            "text": sentence,

            "sentiment":
                sentence_sentiment,

            "score":
                sentence_score
        })


    # =====================================================
    # CREATE CHART
    # =====================================================

    create_emotion_chart(
        dict(emotion_count)
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "sentiment":
            sentiment,

        "sentiment_score":
            compound,

        "sentiment_probabilities":
            sentiment_probabilities,

        "emotions":
            emotions,

        "emotion_count":
            dict(emotion_count),

        "keywords":
            keywords,

        "statistics":
            statistics,

        "sentence_analysis":
            sentence_analysis
    }