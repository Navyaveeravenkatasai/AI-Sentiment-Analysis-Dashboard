import os
import re
import string
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# =========================================================
# PROJECT PATH
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
# CREATE REQUIRED DIRECTORIES
# =========================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# =========================================================
# LOAD EMOTIONS
# =========================================================

def load_emotions():

    emotions = {}


    if not os.path.exists(
        EMOTIONS_FILE
    ):

        print(
            "ERROR: emotions.txt not found:"
        )

        print(
            EMOTIONS_FILE
        )

        return emotions


    try:

        with open(
            EMOTIONS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()


                # Skip empty lines

                if not line:
                    continue


                # Skip comments

                if line.startswith("#"):
                    continue


                # Check format

                if ":" not in line:
                    continue


                word, emotion = line.split(
                    ":",
                    1
                )


                word = word.strip().lower()

                emotion = emotion.strip().lower()


                if word and emotion:

                    emotions[word] = emotion


    except Exception as error:

        print(
            "Error reading emotions.txt:"
        )

        print(
            str(error)
        )


    print(
        "Emotion dictionary loaded:",
        len(emotions),
        "words"
    )


    return emotions


# =========================================================
# CREATE EMOTION CHART
# =========================================================

def create_emotion_chart(emotion_count):

    import numpy as np

    # =====================================================
    # CREATE FIGURE
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )


    # =====================================================
    # EMPTY GRAPH
    # =====================================================

    if not emotion_count:

        ax.text(

            0.5,

            0.5,

            "No emotions detected",

            ha="center",

            va="center",

            fontsize=20,

            fontweight="bold"

        )

        ax.axis("off")


    else:

        # =================================================
        # SORT EMOTIONS
        # =================================================

        sorted_emotions = sorted(

            emotion_count.items(),

            key=lambda item:
                item[1],

            reverse=True

        )


        emotion_names = [

            item[0].title()

            for item
            in sorted_emotions

        ]


        emotion_values = [

            item[1]

            for item
            in sorted_emotions

        ]


        # =================================================
        # POSITIONS
        # =================================================

        x = np.arange(
            len(emotion_names)
        )


        # =================================================
        # GRADIENT-LIKE BAR HEIGHT
        # =================================================

        bars = ax.bar(

            x,

            emotion_values,

            width=0.62,

            edgecolor="white",

            linewidth=0.8,

            alpha=0.9

        )


        # =================================================
        # TITLE
        # =================================================

        ax.set_title(

            "Emotion Distribution",

            fontsize=21,

            fontweight="bold",

            pad=20

        )


        # =================================================
        # AXIS LABELS
        # =================================================

        ax.set_xlabel(

            "Detected Emotions",

            fontsize=12,

            labelpad=12

        )


        ax.set_ylabel(

            "Frequency",

            fontsize=12,

            labelpad=12

        )


        # =================================================
        # X AXIS
        # =================================================

        ax.set_xticks(x)

        ax.set_xticklabels(

            emotion_names,

            rotation=35,

            ha="right",

            fontsize=10

        )


        # =================================================
        # VALUE LABELS
        # =================================================

        for bar, value in zip(

            bars,

            emotion_values

        ):

            ax.text(

                bar.get_x()
                + bar.get_width() / 2,

                value + 0.05,

                str(value),

                ha="center",

                va="bottom",

                fontsize=11,

                fontweight="bold"

            )


        # =================================================
        # GRID
        # =================================================

        ax.grid(

            axis="y",

            linestyle="--",

            alpha=0.25

        )


        # =================================================
        # REMOVE TOP/RIGHT BORDER
        # =================================================

        ax.spines[
            "top"
        ].set_visible(False)


        ax.spines[
            "right"
        ].set_visible(False)


        # =================================================
        # PADDING
        # =================================================

        ax.margins(
            x=0.05
        )


    # =====================================================
    # LAYOUT
    # =====================================================

    plt.tight_layout()


    # =====================================================
    # SAVE
    # =====================================================

    plt.savefig(

        CHART_FILE,

        dpi=180,

        bbox_inches="tight",

        facecolor="white"

    )


    # =====================================================
    # CLOSE
    # =====================================================

    plt.close(fig)


    print(
        "Emotion chart created successfully."
    )

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    # -----------------------------------------------------
    # Emotions exist
    # -----------------------------------------------------

    if emotion_count:

        emotion_names = list(
            emotion_count.keys()
        )


        emotion_values = list(
            emotion_count.values()
        )


        bars = plt.bar(

            emotion_names,

            emotion_values
        )


        # Title

        plt.title(
            "Emotion Distribution",
            fontsize=18,
            fontweight="bold"
        )


        # Axis labels

        plt.xlabel(
            "Emotion",
            fontsize=12
        )


        plt.ylabel(
            "Count",
            fontsize=12
        )


        # Rotate labels

        plt.xticks(
            rotation=45,
            ha="right"
        )


        # Add values above bars

        for bar in bars:

            height = bar.get_height()

            plt.text(

                bar.get_x()
                + bar.get_width() / 2,

                height,

                str(
                    int(height)
                ),

                ha="center",

                va="bottom",

                fontsize=11

            )


        plt.grid(
            axis="y",
            alpha=0.2
        )


    # -----------------------------------------------------
    # No emotions
    # -----------------------------------------------------

    else:

        plt.text(

            0.5,

            0.5,

            "No emotions detected",

            horizontalalignment="center",

            verticalalignment="center",

            fontsize=18

        )


        plt.axis(
            "off"
        )


    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    plt.tight_layout()


    # -----------------------------------------------------
    # Save chart
    # -----------------------------------------------------

    plt.savefig(

        CHART_FILE,

        dpi=150,

        bbox_inches="tight"
    )


    # -----------------------------------------------------
    # Close figure
    # -----------------------------------------------------

    plt.close()


    print(
        "Emotion chart created:"
    )

    print(
        CHART_FILE
    )


# =========================================================
# ANALYZE SENTIMENT
# =========================================================

def analyze_sentiment():

    # =====================================================
    # READ TEXT
    # =====================================================

    if not os.path.exists(
        READ_FILE
    ):

        return {

            "sentiment": "Neutral 😐",

            "sentiment_score": 0,

            "sentiment_probabilities": {

                "positive": 0,

                "neutral": 0,

                "negative": 0

            },

            "emotions":
                "No emotions detected",

            "emotion_count":
                {},

            "keywords":
                [],

            "statistics": {

                "words": 0,

                "characters": 0,

                "sentences": 0,

                "unique_words": 0

            },

            "sentence_analysis":
                []

        }


    with open(

        READ_FILE,

        "r",

        encoding="utf-8"

    ) as file:

        text = file.read().strip()


    # =====================================================
    # EMPTY TEXT
    # =====================================================

    if not text:

        return {

            "sentiment": "Neutral 😐",

            "sentiment_score": 0,

            "sentiment_probabilities": {

                "positive": 0,

                "neutral": 0,

                "negative": 0

            },

            "emotions":
                "No emotions detected",

            "emotion_count":
                {},

            "keywords":
                [],

            "statistics": {

                "words": 0,

                "characters": 0,

                "sentences": 0,

                "unique_words": 0

            },

            "sentence_analysis":
                []

        }


    # =====================================================
    # LOWERCASE
    # =====================================================

    lower_text = text.lower()


    # =====================================================
    # EXTRACT WORDS
    # =====================================================

    words = re.findall(

        r"[a-zA-Z]+",

        lower_text
    )


    # =====================================================
    # LOAD EMOTION DICTIONARY
    # =====================================================

    emotion_dictionary = load_emotions()


    # =====================================================
    # DETECT EMOTIONS
    # =====================================================

    emotion_list = []

    detected_words = []


    for word in words:

        if word in emotion_dictionary:

            emotion = emotion_dictionary[word]


            emotion_list.append(
                emotion
            )


            detected_words.append(
                word
            )


    # =====================================================
    # COUNT EMOTIONS
    # =====================================================

    emotion_count = dict(

        Counter(
            emotion_list
        )
    )


    # =====================================================
    # DEBUG OUTPUT
    # =====================================================

    print("\n===================================")

    print(
        "EMOTION DETECTION"
    )

    print("===================================")


    print(
        "Detected words:",
        detected_words
    )


    print(
        "Detected emotions:",
        emotion_list
    )


    print(
        "Emotion count:",
        emotion_count
    )


    print(
        "===================================\n"
    )


    # =====================================================
    # CREATE GRAPH
    # =====================================================

    create_emotion_chart(
        emotion_count
    )


    # =====================================================
    # VADER SENTIMENT
    # =====================================================

    analyzer = SentimentIntensityAnalyzer()


    scores = analyzer.polarity_scores(
        text
    )


    positive = scores["pos"]

    neutral = scores["neu"]

    negative = scores["neg"]

    compound = scores["compound"]


    # =====================================================
    # DETERMINE SENTIMENT
    # =====================================================

    if compound >= 0.05:

        sentiment = "Positive 😊"

    elif compound <= -0.05:

        sentiment = "Negative 😞"

    else:

        sentiment = "Neutral 😐"


    # =====================================================
    # SENTIMENT PERCENTAGES
    # =====================================================

    sentiment_probabilities = {

        "positive":
            round(
                positive * 100,
                2
            ),

        "neutral":
            round(
                neutral * 100,
                2
            ),

        "negative":
            round(
                negative * 100,
                2
            )

    }


    # =====================================================
    # CLEAN TEXT
    # =====================================================

    cleaned_text = lower_text.translate(

        str.maketrans(

            "",

            "",

            string.punctuation
        )
    )


    # =====================================================
    # TOKENIZATION
    # =====================================================

    try:

        tokenized_words = word_tokenize(
            cleaned_text
        )

    except LookupError:

        tokenized_words = cleaned_text.split()


    # =====================================================
    # STOPWORDS
    # =====================================================

    try:

        stop_words = set(
            stopwords.words(
                "english"
            )
        )

    except LookupError:

        stop_words = set()


    final_words = [

        word

        for word in tokenized_words

        if word not in stop_words

        and word.isalpha()

    ]


    # =====================================================
    # KEYWORDS
    # =====================================================

    word_frequency = Counter(
        final_words
    )


    keywords = [

        word

        for word, count

        in word_frequency.most_common(
            10
        )

    ]


    # =====================================================
    # SENTENCE COUNT
    # =====================================================

    sentence_list = re.split(

        r"[.!?]+",

        text
    )


    sentence_list = [

        sentence.strip()

        for sentence
        in sentence_list

        if sentence.strip()

    ]


    # =====================================================
    # STATISTICS
    # =====================================================

    statistics = {

        "words":
            len(words),

        "characters":
            len(text),

        "sentences":
            len(sentence_list),

        "unique_words":
            len(
                set(words)
            )

    }


    # =====================================================
    # SENTENCE ANALYSIS
    # =====================================================

    sentence_analysis = []


    for sentence in sentence_list:

        sentence_score = (
            analyzer.polarity_scores(
                sentence
            )
        )


        sentence_compound = (
            sentence_score["compound"]
        )


        if sentence_compound >= 0.05:

            sentence_sentiment = (
                "Positive 😊"
            )

        elif sentence_compound <= -0.05:

            sentence_sentiment = (
                "Negative 😞"
            )

        else:

            sentence_sentiment = (
                "Neutral 😐"
            )


        sentence_analysis.append({

            "text":
                sentence,

            "sentiment":
                sentence_sentiment,

            "score":
                round(
                    sentence_compound,
                    3
                )

        })


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "sentiment":
            sentiment,

        "sentiment_score":
            round(
                compound,
                3
            ),

        "sentiment_probabilities":
            sentiment_probabilities,

        "emotions":

            ", ".join(
                emotion_list
            )

            if emotion_list

            else

            "No emotions detected",

        "emotion_count":
            emotion_count,

        "keywords":
            keywords,

        "statistics":
            statistics,

        "sentence_analysis":
            sentence_analysis

    }