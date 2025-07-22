import polars as pl
import pycountry
from datetime import datetime

emotion_anchors = [
    # Labels from google-research-datasets/go_emotions, minus "neutral"
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",

    # Custom
    "anxiety",
    "anticipation",
    "awe",
    "boredom",
    "contempt",
    "contentment",
    "compassion",
    "despair",
    "ecstasy",
    "envy",
    "frustration",
    "guilt",
    "happiness",
    "hate",
    "hope",
    "humiliation",
    "jealousy",
    "loneliness",
    "nostalgia",
    "panic",
    "pity",
    "rage",
    "regret",
    "resentment",
    "shame",
    "sympathy",
    "trust",
]

intent_anchors = [
    "to analyze, create, debug, or update a code snippet",
    "to be amused or entertained",
    "to brainstorm ideas or come up with a plan or strategy",
    "to complete some errand or chore by using the internet",
    "to compose a message or a textual document",
    "to engage in a casual conversation or get some casual advice",
    "to get emotional support",
    "to learn more about a subject or concept through explanation",
    "to provide data for analysis",
    "to solve a logical or mathematical problem",
    "to summarize or translate some text",
]

location_anchors = ([f"{country.name}" for country in pycountry.countries]
                    + [f"{subdivision.name}, {subdivision.country.name}" for subdivision in pycountry.subdivisions])

now_year = datetime.now().year

temporal_anchors = ([f"the {era} Era" for era in ("Paleozoic", "Mesozoic", "Cenozoic")]
                    + [f"the {period} Period" for period in ("Cambrian", "Ordovician", "Silurian", "Devonian",
                                                             "Carboniferous", "Permian", "Triassic", "Jurassic",
                                                             "Cretaceous", "Paleogene", "Neogene", "Quaternary")]
                    + [f"the {epoch} Epoch" for epoch in ("Paleocene", "Eocene", "Oligocene", "Miocene",
                                                          "Pliocene", "Pleistocene", "Holocene")]
                    + [f"{millennia + 1000}-{millennia} BC" for millennia in list(range(3000, 10000, 1000))]
                    + [f"{century + 100}-{century} BC" for century in list(range(0, 3000, 100))]
                    + [f"{century}-{century + 100} AD" for century in list(range(0, 1800, 100))]
                    + [f"AD {decade}s" for decade in list(range(1000, now_year, 10))]
                    + [f"AD {yr}" for yr in list(range(1000, now_year))])

topic_anchors = [
    "abilities",
    "achievements",
    "aging",
    "animals",
    "aviation",
    "baseball",
    "basketball",
    "biology",
    "careers",
    "chemistry",
    "children",
    "cloud computing",
    "color",
    "computer programming",
    "computers",
    "courtship",
    "crime",
    "death",
    "economics",
    "entertainment",
    "events",
    "family",
    "fashion",
    "finance",
    "fitness",
    "food",
    "football",
    "games",
    "gardens",
    "geography",
    "goals",
    "gossip",
    "government",
    "history",
    "hygiene",
    "ideals",
    "illness",
    "injury",
    "jokes",
    "law",
    "lifestyle",
    "linguistics",
    "literature",
    "logistics",
    "marriage",
    "mathematics",
    "medicine",
    "mental health",
    "movies",
    "music",
    "nature",
    "news",
    "nightlife",
    "nursing",
    "pain",
    "parents",
    "pets",
    "phenomena",
    "philosophy",
    "physical health",
    "physics",
    "plants",
    "poetry",
    "politics",
    "preferences",
    "psychology",
    "qualities",
    "relationships",
    "religion",
    "scenery",
    "smells",
    "sports",
    "tactile sensations",
    "teamwork",
    "techniques",
    "technology",
    "textures",
    "travel",
    "vehicles",
    "visual appearance",
    "war",
    "work",
]

wiki_df = pl.read_csv("data/enwiki-latest-fa-ga-titles.tsv", separator="\t", quote_char=None)
fa_wiki_anchors = (wiki_df
                   .filter(pl.col("quality") == 'FA')
                   .select(pl.concat_str(['url_title', 'short_description'], separator=": "))
                   .drop_nulls()
                   .to_series()
                   .to_list())
ga_wiki_anchors = (wiki_df
                   .filter(pl.col("quality") == 'GA')
                   .select(pl.concat_str(['url_title', 'short_description'], separator=": "))
                   .drop_nulls()
                   .to_series()
                   .to_list())
#wiki_anchors = fa_wiki_anchors + ga_wiki_anchors  # Start with higher quality pages
wiki_anchors = []
